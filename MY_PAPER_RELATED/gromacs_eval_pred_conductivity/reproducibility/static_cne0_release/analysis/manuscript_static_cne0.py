#!/usr/bin/env python
"""Recalculate static cNE0 with the protocol stated in the manuscript and SI.

The only association definition implemented here is a replica/trajectory RDF
first-minimum cutoff followed by the Li-TFSI coordination-state filter: a
Li-TFSI edge is retained when at least two oxygen atoms of that TFSI are within
the cutoff in the same frame. No persistence or RDF-peak alternative is
implemented in this script.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import math
import warnings

import numpy as np
import pandas as pd

from persistence_sweep_and_hybrid import (
    add_pop_from_codes,
    load_valid_metadata,
    pop_descriptors,
    safe_log10_ratio,
    sigma_from_pop,
)


BASE = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_BASE = Path("../eval_top10_bottom10_stratified100")


@dataclass(frozen=True)
class Sample:
    dataset: str
    trajectory_id: int
    replica: int
    replica_id: str
    stage: str
    sample_group: str
    selection_reference_conductivity_S_cm: float
    sigma_NE_S_cm: float
    D_Li_cm2s: float
    D_TFSI_cm2s: float
    V_nm3: float
    N_LI: int
    N_TFSI: int
    gro: Path
    xtc: Path
    rdf_first_minimum_nm: float


def positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def read_replica_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    row = pd.read_csv(path).iloc[0]

    def number(column: str) -> float:
        return float(pd.to_numeric(row.get(column, np.nan), errors="coerce"))

    return {
        "sigma_NE_S_cm": number("sigma_NE_htpmd_S_cm"),
        "D_Li_cm2s": number("D_Li_cm2s"),
        "D_TFSI_cm2s": number("D_an_cm2s"),
        "V_nm3": number("V_nm3"),
        "N_LI": int(number("N_LI")),
        "N_TFSI": int(number("N_AN")),
    }


def load_generated_samples(base: Path) -> list[Sample]:
    result_root = base / "github_results" / "latest_notebook_manifest_60"
    run_results = pd.read_csv(result_root / "run_results.csv")
    rdf = pd.read_csv(
        base / "results" / "rdf_cutoff_latest60_6groups" / "replica_rdf_peak_summary.csv"
    )
    if len(run_results) != 60 or run_results["Trajectory ID"].nunique() != 60:
        raise RuntimeError("Expanded generated manifest must contain 60 candidates")

    rdf = rdf[rdf["rdf_status"].eq("ok")].copy()
    rdf_by_key = {
        (int(row["traj_id"]), int(row["replica"])): float(
            pd.to_numeric(row["Li_TFSI_O_first_min_nm"], errors="coerce")
        )
        for _, row in rdf.iterrows()
    }
    samples: list[Sample] = []
    for _, row in run_results.iterrows():
        trajectory_id = int(row["Trajectory ID"])
        sigma_ref = float(pd.to_numeric(row["CONDUCTIVITY"], errors="coerce"))
        for replica in (1, 2, 3):
            key = (trajectory_id, replica)
            cutoff = rdf_by_key.get(key, np.nan)
            stage = "production" if replica == 1 else f"production_rep{replica}"
            run = base / "runs" / f"Traj_{trajectory_id}"
            stage_dir = run / "md" / stage
            gro = stage_dir / f"{stage}.gro"
            xtc = stage_dir / f"{stage}.xtc"
            summary = run / "analysis" / f"replica_{replica}" / "conductivity_summary_htpmd_ref.csv"
            if not positive(cutoff):
                raise RuntimeError(f"Missing first-minimum cutoff for generated {key}")
            for path in (gro, xtc, summary):
                if not path.exists():
                    raise FileNotFoundError(path)
            values = read_replica_summary(summary)
            samples.append(
                Sample(
                    dataset="generated",
                    trajectory_id=trajectory_id,
                    replica=replica,
                    replica_id=f"Traj_{trajectory_id}_rep{replica}",
                    stage=stage,
                    sample_group=str(row["sample_group"]),
                    selection_reference_conductivity_S_cm=sigma_ref,
                    gro=gro,
                    xtc=xtc,
                    rdf_first_minimum_nm=cutoff,
                    **values,
                )
            )
    keys = {(sample.trajectory_id, sample.replica) for sample in samples}
    if len(samples) != 180 or len(keys) != 180:
        raise RuntimeError(f"Expected 180 unique generated replicas, found {len(samples)}")
    return samples


def load_reference_samples(base: Path) -> list[Sample]:
    rdf = pd.read_csv(
        base
        / "results"
        / "rdf_all_trajs_li_tfsiO_li_tfsiN_gmx"
        / "rdf_first_minima_summary.csv"
    )
    rdf_by_id = {
        int(row["traj_id"]): float(
            pd.to_numeric(row["Li_TFSI_O_first_min_nm"], errors="coerce")
        )
        for _, row in rdf.iterrows()
    }
    samples: list[Sample] = []
    for meta in load_valid_metadata(base):
        cutoff = rdf_by_id.get(meta.traj_id, np.nan)
        if not positive(cutoff):
            raise RuntimeError(f"Missing first-minimum cutoff for reference Traj_{meta.traj_id}")
        samples.append(
            Sample(
                dataset="reference_reassessment",
                trajectory_id=meta.traj_id,
                replica=1,
                replica_id=f"Traj_{meta.traj_id}_rep1",
                stage="production",
                sample_group=meta.group,
                selection_reference_conductivity_S_cm=meta.sigma_ref,
                sigma_NE_S_cm=meta.sigma_ne,
                D_Li_cm2s=meta.d_li_cm2s,
                D_TFSI_cm2s=meta.d_an_cm2s,
                V_nm3=meta.v_nm3,
                N_LI=meta.n_li,
                N_TFSI=meta.n_an,
                gro=meta.gro,
                xtc=meta.xtc,
                rdf_first_minimum_nm=cutoff,
            )
        )
    if len(samples) != 108 or len({sample.trajectory_id for sample in samples}) != 108:
        raise RuntimeError(f"Expected 108 unique completed references, found {len(samples)}")
    return samples


def compute_one(
    sample: Sample,
    temperature_K: float,
    target_frame_spacing_ps: float,
    max_cluster_size: int,
) -> dict[str, object]:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    u = mda.Universe(str(sample.gro), str(sample.xtc))
    li = u.select_atoms("resname LI")
    oxygen = u.select_atoms("resname TFSI and name O*")
    nitrogen = u.select_atoms("resname TFSI and name N*")
    if (len(li), len(nitrogen)) != (sample.N_LI, sample.N_TFSI):
        raise RuntimeError(
            f"{sample.replica_id}: selection counts LI/TFSI "
            f"{len(li)}/{len(nitrogen)} != summary {sample.N_LI}/{sample.N_TFSI}"
        )
    if len(oxygen) != 4 * len(nitrogen):
        raise RuntimeError(
            f"{sample.replica_id}: expected four TFSI oxygen atoms per anion, "
            f"found O/TFSI={len(oxygen)}/{len(nitrogen)}"
        )

    saved_spacing_ps = float(u.trajectory.dt)
    stride = int(round(target_frame_spacing_ps / saved_spacing_ps))
    if stride < 1 or not math.isclose(
        stride * saved_spacing_ps, target_frame_spacing_ps, rel_tol=0, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"{sample.replica_id}: saved spacing {saved_spacing_ps:g} ps cannot produce "
            f"the required {target_frame_spacing_ps:g} ps analysis spacing"
        )

    anion_by_resindex = {
        int(resindex): index for index, resindex in enumerate(nitrogen.resindices)
    }
    oxygen_to_anion = np.asarray(
        [anion_by_resindex[int(atom.resindex)] for atom in oxygen], dtype=np.int32
    )
    n_li = len(li)
    n_anion = len(nitrogen)
    cutoff_A = sample.rdf_first_minimum_nm * 10.0
    shape = (
        max(2, min(max_cluster_size, n_li + 1)),
        max(2, min(max_cluster_size, n_anion + 1)),
    )
    population = np.zeros(shape, dtype=np.float64)
    largest_observed = 0
    frame_count = 0
    first_time_ps = np.nan
    last_time_ps = np.nan

    for ts in u.trajectory[::stride]:
        if frame_count == 0:
            first_time_ps = float(ts.time)
        last_time_ps = float(ts.time)
        contacts = capped_distance(
            li.positions,
            oxygen.positions,
            cutoff_A,
            box=ts.dimensions,
            return_distances=False,
        )
        if contacts.size:
            li_index = contacts[:, 0].astype(np.int32, copy=False)
            oxygen_index = contacts[:, 1].astype(np.int32, copy=False)
            pair_codes = li_index * n_anion + oxygen_to_anion[oxygen_index]
            unique_codes, oxygen_counts = np.unique(pair_codes, return_counts=True)
            retained_codes = unique_codes[oxygen_counts >= 2].astype(np.int32, copy=False)
        else:
            retained_codes = np.empty(0, dtype=np.int32)
        largest_observed = max(
            largest_observed,
            add_pop_from_codes(population, retained_codes, n_li, n_anion),
        )
        frame_count += 1

    if frame_count != 50001:
        raise RuntimeError(
            f"{sample.replica_id}: expected 50,001 frames at 1 ps, found {frame_count}"
        )
    if not (
        math.isclose(first_time_ps, 0.0, abs_tol=1e-6)
        and math.isclose(last_time_ps, 50000.0, abs_tol=1e-3)
    ):
        raise RuntimeError(
            f"{sample.replica_id}: expected 0-50,000 ps, found "
            f"{first_time_ps:g}-{last_time_ps:g} ps"
        )

    population /= float(frame_count)
    sigma = sigma_from_pop(
        population,
        sample.D_Li_cm2s,
        sample.D_TFSI_cm2s,
        sample.V_nm3,
        temperature_K,
        max_cluster_size,
    )
    descriptors = pop_descriptors(population, n_li, n_anion)
    descriptors["largest_cluster_size"] = max(
        int(descriptors["largest_cluster_size"]), largest_observed
    )
    return {
        "dataset": sample.dataset,
        "trajectory_id": sample.trajectory_id,
        "replica": sample.replica,
        "replica_id": sample.replica_id,
        "stage": sample.stage,
        "sample_group": sample.sample_group,
        "protocol": "manuscript_static_cNE0",
        "cutoff_source": "Li_TFSI_O_RDF_first_minimum",
        "cutoff_nm": sample.rdf_first_minimum_nm,
        "association_filter": "Li_TFSI_O_contacts_ge_2",
        "persistence_filter_applied": False,
        "persistence_threshold_ps": 0.0,
        "temperature_K": temperature_K,
        "saved_frame_spacing_ps": saved_spacing_ps,
        "analysis_stride_frames": stride,
        "analysis_frame_spacing_ps": stride * saved_spacing_ps,
        "first_frame_time_ps": first_time_ps,
        "last_frame_time_ps": last_time_ps,
        "n_frames": frame_count,
        "max_cluster_size": max_cluster_size,
        "cluster_charge_definition": "formal_z_ij=i-j",
        "trajectory_ionic_charge_scale": 0.7,
        "sigma_static_cNE0_S_cm": sigma,
        "sigma_NE_S_cm": sample.sigma_NE_S_cm,
        "selection_reference_conductivity_S_cm": sample.selection_reference_conductivity_S_cm,
        "static_cNE0_over_reference": sigma / sample.selection_reference_conductivity_S_cm,
        "NE_over_reference": sample.sigma_NE_S_cm / sample.selection_reference_conductivity_S_cm,
        "static_cNE0_over_NE": sigma / sample.sigma_NE_S_cm,
        "abs_log10_error": safe_log10_ratio(
            sigma, sample.selection_reference_conductivity_S_cm
        ),
        "D_Li_cm2s": sample.D_Li_cm2s,
        "D_TFSI_cm2s": sample.D_TFSI_cm2s,
        "V_nm3": sample.V_nm3,
        "N_LI": n_li,
        "N_TFSI": n_anion,
        **descriptors,
    }


def run_samples(
    samples: list[Sample],
    output_dir: Path,
    temperature_K: float,
    target_frame_spacing_ps: float,
    max_cluster_size: int,
    workers: int,
    force: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "per_replica_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    pending: list[Sample] = []
    for sample in samples:
        cache = cache_dir / f"{sample.replica_id}.csv"
        if cache.exists() and not force:
            frames.append(pd.read_csv(cache))
        else:
            pending.append(sample)

    errors: list[dict[str, object]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    compute_one,
                    sample,
                    temperature_K,
                    target_frame_spacing_ps,
                    max_cluster_size,
                ): sample
                for sample in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                sample = futures[future]
                try:
                    row = future.result()
                    frame = pd.DataFrame([row])
                    frame.to_csv(cache_dir / f"{sample.replica_id}.csv", index=False)
                    frames.append(frame)
                    print(
                        f"[{sample.dataset}] {index}/{len(futures)} "
                        f"{sample.replica_id} ok",
                        flush=True,
                    )
                except Exception as error:
                    error_row = {
                        **{
                            key: value
                            for key, value in asdict(sample).items()
                            if key not in {"gro", "xtc"}
                        },
                        "error": repr(error),
                    }
                    errors.append(error_row)
                    print(
                        f"[{sample.dataset}] {index}/{len(futures)} "
                        f"{sample.replica_id} failed: {error}",
                        flush=True,
                    )

    error_path = output_dir / "missing_or_failed.csv"
    pd.DataFrame(errors).to_csv(error_path, index=False)
    if not frames:
        raise RuntimeError(f"No successful calculations for {output_dir}")
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["sample_group", "trajectory_id", "replica"]
    )
    result.to_csv(output_dir / "replica_results.csv", index=False)
    return result


def bootstrap_location_ci(
    values: pd.Series,
    resamples: int = 5000,
    seed: int = 20250821,
) -> tuple[float, float, float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    medians = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.choice(array, size=array.size, replace=True)
        means[index] = sample.mean()
        medians[index] = np.median(sample)
    mean_lower, mean_upper = np.quantile(means, [0.025, 0.975])
    median_lower, median_upper = np.quantile(medians, [0.025, 0.975])
    return (
        float(mean_lower),
        float(mean_upper),
        float(median_lower),
        float(median_upper),
    )


def summarize_groups(
    frame: pd.DataFrame,
    group_order: list[str],
    group_labels: dict[str, str],
    replicas_per_item: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selections = [(group, frame[frame["sample_group"].eq(group)]) for group in group_order]
    selections.append(("ALL", frame))
    for group_index, (group, subset) in enumerate(selections):
        log_static = np.log10(subset["sigma_static_cNE0_mean_S_cm"])
        mean_ci_low, mean_ci_high, median_ci_low, median_ci_high = bootstrap_location_ci(
            log_static, seed=20250821 + group_index
        )
        rows.append(
            {
                "sample_group": group,
                "manuscript_group": group_labels.get(group, "Overall"),
                "candidates_or_trajectories": len(subset),
                "production_replicas": len(subset) * replicas_per_item,
                "mean_abs_delta_log10": subset["abs_log10_error"].mean(),
                "median_abs_delta_log10": subset["abs_log10_error"].median(),
                "median_static_cNE0_over_reference": subset[
                    "static_cNE0_over_reference"
                ].median(),
                "median_reference_over_static_cNE0": subset[
                    "reference_over_static_cNE0"
                ].median(),
                "mean_static_cNE0_over_reference": subset[
                    "static_cNE0_over_reference"
                ].mean(),
                "median_static_cNE0_over_NE": subset[
                    "static_cNE0_over_NE"
                ].median(),
                "median_RDF_first_minimum_nm": subset[
                    "RDF_first_minimum_median_nm"
                ].median(),
                "arithmetic_mean_static_cNE0_S_cm": subset[
                    "sigma_static_cNE0_mean_S_cm"
                ].mean(),
                "geometric_mean_static_cNE0_S_cm": 10.0 ** log_static.mean(),
                "mean_log10_static_cNE0": log_static.mean(),
                "median_log10_static_cNE0": log_static.median(),
                "bootstrap_mean_log10_95CI_low": mean_ci_low,
                "bootstrap_mean_log10_95CI_high": mean_ci_high,
                "bootstrap_median_log10_95CI_low": median_ci_low,
                "bootstrap_median_log10_95CI_high": median_ci_high,
                "bootstrap_resamples": 5000,
                "bootstrap_seed": 20250821 + group_index,
            }
        )
    return pd.DataFrame(rows)


def write_generated_summaries(replica: pd.DataFrame, output_dir: Path) -> None:
    candidate = (
        replica.groupby(["trajectory_id", "sample_group"], as_index=False)
        .agg(
            production_replicas=("replica", "nunique"),
            temperature_K=("temperature_K", "first"),
            sigma_static_cNE0_mean_S_cm=("sigma_static_cNE0_S_cm", "mean"),
            sigma_static_cNE0_std_S_cm=("sigma_static_cNE0_S_cm", "std"),
            sigma_static_cNE0_min_S_cm=("sigma_static_cNE0_S_cm", "min"),
            sigma_static_cNE0_max_S_cm=("sigma_static_cNE0_S_cm", "max"),
            sigma_NE_mean_S_cm=("sigma_NE_S_cm", "mean"),
            sigma_NE_std_S_cm=("sigma_NE_S_cm", "std"),
            selection_reference_conductivity_S_cm=(
                "selection_reference_conductivity_S_cm",
                "first",
            ),
            D_Li_mean_cm2s=("D_Li_cm2s", "mean"),
            D_Li_std_cm2s=("D_Li_cm2s", "std"),
            D_TFSI_mean_cm2s=("D_TFSI_cm2s", "mean"),
            D_TFSI_std_cm2s=("D_TFSI_cm2s", "std"),
            RDF_first_minimum_mean_nm=("cutoff_nm", "mean"),
            RDF_first_minimum_median_nm=("cutoff_nm", "median"),
            RDF_first_minimum_min_nm=("cutoff_nm", "min"),
            RDF_first_minimum_max_nm=("cutoff_nm", "max"),
        )
        .sort_values(["sample_group", "trajectory_id"])
    )
    candidate["static_cNE0_over_reference"] = (
        candidate["sigma_static_cNE0_mean_S_cm"]
        / candidate["selection_reference_conductivity_S_cm"]
    )
    candidate["reference_over_static_cNE0"] = (
        candidate["selection_reference_conductivity_S_cm"]
        / candidate["sigma_static_cNE0_mean_S_cm"]
    )
    candidate["static_cNE0_over_NE"] = (
        candidate["sigma_static_cNE0_mean_S_cm"] / candidate["sigma_NE_mean_S_cm"]
    )
    candidate["abs_log10_error"] = np.abs(
        np.log10(candidate["static_cNE0_over_reference"])
    )
    if len(candidate) != 60 or not candidate["production_replicas"].eq(3).all():
        raise RuntimeError("Generated aggregation must contain 60 candidates with three replicas")
    candidate.to_csv(output_dir / "candidate_means.csv", index=False)

    order = [
        "HIGH_top",
        "HIGH_middle_stratified",
        "HIGH_bottom",
        "LOW_top",
        "LOW_middle_stratified",
        "LOW_bottom",
    ]
    labels = {
        "HIGH_top": "TT-top",
        "HIGH_middle_stratified": "TT-middle",
        "HIGH_bottom": "TT-bottom",
        "LOW_top": "LT-top",
        "LOW_middle_stratified": "LT-middle",
        "LOW_bottom": "LT-bottom",
    }
    summary = summarize_groups(candidate, order, labels, replicas_per_item=3)
    summary.to_csv(output_dir / "group_summary_candidate_level.csv", index=False)

    from scipy.stats import mannwhitneyu

    pairwise_rows: list[dict[str, object]] = []
    for left_index, left in enumerate(order):
        for right in order[left_index + 1 :]:
            left_values = np.log10(
                candidate.loc[
                    candidate["sample_group"].eq(left),
                    "sigma_static_cNE0_mean_S_cm",
                ].to_numpy()
            )
            right_values = np.log10(
                candidate.loc[
                    candidate["sample_group"].eq(right),
                    "sigma_static_cNE0_mean_S_cm",
                ].to_numpy()
            )
            delta_log10 = float(np.median(left_values) - np.median(right_values))
            differences = left_values[:, None] - right_values[None, :]
            cliffs_delta = float(np.sign(differences).mean())
            test = mannwhitneyu(
                left_values, right_values, alternative="two-sided", method="exact"
            )
            pairwise_rows.append(
                {
                    "comparison": f"{labels[left]} vs {labels[right]}",
                    "left_group": left,
                    "right_group": right,
                    "median_delta_log10_static_cNE0": delta_log10,
                    "fold_difference": 10.0**delta_log10,
                    "cliffs_delta": cliffs_delta,
                    "mann_whitney_U": float(test.statistic),
                    "exact_two_sided_p": float(test.pvalue),
                }
            )
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "pairwise_group_tests_candidate_level.csv", index=False
    )


def write_reference_summaries(replica: pd.DataFrame, output_dir: Path) -> None:
    trajectory = replica.copy()
    trajectory["sigma_static_cNE0_mean_S_cm"] = trajectory[
        "sigma_static_cNE0_S_cm"
    ]
    trajectory["RDF_first_minimum_median_nm"] = trajectory["cutoff_nm"]
    trajectory["reference_over_static_cNE0"] = 1.0 / trajectory[
        "static_cNE0_over_reference"
    ]
    trajectory.to_csv(output_dir / "trajectory_results.csv", index=False)
    order = ["bottom", "middle_stratified", "top"]
    labels = {
        "bottom": "Bottom",
        "middle_stratified": "Middle-stratified",
        "top": "Top",
    }
    summary = summarize_groups(trajectory, order, labels, replicas_per_item=1)
    summary.to_csv(output_dir / "group_summary_trajectory_level.csv", index=False)

    from scipy.stats import kendalltau, mannwhitneyu, spearmanr

    log_reference = np.log10(
        trajectory["selection_reference_conductivity_S_cm"].to_numpy()
    )
    log_static = np.log10(trajectory["sigma_static_cNE0_S_cm"].to_numpy())
    bottom = np.log10(
        trajectory.loc[
            trajectory["sample_group"].eq("bottom"), "sigma_static_cNE0_S_cm"
        ].to_numpy()
    )
    top = np.log10(
        trajectory.loc[
            trajectory["sample_group"].eq("top"), "sigma_static_cNE0_S_cm"
        ].to_numpy()
    )
    shift = float(np.median(top) - np.median(bottom))
    test = mannwhitneyu(top, bottom, alternative="two-sided", method="exact")
    statistics = pd.DataFrame(
        [
            {
                "completed_trajectories": len(trajectory),
                "overall_MALogE": trajectory["abs_log10_error"].mean(),
                "overall_median_abs_log10_error": trajectory[
                    "abs_log10_error"
                ].median(),
                "overall_median_static_cNE0_over_reference": trajectory[
                    "static_cNE0_over_reference"
                ].median(),
                "top_bottom_median_shift_log10": shift,
                "top_bottom_fold_difference": 10.0**shift,
                "spearman_rho": float(spearmanr(log_reference, log_static).statistic),
                "kendall_tau": float(kendalltau(log_reference, log_static).statistic),
                "top_bottom_mann_whitney_U": float(test.statistic),
                "top_bottom_exact_two_sided_p": float(test.pvalue),
                "top_bottom_cliffs_delta": float(
                    np.sign(top[:, None] - bottom[None, :]).mean()
                ),
            }
        ]
    )
    statistics.to_csv(output_dir / "overall_statistics.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--reference-base", type=Path, default=DEFAULT_REFERENCE_BASE)
    parser.add_argument(
        "--dataset", choices=["generated", "reference", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--frame-spacing-ps", type=float, default=1.0)
    parser.add_argument("--max-cluster-size", type=int, default=101)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = args.base.resolve()
    reference_base = args.reference_base.resolve()
    jobs: list[tuple[str, list[Sample], Path]] = []
    if args.dataset in {"generated", "all"}:
        samples = load_generated_samples(base)
        jobs.append(
            (
                "generated",
                samples,
                base / "results" / "manuscript_static_cne0_generated_180",
            )
        )
    if args.dataset in {"reference", "all"}:
        samples = load_reference_samples(reference_base)
        jobs.append(
            (
                "reference",
                samples,
                reference_base / "results" / "manuscript_static_cne0_reference_108",
            )
        )

    for label, samples, output_dir in jobs:
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"[{label}] samples={len(samples)} output={output_dir}", flush=True)
        result = run_samples(
            samples,
            output_dir,
            args.temperature_k,
            args.frame_spacing_ps,
            args.max_cluster_size,
            args.workers,
            args.force,
        )
        if label == "generated":
            write_generated_summaries(result, output_dir)
        else:
            write_reference_summaries(result, output_dir)
        print(f"[{label}] completed={len(result)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
