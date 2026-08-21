from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]


def read_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        raise ValueError(f"No numeric MSD rows in {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def linear_fit(time_ps: np.ndarray, msd_nm2: np.ndarray, begin_ns: float, end_ns: float) -> dict[str, float]:
    time_ns = time_ps / 1000.0
    keep = np.isfinite(time_ns) & np.isfinite(msd_nm2) & (time_ns >= begin_ns) & (time_ns <= end_ns)
    x = time_ns[keep]
    y = msd_nm2[keep]
    if x.size < 3:
        raise ValueError(f"Only {x.size} points in fit window {begin_ns:g}-{end_ns:g} ns")
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    diffusivity_cm2s = float(slope / 6.0 * 1.0e-5)
    return {
        "fit_begin_ns": float(begin_ns),
        "fit_end_ns": float(end_ns),
        "fit_points": int(x.size),
        "slope_nm2_per_ns": float(slope),
        "intercept_nm2": float(intercept),
        "fit_r2": float(r2),
        "diffusivity_cm2s_recomputed": diffusivity_cm2s,
    }


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def build(base: Path, run_results: Path, out_dir: Path) -> None:
    runs = pd.read_csv(run_results)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    species_files = {"Li": "msd_li.xvg", "TFSI_N": "msd_tfsi_N.xvg"}
    expected_diffusivity = {"Li": "D_Li_cm2s", "TFSI_N": "D_an_cm2s"}

    for _, run in runs.iterrows():
        traj_id = int(run["Trajectory ID"])
        group = str(run["sample_group"])
        for replica in (1, 2, 3):
            analysis_dir = base / "runs" / f"Traj_{traj_id}" / "analysis" / f"replica_{replica}"
            summary_path = analysis_dir / "conductivity_summary_htpmd_ref.csv"
            if not summary_path.exists():
                failures.append({"traj_id": traj_id, "replica": replica, "species": "ALL", "reason": "missing analysis summary", "path": relative_path(summary_path, base)})
                continue
            summary = pd.read_csv(summary_path).iloc[0]
            begin_ns = float(summary["analysis_begin_ns"])
            end_ns = float(summary["analysis_end_ns"])
            for species, filename in species_files.items():
                msd_path = analysis_dir / filename
                try:
                    time_ps, msd_nm2 = read_xvg(msd_path)
                    fit = linear_fit(time_ps, msd_nm2, begin_ns, end_ns)
                    expected = float(summary[expected_diffusivity[species]])
                    delta = fit["diffusivity_cm2s_recomputed"] - expected
                    rows.append(
                        {
                            "traj_id": traj_id,
                            "replica": replica,
                            "replica_id": f"Traj_{traj_id}_rep{replica}",
                            "group": group,
                            "species": species,
                            **fit,
                            "diffusivity_cm2s_summary": expected,
                            "diffusivity_abs_delta_cm2s": abs(delta),
                            "analysis_summary_csv": relative_path(summary_path, base),
                            "msd_xvg": relative_path(msd_path, base),
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "traj_id": traj_id,
                            "replica": replica,
                            "species": species,
                            "reason": repr(exc),
                            "path": relative_path(msd_path, base),
                        }
                    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_replica = pd.DataFrame(rows).sort_values(["traj_id", "replica", "species"])
    failed = pd.DataFrame(failures, columns=["traj_id", "replica", "species", "reason", "path"])
    per_replica.to_csv(out_dir / "msd_fit_r2_per_replica.csv", index=False)
    failed.to_csv(out_dir / "msd_fit_r2_missing_or_failed.csv", index=False)

    candidate = (
        per_replica.groupby(["traj_id", "group", "species"], as_index=False)
        .agg(n_replicas=("replica", "nunique"), mean_fit_r2=("fit_r2", "mean"), min_fit_r2=("fit_r2", "min"), max_fit_r2=("fit_r2", "max"))
    )
    candidate.to_csv(out_dir / "msd_fit_r2_candidate_mean.csv", index=False)

    summary_rows = []
    for species, part in per_replica.groupby("species", sort=True):
        means = candidate[candidate["species"].eq(species)]
        summary_rows.append(
            {
                "species": species,
                "replica_fits": int(len(part)),
                "candidates": int(means["traj_id"].nunique()),
                "candidate_mean_r2_min": float(means["mean_fit_r2"].min()),
                "candidate_mean_r2_max": float(means["mean_fit_r2"].max()),
                "minimum_replica_fit_r2": float(part["fit_r2"].min()),
                "fit_begin_ns": float(part["fit_begin_ns"].min()),
                "fit_end_ns": float(part["fit_end_ns"].max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "msd_fit_r2_summary.csv", index=False)

    expected = len(runs) * 3 * len(species_files)
    if len(per_replica) != expected or not failed.empty:
        raise RuntimeError(f"MSD fit audit failed: expected {expected}, got {len(per_replica)}, failures={len(failed)}")
    print(f"[done] {len(per_replica)} fits from {len(runs)} candidates; failures=0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute replica-level MSD linear-fit R2 diagnostics.")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--run-results", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    run_results = (args.run_results or base / "github_results" / "latest_notebook_manifest_60" / "run_results.csv").resolve()
    out_dir = (args.out_dir or base / "github_results" / "latest_notebook_manifest_60" / "s14b_msd_fit_r2").resolve()
    build(base, run_results, out_dir)


if __name__ == "__main__":
    main()
