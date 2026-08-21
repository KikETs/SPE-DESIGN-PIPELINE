from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from persistence_sweep_and_hybrid import add_pop_from_codes, pop_descriptors, safe_log10_ratio, sigma_from_pop


@dataclass(frozen=True)
class ReplicaMeta:
    traj_id: int
    replica: int
    stage: str
    group: str
    sigma_ref: float
    sigma_ne: float
    d_li_cm2s: float
    d_an_cm2s: float
    v_nm3: float
    n_li: int
    n_an: int
    gro: Path
    tpr: Path
    xtc: Path
    analysis_csv: Path

    @property
    def replica_id(self) -> str:
        return f"Traj_{self.traj_id}_rep{self.replica}"


def read_xvg_numeric(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.asarray(rows, dtype=float)


def smooth(y: np.ndarray, window: int = 7) -> np.ndarray:
    if y.size < window:
        return y.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    return np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")


def first_peak_minimum(r: np.ndarray, g: np.ndarray) -> dict[str, float]:
    out = {"peak_nm": np.nan, "peak_g": np.nan, "first_min_nm": np.nan, "first_min_g": np.nan}
    if r.size == 0 or g.size == 0:
        return out
    gs = smooth(np.asarray(g, dtype=float), 7)
    peak_mask = (r >= 0.12) & (r <= 0.30)
    if not np.any(peak_mask):
        return out
    peak_candidates = np.where(peak_mask)[0]
    peak_idx = int(peak_candidates[np.nanargmax(gs[peak_candidates])])
    out["peak_nm"] = float(r[peak_idx])
    out["peak_g"] = float(g[peak_idx])

    min_mask = (r > r[peak_idx] + 0.02) & (r <= 0.60)
    idxs = np.where(min_mask)[0]
    chosen = None
    for idx in idxs[1:-1]:
        if gs[idx] <= gs[idx - 1] and gs[idx] <= gs[idx + 1]:
            chosen = int(idx)
            break
    if chosen is None and idxs.size:
        chosen = int(idxs[np.nanargmin(gs[idxs])])
    if chosen is not None:
        out["first_min_nm"] = float(r[chosen])
        out["first_min_g"] = float(g[chosen])
    return out


def interp_cn(cn_arr: np.ndarray, r_nm: float, col: int) -> float:
    if cn_arr.size == 0 or not np.isfinite(r_nm) or cn_arr.shape[1] <= col:
        return np.nan
    return float(np.interp(r_nm, cn_arr[:, 0], cn_arr[:, col]))


def read_replica_summary(path: Path) -> dict[str, float]:
    row = pd.read_csv(path).iloc[0]

    def get(name: str) -> float:
        return float(pd.to_numeric(row.get(name, np.nan), errors="coerce"))

    return {
        "sigma_ne": get("sigma_NE_htpmd_S_cm"),
        "d_li_cm2s": get("D_Li_cm2s"),
        "d_an_cm2s": get("D_an_cm2s"),
        "v_nm3": get("V_nm3"),
        "n_li": int(get("N_LI")),
        "n_an": int(get("N_AN")),
    }


def discover_recent_replicas(base: Path) -> list[ReplicaMeta]:
    rr_path = base / "results" / "run_results.csv"
    if not rr_path.exists():
        raise FileNotFoundError(f"Missing {rr_path}")
    rr = pd.read_csv(rr_path)
    required = {"Trajectory ID", "sample_group", "CONDUCTIVITY", "analysis_replicas_csv"}
    missing = required - set(rr.columns)
    if missing:
        raise KeyError(f"run_results.csv missing columns: {sorted(missing)}")

    metas: list[ReplicaMeta] = []
    for _, row in rr.iterrows():
        traj_id = int(row["Trajectory ID"])
        sigma_ref = float(pd.to_numeric(row["CONDUCTIVITY"], errors="coerce"))
        group = str(row["sample_group"])
        csvs = [Path(x) for x in str(row["analysis_replicas_csv"]).split("|") if x.strip()]
        if len(csvs) != 3:
            warnings.warn(f"Traj_{traj_id}: expected 3 replica CSVs, found {len(csvs)}")
        for replica in (1, 2, 3):
            stage = "production" if replica == 1 else f"production_rep{replica}"
            stage_dir = base / "runs" / f"Traj_{traj_id}" / "md" / stage
            analysis_csv = base / "runs" / f"Traj_{traj_id}" / "analysis" / f"replica_{replica}" / "conductivity_summary_htpmd_ref.csv"
            gro = stage_dir / f"{stage}.gro"
            tpr = stage_dir / f"{stage}.tpr"
            xtc = stage_dir / f"{stage}.xtc"
            missing_paths = [p for p in (analysis_csv, gro, tpr, xtc) if not p.exists()]
            if missing_paths:
                warnings.warn(f"Skipping {traj_id} replica {replica}: missing {missing_paths}")
                continue
            h = read_replica_summary(analysis_csv)
            metas.append(
                ReplicaMeta(
                    traj_id=traj_id,
                    replica=replica,
                    stage=stage,
                    group=group,
                    sigma_ref=sigma_ref,
                    sigma_ne=float(h["sigma_ne"]),
                    d_li_cm2s=float(h["d_li_cm2s"]),
                    d_an_cm2s=float(h["d_an_cm2s"]),
                    v_nm3=float(h["v_nm3"]),
                    n_li=int(h["n_li"]),
                    n_an=int(h["n_an"]),
                    gro=gro,
                    tpr=tpr,
                    xtc=xtc,
                    analysis_csv=analysis_csv,
                )
            )
    if not metas:
        raise RuntimeError("No current replica inputs found.")
    return metas


def run_rdf_one(meta: ReplicaMeta, out_dir: Path, bin_nm: float, rmax_nm: float, dt_ps: float, force: bool) -> dict[str, object]:
    traj_dir = out_dir / "per_replica_rdf"
    traj_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"Traj_{meta.traj_id}_rep{meta.replica}"
    rdf_path = traj_dir / f"{prefix}_rdf_li_tfsiO_li_tfsiN.xvg"
    cn_path = traj_dir / f"{prefix}_rdf_cn_li_tfsiO_li_tfsiN.xvg"
    log_path = traj_dir / f"{prefix}_gmx_rdf.log"
    if force:
        for path in (rdf_path, cn_path):
            if path.exists():
                path.unlink()

    if not rdf_path.exists() or not cn_path.exists():
        cmd = [
            "gmx",
            "rdf",
            "-f",
            str(meta.xtc),
            "-s",
            str(meta.tpr),
            "-o",
            str(rdf_path),
            "-cn",
            str(cn_path),
            "-ref",
            "resname LI",
            "-sel",
            "resname TFSI and name O03 O04 O10 O11; resname TFSI and name N01",
            "-bin",
            f"{bin_nm:g}",
            "-rmax",
            f"{rmax_nm:g}",
            "-xvg",
            "none",
        ]
        if dt_ps > 0:
            cmd.extend(["-dt", f"{dt_ps:g}"])
        env = os.environ.copy()
        env["GMX_MAXBACKUP"] = "-1"
        env["OMP_NUM_THREADS"] = "1"
        env["GMX_NUM_THREADS"] = "1"
        proc = subprocess.run(cmd, cwd=str(BASE), env=env, text=True, capture_output=True)
        log_path.write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            return {
                "traj_id": meta.traj_id,
                "Trajectory ID": meta.traj_id,
                "replica": meta.replica,
                "replica_id": meta.replica_id,
                "stage": meta.stage,
                "group": meta.group,
                "rdf_status": "failed",
                "error_tail": (proc.stderr or proc.stdout)[-1200:],
            }

    arr = read_xvg_numeric(rdf_path)
    cn = read_xvg_numeric(cn_path)
    if arr.size == 0 or arr.shape[1] < 3:
        return {
            "traj_id": meta.traj_id,
            "Trajectory ID": meta.traj_id,
            "replica": meta.replica,
            "replica_id": meta.replica_id,
            "stage": meta.stage,
            "group": meta.group,
            "rdf_status": "failed",
            "error_tail": "RDF output had no numeric columns",
        }
    r = arr[:, 0]
    o_stats = first_peak_minimum(r, arr[:, 1])
    n_stats = first_peak_minimum(r, arr[:, 2])
    return {
        "traj_id": meta.traj_id,
        "Trajectory ID": meta.traj_id,
        "replica": meta.replica,
        "replica_id": meta.replica_id,
        "stage": meta.stage,
        "group": meta.group,
        "rdf_status": "ok",
        "rdf_xvg": str(rdf_path),
        "cn_xvg": str(cn_path),
        "Li_TFSI_O_peak_nm": o_stats["peak_nm"],
        "Li_TFSI_O_peak_g": o_stats["peak_g"],
        "Li_TFSI_O_first_min_nm": o_stats["first_min_nm"],
        "Li_TFSI_O_first_min_g": o_stats["first_min_g"],
        "Li_TFSI_O_coord_at_first_min": interp_cn(cn, o_stats["first_min_nm"], 1),
        "Li_TFSI_N_peak_nm": n_stats["peak_nm"],
        "Li_TFSI_N_peak_g": n_stats["peak_g"],
        "Li_TFSI_N_first_min_nm": n_stats["first_min_nm"],
        "Li_TFSI_N_first_min_g": n_stats["first_min_g"],
        "Li_TFSI_N_coord_at_first_min": interp_cn(cn, n_stats["first_min_nm"], 2),
        "error_tail": "",
    }


def compute_tau0_cne_one(meta: ReplicaMeta, cutoff_nm: float, temperature_k: float, max_cluster: int) -> dict[str, object]:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_bonds
    from MDAnalysis.lib.nsgrid import FastNS

    if not np.isfinite(cutoff_nm) or cutoff_nm <= 0:
        raise RuntimeError(f"{meta.replica_id}: invalid RDF peak cutoff {cutoff_nm}")

    u = mda.Universe(str(meta.gro), str(meta.xtc))
    sel_li = u.select_atoms("resname LI")
    sel_o = u.select_atoms("resname TFSI and name O*")
    sel_n = u.select_atoms("resname TFSI and name N*")
    if len(sel_li) == 0 or len(sel_o) == 0 or len(sel_n) == 0:
        raise RuntimeError(f"{meta.replica_id}: empty selection LI={len(sel_li)} O={len(sel_o)} N={len(sel_n)}")

    n_li = len(sel_li)
    n_an = len(sel_n)
    if n_li != meta.n_li or n_an != meta.n_an:
        warnings.warn(f"{meta.replica_id}: selection counts {n_li}/{n_an} differ from summary {meta.n_li}/{meta.n_an}")

    anion_resindex_to_i = {int(residx): i for i, residx in enumerate(list(sel_n.resindices))}
    o_to_an = np.array([anion_resindex_to_i[int(atom.resindex)] for atom in sel_o], dtype=np.int32)
    cutoff_a = float(cutoff_nm) * 10.0
    pop_shape = (max(2, min(max_cluster, n_li + 1)), max(2, min(max_cluster, n_an + 1)))
    pop = np.zeros(pop_shape, dtype=np.float64)
    largest = 0
    n_frames = 0

    for ts in u.trajectory:
        coords = np.vstack([sel_li.positions, sel_o.positions]).astype(np.float32)
        pairs = FastNS(cutoff_a, coords, np.asarray(ts.dimensions, dtype=np.float32), pbc=True).self_search().get_pairs()
        if pairs.size:
            a = pairs[:, 0]
            b = pairs[:, 1]
            mask = ((a < n_li) & (b >= n_li)) | ((b < n_li) & (a >= n_li))
            pair_contacts = pairs[mask]
            if pair_contacts.size:
                li_idx = np.where(pair_contacts[:, 0] < n_li, pair_contacts[:, 0], pair_contacts[:, 1]).astype(np.int32)
                o_idx = np.where(
                    pair_contacts[:, 0] >= n_li,
                    pair_contacts[:, 0] - n_li,
                    pair_contacts[:, 1] - n_li,
                ).astype(np.int32)
                an_idx = o_to_an[o_idx]
                d_a = calc_bonds(sel_li.positions[li_idx], sel_o.positions[o_idx], box=ts.dimensions)
                keep = d_a <= cutoff_a
                codes = np.unique((li_idx[keep] * n_an + an_idx[keep]).astype(np.int32)) if np.any(keep) else np.empty(0, dtype=np.int32)
            else:
                codes = np.empty(0, dtype=np.int32)
        else:
            codes = np.empty(0, dtype=np.int32)
        largest = max(largest, add_pop_from_codes(pop, codes, n_li, n_an))
        n_frames += 1

    if n_frames == 0:
        raise RuntimeError(f"{meta.replica_id}: no trajectory frames")
    pop /= float(n_frames)
    # cNE uses formal cluster charge z_ij = i - j. alpha_ij is a frame-averaged
    # cluster count/population, not a normalized fraction.
    sigma = sigma_from_pop(pop, meta.d_li_cm2s, meta.d_an_cm2s, meta.v_nm3, temperature_k, max_cluster)
    desc = pop_descriptors(pop, n_li, n_an)
    desc["largest_cluster_size"] = max(desc["largest_cluster_size"], largest)
    return {
        "traj_id": meta.traj_id,
        "Trajectory ID": meta.traj_id,
        "replica": meta.replica,
        "replica_id": meta.replica_id,
        "stage": meta.stage,
        "group": meta.group,
        "sample_group": meta.group,
        "cutoff_variant": "rdf_peak",
        "cutoff_nm": cutoff_nm,
        "persistence_threshold_ps": 0.0,
        "tau_ps": 0.0,
        "temperature_K": temperature_k,
        "n_frames": n_frames,
        "sigma_cNE_tau0_rdf_peak_S_cm": sigma,
        "sigma_cNE_tau": sigma,
        "sigma_NE": meta.sigma_ne,
        "sigma_ref": meta.sigma_ref,
        "pred_ref": sigma / meta.sigma_ref if meta.sigma_ref > 0 else np.nan,
        "NE_ref": meta.sigma_ne / meta.sigma_ref if meta.sigma_ref > 0 else np.nan,
        "s_tau": sigma / meta.sigma_ne if meta.sigma_ne > 0 else np.nan,
        "abs_log_error": safe_log10_ratio(sigma, meta.sigma_ref),
        "D_Li_cm2s": meta.d_li_cm2s,
        "D_an_cm2s": meta.d_an_cm2s,
        "V_nm3": meta.v_nm3,
        "N_LI": n_li,
        "N_AN": n_an,
        **desc,
    }


def summarize(df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    valid = df[np.isfinite(pd.to_numeric(df["abs_log_error"], errors="coerce"))].copy()
    rows: list[dict[str, object]] = []

    def med(sub: pd.DataFrame, col: str) -> float:
        if col not in sub.columns:
            return np.nan
        vals = pd.to_numeric(sub[col], errors="coerce")
        return float(vals.median()) if vals.notna().any() else np.nan

    for group, sub in valid.groupby(group_col, sort=True):
        rows.append(
            {
                group_col: group,
                "n": int(len(sub)),
                "MAlogE": float(sub["abs_log_error"].mean()),
                "median_AlogE": float(sub["abs_log_error"].median()),
                "median_pred_ref": float(sub["pred_ref"].median()),
                "mean_pred_ref": float(sub["pred_ref"].mean()),
                "median_sigma_cNE_S_cm": float(sub["sigma_cNE_tau"].median()),
                "median_sigma_ref_S_cm": float(sub["sigma_ref"].median()),
                "median_sigma_NE_S_cm": float(sub["sigma_NE"].median()),
                "median_NE_ref": float(sub["NE_ref"].median()),
                "median_cNE_over_NE": float(sub["s_tau"].median()),
                "median_cutoff_nm": float(sub["cutoff_nm"].median()),
                "median_free_cation_fraction": med(sub, "free_cation_fraction"),
                "median_free_anion_fraction": med(sub, "free_anion_fraction"),
                "median_neutral_pop": med(sub, "neutral_pop"),
                "median_charged_pop": med(sub, "charged_pop"),
            }
        )
    rows.append(
        {
            group_col: "ALL",
            "n": int(len(valid)),
            "MAlogE": float(valid["abs_log_error"].mean()),
            "median_AlogE": float(valid["abs_log_error"].median()),
            "median_pred_ref": float(valid["pred_ref"].median()),
            "mean_pred_ref": float(valid["pred_ref"].mean()),
            "median_sigma_cNE_S_cm": float(valid["sigma_cNE_tau"].median()),
            "median_sigma_ref_S_cm": float(valid["sigma_ref"].median()),
            "median_sigma_NE_S_cm": float(valid["sigma_NE"].median()),
            "median_NE_ref": float(valid["NE_ref"].median()),
            "median_cNE_over_NE": float(valid["s_tau"].median()),
            "median_cutoff_nm": float(valid["cutoff_nm"].median()),
            "median_free_cation_fraction": med(valid, "free_cation_fraction"),
            "median_free_anion_fraction": med(valid, "free_anion_fraction"),
            "median_neutral_pop": med(valid, "neutral_pop"),
            "median_charged_pop": med(valid, "charged_pop"),
        }
    )
    return pd.DataFrame(rows)


def summarize_system_mean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = (
        df.groupby(["traj_id", "group"], as_index=False)
        .agg(
            n_replicas=("replica", "count"),
            sigma_cNE_tau=("sigma_cNE_tau", "mean"),
            sigma_NE=("sigma_NE", "mean"),
            sigma_ref=("sigma_ref", "first"),
            cutoff_nm=("cutoff_nm", "median"),
            free_cation_fraction=("free_cation_fraction", "mean"),
            free_anion_fraction=("free_anion_fraction", "mean"),
            neutral_pop=("neutral_pop", "mean"),
            charged_pop=("charged_pop", "mean"),
        )
    )
    agg["pred_ref"] = agg["sigma_cNE_tau"] / agg["sigma_ref"]
    agg["NE_ref"] = agg["sigma_NE"] / agg["sigma_ref"]
    agg["s_tau"] = agg["sigma_cNE_tau"] / agg["sigma_NE"]
    agg["abs_log_error"] = [
        safe_log10_ratio(float(pred), float(ref)) for pred, ref in zip(agg["sigma_cNE_tau"], agg["sigma_ref"])
    ]
    return agg, summarize(agg, "group")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=BASE / "results" / "rdf_peak_tau0_replica60")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bin-nm", type=float, default=0.002)
    parser.add_argument("--rmax-nm", type=float, default=1.0)
    parser.add_argument("--rdf-dt-ps", type=float, default=0.0, help="0 means use every saved frame for RDF.")
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--max-cluster", type=int, default=101)
    parser.add_argument("--force-rdf", action="store_true")
    parser.add_argument("--force-cne", action="store_true")
    args = parser.parse_args()

    base = args.base.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = discover_recent_replicas(base)
    print(f"[discover] recent replicas: {len(metas)} from {len(set(m.traj_id for m in metas))} systems", flush=True)

    meta_by_key = {(m.traj_id, m.replica): m for m in metas}

    rdf_summary_path = out_dir / "replica_rdf_peak_summary.csv"
    if rdf_summary_path.exists() and not args.force_rdf:
        rdf = pd.read_csv(rdf_summary_path)
        print(f"[rdf] using cached {rdf_summary_path}", flush=True)
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {
                ex.submit(run_rdf_one, meta, out_dir, args.bin_nm, args.rmax_nm, args.rdf_dt_ps, args.force_rdf): meta
                for meta in metas
            }
            for i, fut in enumerate(as_completed(futs), start=1):
                meta = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "traj_id": meta.traj_id,
                        "Trajectory ID": meta.traj_id,
                        "replica": meta.replica,
                        "replica_id": meta.replica_id,
                        "stage": meta.stage,
                        "group": meta.group,
                        "rdf_status": "failed",
                        "error_tail": repr(exc),
                    }
                rows.append(row)
                print(f"[rdf] {i}/{len(futs)} {meta.replica_id} {row.get('rdf_status')}", flush=True)
        rdf = pd.DataFrame(rows).sort_values(["group", "traj_id", "replica"])
        rdf.to_csv(rdf_summary_path, index=False)

    cne_path = out_dir / "rdf_peak_tau0_per_replica.csv"
    if cne_path.exists() and not args.force_cne:
        cne = pd.read_csv(cne_path)
        print(f"[cNE] using cached {cne_path}", flush=True)
    else:
        ok_rdf = rdf[rdf["rdf_status"].eq("ok")].copy()
        rows = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {}
            for _, row in ok_rdf.iterrows():
                key = (int(row["traj_id"]), int(row["replica"]))
                meta = meta_by_key[key]
                cutoff_nm = float(pd.to_numeric(row["Li_TFSI_O_peak_nm"], errors="coerce"))
                futs[ex.submit(compute_tau0_cne_one, meta, cutoff_nm, args.temperature_k, args.max_cluster)] = (meta, cutoff_nm)
            for i, fut in enumerate(as_completed(futs), start=1):
                meta, cutoff_nm = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "traj_id": meta.traj_id,
                        "Trajectory ID": meta.traj_id,
                        "replica": meta.replica,
                        "replica_id": meta.replica_id,
                        "stage": meta.stage,
                        "group": meta.group,
                        "cutoff_variant": "rdf_peak",
                        "cutoff_nm": cutoff_nm,
                        "sigma_ref": meta.sigma_ref,
                        "sigma_NE": meta.sigma_ne,
                        "sigma_cNE_tau": np.nan,
                        "pred_ref": np.nan,
                        "NE_ref": meta.sigma_ne / meta.sigma_ref if meta.sigma_ref > 0 else np.nan,
                        "s_tau": np.nan,
                        "abs_log_error": np.nan,
                        "error_tail": repr(exc),
                    }
                rows.append(row)
                print(f"[cNE] {i}/{len(futs)} {meta.replica_id} done", flush=True)
        cne = pd.DataFrame(rows).sort_values(["group", "traj_id", "replica"])
        cne.to_csv(cne_path, index=False)

    replica_summary = summarize(cne, "group")
    replica_summary.to_csv(out_dir / "rdf_peak_tau0_group_summary_replica_unit.csv", index=False)
    system_mean, system_summary = summarize_system_mean(cne[np.isfinite(pd.to_numeric(cne["abs_log_error"], errors="coerce"))].copy())
    system_mean.to_csv(out_dir / "rdf_peak_tau0_per_system_mean.csv", index=False)
    system_summary.to_csv(out_dir / "rdf_peak_tau0_group_summary_system_mean.csv", index=False)

    print("[summary] replica-unit", flush=True)
    print(replica_summary[["group", "n", "MAlogE", "median_pred_ref", "median_cutoff_nm"]].to_string(index=False), flush=True)
    print("[summary] system-mean", flush=True)
    print(system_summary[["group", "n", "MAlogE", "median_pred_ref", "median_cutoff_nm"]].to_string(index=False), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
