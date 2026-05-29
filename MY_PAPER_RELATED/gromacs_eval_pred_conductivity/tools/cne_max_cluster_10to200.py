from __future__ import annotations

import argparse
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


KB = 1.380649e-23
E_CHG = 1.602176634e-19
TEMP_K = 353.0


def add_atom_graph_frame(pop: np.ndarray, n_li: int, core_is_n: np.ndarray, pairs: np.ndarray) -> None:
    n_tot = n_li + len(core_is_n)
    if pairs.size:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_tot, n_tot)).tocsr()
        _, labels = connected_components(graph, directed=False, return_labels=True)
    else:
        labels = np.arange(n_tot, dtype=np.int32)

    li_counts = np.bincount(labels[:n_li], minlength=n_tot)
    an_counts = np.bincount(labels[n_li:][core_is_n], minlength=n_tot)
    roots = np.flatnonzero((li_counts + an_counts) > 0)
    for root in roots:
        i = int(li_counts[root])
        j = int(an_counts[root])
        if i < pop.shape[0] and j < pop.shape[1]:
            pop[i, j] += 1.0


def full_atom_pop(run_dir: Path, cache_path: Path, force: bool = False) -> np.ndarray:
    if cache_path.exists() and not force:
        return np.load(cache_path)

    gro = run_dir / "md" / "production" / "production.gro"
    xtc = run_dir / "md" / "production" / "production.xtc"
    u = mda.Universe(str(gro), str(xtc))
    sel_li = u.select_atoms("resname LI")
    sel_core = u.select_atoms("resname TFSI and (name O* or name S* or name N*)")
    sel_n = u.select_atoms("resname TFSI and name N*")
    if len(sel_li) == 0 or len(sel_core) == 0 or len(sel_n) == 0:
        raise RuntimeError(f"empty selection: LI={len(sel_li)} core={len(sel_core)} N={len(sel_n)}")

    n_li = len(sel_li)
    n_an = len(sel_n)
    core_is_n = np.isin(sel_core.indices, sel_n.indices)
    pop = np.zeros((n_li + 1, n_an + 1), dtype=np.float64)
    frames = 0
    for ts in u.trajectory:
        coords = np.vstack([sel_li.positions, sel_core.positions]).astype(np.float32)
        pairs = FastNS(3.4, coords, ts.dimensions.astype(np.float32), pbc=True).self_search().get_pairs()
        add_atom_graph_frame(pop, n_li, core_is_n, pairs)
        frames += 1
    if frames == 0:
        raise RuntimeError(f"no frames read: {run_dir}")
    pop /= float(frames)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pop)
    return pop


def sigma_cne(pop: np.ndarray, max_cluster: int, d_li: float, d_an: float, volume_nm3: float) -> float:
    pref = (E_CHG**2) / (KB * TEMP_K * (volume_nm3 * 1e-21))
    imax = min(int(max_cluster), int(pop.shape[0]))
    jmax = min(int(max_cluster), int(pop.shape[1]))
    sigma = 0.0
    for i in range(imax):
        for j in range(jmax):
            if i == j:
                continue
            nij = float(pop[i, j])
            if nij == 0.0:
                continue
            q = i - j
            sigma += pref * (q * q) * nij * (d_li if i > j else d_an)
    return float(sigma)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    eval_df = pd.read_csv(root / "results" / "current_cne_eval_completed_ok_only.csv")
    eval_df = eval_df[eval_df["analysis_status"].eq("ok")].copy()
    cache_root = root / "results" / "cne_full_atom_0p34_pop_cache"

    rows = []
    max_values = list(range(10, 201, 10))
    for pos, (_, row) in enumerate(eval_df.iterrows(), start=1):
        tid = int(row["Trajectory ID"])
        print(f"[{pos}/{len(eval_df)}] Traj_{tid}", flush=True)
        run_dir = root / "runs" / f"Traj_{tid}"
        pop = full_atom_pop(run_dir, cache_root / f"Traj_{tid}_pop_0p34_full.npy", force=args.force)
        h = pd.read_csv(run_dir / "analysis" / "conductivity_summary_htpmd_ref.csv").iloc[0]
        d_li = float(h["D_Li_cm2s"])
        d_an = float(h["D_an_cm2s"])
        volume_nm3 = float(h["V_nm3"])
        ref = float(row["reference_CONDUCTIVITY_S_cm"])
        for max_cluster in max_values:
            pred = sigma_cne(pop, max_cluster, d_li, d_an, volume_nm3)
            signed = np.log10(pred) - np.log10(ref) if pred > 0 and ref > 0 else np.nan
            rows.append(
                {
                    "Trajectory ID": tid,
                    "sample_group": row["sample_group"],
                    "max_cluster": max_cluster,
                    "reference_CONDUCTIVITY_S_cm": ref,
                    "sigma_cNE_S_cm": pred,
                    "signed_log10_error": signed,
                    "ma_loge": abs(signed) if np.isfinite(signed) else np.nan,
                    "fold_error": 10 ** abs(signed) if np.isfinite(signed) else np.nan,
                    "pred_ref_ratio": pred / ref if ref > 0 else np.nan,
                    "pop_shape_i": pop.shape[0],
                    "pop_shape_j": pop.shape[1],
                }
            )

    per = pd.DataFrame(rows)
    summary_rows = []
    for (max_cluster, group), dfg in per.groupby(["max_cluster", "sample_group"], sort=True):
        summary_rows.append(
            {
                "max_cluster": max_cluster,
                "sample_group": group,
                "n": len(dfg),
                "MA_logE": dfg["ma_loge"].mean(),
                "Median_A_logE": dfg["ma_loge"].median(),
                "mean_signed_log10_error": dfg["signed_log10_error"].mean(),
                "median_signed_log10_error": dfg["signed_log10_error"].median(),
                "geomean_fold_error": 10 ** dfg["ma_loge"].mean(),
                "median_pred_ref_ratio": dfg["pred_ref_ratio"].median(),
                "mean_pred_sigma_S_cm": dfg["sigma_cNE_S_cm"].mean(),
            }
        )
    for max_cluster, dfg in per.groupby("max_cluster", sort=True):
        summary_rows.append(
            {
                "max_cluster": max_cluster,
                "sample_group": "all_completed",
                "n": len(dfg),
                "MA_logE": dfg["ma_loge"].mean(),
                "Median_A_logE": dfg["ma_loge"].median(),
                "mean_signed_log10_error": dfg["signed_log10_error"].mean(),
                "median_signed_log10_error": dfg["signed_log10_error"].median(),
                "geomean_fold_error": 10 ** dfg["ma_loge"].mean(),
                "median_pred_ref_ratio": dfg["pred_ref_ratio"].median(),
                "mean_pred_sigma_S_cm": dfg["sigma_cNE_S_cm"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    per_path = root / "results" / "cne_max_cluster_10to200_per_traj.csv"
    summary_path = root / "results" / "cne_max_cluster_10to200_summary.csv"
    per.to_csv(per_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {per_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
