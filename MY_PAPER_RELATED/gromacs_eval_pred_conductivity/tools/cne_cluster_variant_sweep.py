from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


NA = 6.02214076e23
KB = 1.380649e-23
E_CHG = 1.602176634e-19


def uf_init(n: int):
    return np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int8)


def uf_find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return int(x)


def uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = uf_find(parent, int(a))
    rb = uf_find(parent, int(b))
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def add_atom_graph_frame(pop: np.ndarray, n_li: int, core_is_n: np.ndarray, pairs: np.ndarray) -> None:
    n_tot = n_li + len(core_is_n)
    if pairs.size:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_tot, n_tot)).tocsr()
        _, labels = connected_components(graph, directed=False, return_labels=True)
    else:
        labels = np.arange(n_tot, dtype=np.int32)

    li_labels = labels[:n_li]
    an_labels = labels[n_li:][core_is_n]
    li_counts = np.bincount(li_labels, minlength=n_tot)
    an_counts = np.bincount(an_labels, minlength=n_tot)

    roots = np.flatnonzero((li_counts + an_counts) > 0)
    for root in roots:
        i = int(li_counts[root])
        j = int(an_counts[root])
        if i < pop.shape[0] and j < pop.shape[1]:
            pop[i, j] += 1.0


def add_mol_graph_frame(pop: np.ndarray, n_li: int, n_an: int, contacts: np.ndarray) -> None:
    n_tot = n_li + n_an
    if contacts.size:
        a = contacts[:, 0].astype(np.int32)
        b = (n_li + contacts[:, 1]).astype(np.int32)
        rows = np.concatenate([a, b])
        cols = np.concatenate([b, a])
        graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_tot, n_tot)).tocsr()
        _, labels = connected_components(graph, directed=False, return_labels=True)
    else:
        labels = np.arange(n_tot, dtype=np.int32)

    li_counts = np.bincount(labels[:n_li], minlength=n_tot)
    an_counts = np.bincount(labels[n_li:], minlength=n_tot)
    roots = np.flatnonzero((li_counts + an_counts) > 0)
    for root in roots:
        i = int(li_counts[root])
        j = int(an_counts[root])
        if i < pop.shape[0] and j < pop.shape[1]:
            pop[i, j] += 1.0


def sigma_cne_from_pop(
    pop: np.ndarray,
    d_li: float,
    d_an: float,
    volume_nm3: float,
    temperature_k: float,
    max_cluster: int,
) -> float:
    v_cm3 = volume_nm3 * 1e-21
    pref = (E_CHG**2) / (KB * temperature_k * v_cm3)
    imax = min(max_cluster, pop.shape[0])
    jmax = min(max_cluster, pop.shape[1])
    sigma = 0.0
    for i in range(imax):
        for j in range(jmax):
            nij = float(pop[i, j])
            if nij == 0.0 or i == j:
                continue
            q = i - j
            dij_eff = d_li if i > j else d_an
            sigma += pref * (q * q) * nij * dij_eff
    return float(sigma)


def sigma_ne(n_li: int, n_an: int, d_li: float, d_an: float, volume_nm3: float, temperature_k: float) -> float:
    v_cm3 = volume_nm3 * 1e-21
    pref = (E_CHG**2) / (KB * temperature_k * v_cm3)
    return float(pref * (n_li * d_li + n_an * d_an))


def read_htpmd_inputs(run_dir: Path) -> dict:
    path = run_dir / "analysis" / "conductivity_summary_htpmd_ref.csv"
    row = pd.read_csv(path).iloc[0].to_dict()
    return {
        "D_Li_cm2s": float(row["D_Li_cm2s"]),
        "D_an_cm2s": float(row["D_an_cm2s"]),
        "V_nm3": float(row["V_nm3"]),
        "N_LI": int(row["N_LI"]),
        "N_AN": int(row["N_AN"]),
        "T_K": 353.0,
    }


def analyze_traj(run_dir: Path, full_atom_cutoffs_nm: list[float], persist_thresholds: list[float]) -> dict[str, np.ndarray]:
    gro = run_dir / "md" / "production" / "production.gro"
    xtc = run_dir / "md" / "production" / "production.xtc"
    if not gro.exists() or not xtc.exists():
        raise FileNotFoundError(f"missing production gro/xtc under {run_dir}")

    u = mda.Universe(str(gro), str(xtc))
    sel_li = u.select_atoms("resname LI")
    sel_core = u.select_atoms("resname TFSI and (name O* or name S* or name N*)")
    sel_n = u.select_atoms("resname TFSI and name N*")
    if len(sel_li) == 0 or len(sel_core) == 0 or len(sel_n) == 0:
        raise RuntimeError(f"empty selection: LI={len(sel_li)} core={len(sel_core)} N={len(sel_n)}")

    n_li = len(sel_li)
    n_an = len(sel_n)
    pop_shape = (n_li + 1, n_an + 1)
    cutoff_max_a = 0.34 * 10.0
    cutoff2_by_nm = {c: (c * 10.0) ** 2 for c in [0.34, *full_atom_cutoffs_nm]}

    core_is_n = np.isin(sel_core.indices, sel_n.indices)
    core_is_o = np.array([atom.name.upper().startswith("O") for atom in sel_core], dtype=bool)
    an_resindices = list(sel_n.resindices)
    an_map = {residx: i for i, residx in enumerate(an_resindices)}
    core_to_an = np.array([an_map[atom.resindex] for atom in sel_core], dtype=np.int16)

    pop: dict[str, np.ndarray] = {}
    for c in full_atom_cutoffs_nm:
        pop[f"step1_full_atom_cutoff_{c:.2f}_max10"] = np.zeros(pop_shape, dtype=np.float64)
    pop["step2_li_tfsi_core_edges_max10"] = np.zeros(pop_shape, dtype=np.float64)
    pop["step3_li_tfsi_o_edges_max10"] = np.zeros(pop_shape, dtype=np.float64)

    o_contact_frames: list[np.ndarray] = []
    o_contact_counts = np.zeros((n_li, n_an), dtype=np.int32)
    frames = 0

    for ts in u.trajectory:
        coords = np.vstack([sel_li.positions, sel_core.positions]).astype(np.float32)
        box = ts.dimensions.astype(np.float32)
        pairs = FastNS(cutoff_max_a, coords, box, pbc=True).self_search().get_pairs()
        if pairs.size:
            delta = coords[pairs[:, 0]] - coords[pairs[:, 1]]
            lengths = box[:3].astype(np.float32)
            delta -= lengths * np.round(delta / lengths)
            dist2 = np.einsum("ij,ij->i", delta, delta)
        else:
            dist2 = np.empty(0, dtype=np.float32)

        for c in full_atom_cutoffs_nm:
            mask = dist2 <= cutoff2_by_nm[c]
            add_atom_graph_frame(pop[f"step1_full_atom_cutoff_{c:.2f}_max10"], n_li, core_is_n, pairs[mask])

        mask034 = dist2 <= cutoff2_by_nm[0.34]
        p034 = pairs[mask034]
        if p034.size:
            a = p034[:, 0]
            b = p034[:, 1]
            li_core = ((a < n_li) & (b >= n_li)) | ((b < n_li) & (a >= n_li))
            pc = p034[li_core]
            if pc.size:
                li_idx = np.where(pc[:, 0] < n_li, pc[:, 0], pc[:, 1]).astype(np.int16)
                core_idx = np.where(pc[:, 0] >= n_li, pc[:, 0] - n_li, pc[:, 1] - n_li).astype(np.int32)
                an_idx = core_to_an[core_idx]
                contacts_core = np.unique(np.stack([li_idx, an_idx], axis=1), axis=0)
                add_mol_graph_frame(pop["step2_li_tfsi_core_edges_max10"], n_li, n_an, contacts_core)

                o_mask = core_is_o[core_idx]
                contacts_o = np.unique(np.stack([li_idx[o_mask], an_idx[o_mask]], axis=1), axis=0)
            else:
                contacts_core = np.empty((0, 2), dtype=np.int16)
                contacts_o = np.empty((0, 2), dtype=np.int16)
        else:
            contacts_core = np.empty((0, 2), dtype=np.int16)
            contacts_o = np.empty((0, 2), dtype=np.int16)

        add_mol_graph_frame(pop["step3_li_tfsi_o_edges_max10"], n_li, n_an, contacts_o)
        o_contact_frames.append(contacts_o)
        if contacts_o.size:
            o_contact_counts[contacts_o[:, 0], contacts_o[:, 1]] += 1

        frames += 1

    if frames == 0:
        raise RuntimeError(f"no frames read for {run_dir}")

    for key in list(pop):
        pop[key] /= float(frames)

    occupancy = o_contact_counts.astype(np.float64) / float(frames)
    for threshold in persist_thresholds:
        key = f"step4_o_edges_persist_ge_{threshold:.2f}_max10"
        mat = np.zeros(pop_shape, dtype=np.float64)
        keep = occupancy >= threshold
        for contacts_o in o_contact_frames:
            if contacts_o.size:
                mask = keep[contacts_o[:, 0], contacts_o[:, 1]]
                contacts = contacts_o[mask]
            else:
                contacts = contacts_o
            add_mol_graph_frame(mat, n_li, n_an, contacts)
        pop[key] = mat / float(frames)

    return pop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--eval-csv", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    eval_csv = args.eval_csv or root / "results" / "current_cne_eval_completed_ok_only.csv"
    eval_df = pd.read_csv(eval_csv)
    eval_df = eval_df[eval_df["analysis_status"].eq("ok")].copy()
    if args.limit:
        eval_df = eval_df.head(args.limit).copy()

    full_atom_cutoffs_nm = [0.32, 0.30, 0.28]
    persist_thresholds = [0.05, 0.10]
    rows = []

    for pos, row in enumerate(eval_df.itertuples(index=False), start=1):
        tid = int(getattr(row, "_0") if hasattr(row, "_0") else row[0])
        run_dir = root / "runs" / f"Traj_{tid}"
        print(f"[{pos}/{len(eval_df)}] Traj_{tid}", flush=True)
        inputs = read_htpmd_inputs(run_dir)
        pops = analyze_traj(run_dir, full_atom_cutoffs_nm, persist_thresholds)
        baseline_pop_path = run_dir / "analysis" / "pop_mat.npy"
        if baseline_pop_path.exists():
            baseline_pop = np.load(baseline_pop_path)
            pops["step1_full_atom_cutoff_0.34_max10"] = baseline_pop
            pops["step5_full_atom_cutoff_0.34_max50"] = baseline_pop
        ref = float(getattr(row, "reference_CONDUCTIVITY_S_cm"))
        group = str(getattr(row, "sample_group"))
        ne = sigma_ne(inputs["N_LI"], inputs["N_AN"], inputs["D_Li_cm2s"], inputs["D_an_cm2s"], inputs["V_nm3"], inputs["T_K"])
        for variant, pop in pops.items():
            max_cluster = 10
            if variant.endswith("_max50"):
                max_cluster = 50
            elif variant.endswith("_max100"):
                max_cluster = 100
            pred = sigma_cne_from_pop(
                pop,
                inputs["D_Li_cm2s"],
                inputs["D_an_cm2s"],
                inputs["V_nm3"],
                inputs["T_K"],
                max_cluster=max_cluster,
            )
            signed = np.log10(pred) - np.log10(ref) if pred > 0 and ref > 0 else np.nan
            rows.append(
                {
                    "Trajectory ID": tid,
                    "sample_group": group,
                    "variant": variant,
                    "reference_CONDUCTIVITY_S_cm": ref,
                    "sigma_variant_S_cm": pred,
                    "sigma_NE_S_cm": ne,
                    "alpha_variant": pred / ne if ne > 0 else np.nan,
                    "signed_log10_error": signed,
                    "ma_loge": abs(signed) if np.isfinite(signed) else np.nan,
                    "fold_error": 10 ** abs(signed) if np.isfinite(signed) else np.nan,
                    "D_Li_cm2s": inputs["D_Li_cm2s"],
                    "D_an_cm2s": inputs["D_an_cm2s"],
                    "V_nm3": inputs["V_nm3"],
                }
            )

    per = pd.DataFrame(rows)
    summary_rows = []
    for (variant, group), dfg in per.groupby(["variant", "sample_group"], sort=True):
        summary_rows.append(
            {
                "variant": variant,
                "sample_group": group,
                "n": len(dfg),
                "MA_logE": dfg["ma_loge"].mean(),
                "Median_A_logE": dfg["ma_loge"].median(),
                "mean_signed_log10_error": dfg["signed_log10_error"].mean(),
                "median_signed_log10_error": dfg["signed_log10_error"].median(),
                "geomean_fold_error": 10 ** dfg["ma_loge"].mean(),
                "median_fold_error": 10 ** dfg["ma_loge"].median(),
                "median_pred_ref_ratio": (dfg["sigma_variant_S_cm"] / dfg["reference_CONDUCTIVITY_S_cm"]).median(),
                "median_alpha": dfg["alpha_variant"].median(),
                "mean_pred_sigma_S_cm": dfg["sigma_variant_S_cm"].mean(),
            }
        )
    for variant, dfg in per.groupby("variant", sort=True):
        summary_rows.append(
            {
                "variant": variant,
                "sample_group": "all_completed",
                "n": len(dfg),
                "MA_logE": dfg["ma_loge"].mean(),
                "Median_A_logE": dfg["ma_loge"].median(),
                "mean_signed_log10_error": dfg["signed_log10_error"].mean(),
                "median_signed_log10_error": dfg["signed_log10_error"].median(),
                "geomean_fold_error": 10 ** dfg["ma_loge"].mean(),
                "median_fold_error": 10 ** dfg["ma_loge"].median(),
                "median_pred_ref_ratio": (dfg["sigma_variant_S_cm"] / dfg["reference_CONDUCTIVITY_S_cm"]).median(),
                "median_alpha": dfg["alpha_variant"].median(),
                "mean_pred_sigma_S_cm": dfg["sigma_variant_S_cm"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    out_dir = root / "results"
    per_path = out_dir / "cne_cluster_variant_sweep_per_traj.csv"
    summary_path = out_dir / "cne_cluster_variant_sweep_summary.csv"
    per.to_csv(per_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {per_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
