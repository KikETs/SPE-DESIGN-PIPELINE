#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.nsgrid import FastNS


E_CHG = 1.602176634e-19
KB = 1.380649e-23


def uf_init(n: int):
    return np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int8)


def uf_find(parent, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return int(x)


def uf_union(parent, rank, a: int, b: int) -> None:
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def gro_volume_nm3(gro: Path) -> float:
    vals = [float(x) for x in gro.read_text().splitlines()[-1].split()]
    return vals[0] * vals[1] * vals[2]


def ref_table(root: Path) -> pd.DataFrame:
    p = root / "results" / "current_cne_eval_completed_ok_only.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def infer_resnames(u: mda.Universe) -> tuple[str, str]:
    resnames = {r.resname for r in u.residues}
    if "LI" not in resnames:
        raise RuntimeError(f"LI resname not found. resnames={sorted(resnames)}")
    for cand in ("TFSI", "NSC", "TFS"):
        if cand in resnames:
            return "LI", cand
    raise RuntimeError(f"anion resname not found. resnames={sorted(resnames)}")


def unwrap_update(pos_unwrap, pos_prev, pos_cur, box_lengths):
    delta = pos_cur - pos_prev
    delta -= box_lengths * np.round(delta / box_lengths)
    pos_unwrap += delta
    return pos_cur.copy()


def mass_com(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    m = masses.astype(np.float64)
    if not np.isfinite(m).all() or np.allclose(m.sum(), 0.0):
        return pos.mean(axis=0)
    return (pos * m[:, None]).sum(axis=0) / m.sum()


def collect_cluster_trace(
    tpr: Path,
    xtc: Path,
    gro: Path,
    cutoff_nm: float,
    begin_ns: float,
    end_ns: float,
    stride_frames: int,
):
    u = mda.Universe(str(gro), str(xtc))
    cation_res, anion_res = infer_resnames(u)
    sel_li = u.select_atoms(f"resname {cation_res}")
    sel_anion_n = u.select_atoms(f"resname {anion_res} and name N*")
    sel_core = u.select_atoms(f"resname {anion_res} and (name O* or name S* or name N*)")
    if len(sel_li) == 0 or len(sel_anion_n) == 0 or len(sel_core) == 0:
        raise RuntimeError(
            f"empty selection: Li={len(sel_li)}, anionN={len(sel_anion_n)}, core={len(sel_core)}"
        )

    n_li = len(sel_li)
    n_core = len(sel_core)
    n_tot = n_li + n_core
    li_atom_indices = sel_li.indices
    core_atom_indices = sel_core.indices

    # Map each anion residue to a compact anion id, then each core atom to that id.
    anion_resindex_to_id = {int(resix): k for k, resix in enumerate(sel_anion_n.resindices)}
    core_anion_ids = np.array(
        [anion_resindex_to_id[int(resix)] for resix in sel_core.resindices],
        dtype=np.int32,
    )

    masses = u.atoms.masses
    if masses is None or np.allclose(masses, 0.0):
        masses = np.ones(len(u.atoms), dtype=np.float64)
    else:
        masses = np.asarray(masses, dtype=np.float64).copy()
        li_zero = (u.atoms.names == "LI") & (~np.isfinite(masses) | np.isclose(masses, 0.0))
        masses[li_zero] = 6.941
        masses[~np.isfinite(masses) | np.isclose(masses, 0.0)] = 1.0

    cutoff_a = cutoff_nm * 10.0
    begin_ps = begin_ns * 1000.0
    end_ps = end_ns * 1000.0
    frame_step = max(1, int(stride_frames))

    clusters_by_frame = []
    times_ps = []
    pop_counts = defaultdict(float)

    u.trajectory[0]
    if not np.allclose(u.trajectory.ts.dimensions[3:], [90.0, 90.0, 90.0], atol=1e-3):
        raise RuntimeError("Only orthorhombic boxes are supported by this quick estimator.")
    pos_prev = u.atoms.positions.copy()
    pos_unwrap = pos_prev.copy()

    for iframe, ts in enumerate(u.trajectory):
        if iframe > 0:
            pos_prev = unwrap_update(pos_unwrap, pos_prev, u.atoms.positions, ts.dimensions[:3])
        if iframe % frame_step != 0:
            continue
        if ts.time < begin_ps or ts.time > end_ps:
            continue

        coords = np.vstack([sel_li.positions, sel_core.positions]).astype(np.float32)
        ns = FastNS(cutoff_a, coords, ts.dimensions.astype(np.float32), pbc=True)
        pairs = ns.self_search().get_pairs()
        parent, rank = uf_init(n_tot)
        for a, b in pairs:
            uf_union(parent, rank, int(a), int(b))

        comp_li = defaultdict(list)
        comp_an = defaultdict(set)
        comp_atoms = defaultdict(list)
        for li_pos, atom_ix in enumerate(li_atom_indices):
            root = uf_find(parent, li_pos)
            comp_li[root].append(li_pos)
            comp_atoms[root].append(int(atom_ix))
        for core_pos, atom_ix in enumerate(core_atom_indices):
            root = uf_find(parent, n_li + core_pos)
            comp_an[root].add(int(core_anion_ids[core_pos]))
            comp_atoms[root].append(int(atom_ix))

        frame_clusters = []
        for root in set(comp_li) | set(comp_an):
            li_ids = comp_li.get(root, [])
            an_ids = sorted(comp_an.get(root, set()))
            comp = (len(li_ids), len(an_ids))
            atoms = np.array(comp_atoms[root], dtype=np.int32)
            members = frozenset(list(li_ids) + [100000 + x for x in an_ids])
            com = mass_com(pos_unwrap[atoms], masses[atoms])
            frame_clusters.append({"comp": comp, "members": members, "com": com})
            pop_counts[comp] += 1.0
        clusters_by_frame.append(frame_clusters)
        times_ps.append(float(ts.time))

    if len(clusters_by_frame) < 3:
        raise RuntimeError("too few frames selected")

    n_frames = float(len(clusters_by_frame))
    pop = {comp: count / n_frames for comp, count in pop_counts.items()}
    return clusters_by_frame, np.array(times_ps), pop


def estimate_dij(
    clusters_by_frame,
    times_ps: np.ndarray,
    lags: list[int],
    min_overlap: float,
    min_samples: int,
) -> tuple[dict[tuple[int, int], float], pd.DataFrame]:
    msd_rows = []
    for lag in lags:
        if lag >= len(clusters_by_frame):
            continue
        dt_ps = float(np.mean(times_ps[lag:] - times_ps[:-lag]))
        accum = defaultdict(list)
        for fi in range(0, len(clusters_by_frame) - lag):
            targets = defaultdict(list)
            for c2 in clusters_by_frame[fi + lag]:
                targets[c2["comp"]].append(c2)
            for c1 in clusters_by_frame[fi]:
                members1 = c1["members"]
                if not members1:
                    continue
                best = None
                best_score = 0.0
                for c2 in targets.get(c1["comp"], []):
                    inter = len(members1 & c2["members"])
                    denom = max(1, min(len(members1), len(c2["members"])))
                    score = inter / denom
                    if score > best_score:
                        best_score = score
                        best = c2
                if best is None or best_score < min_overlap:
                    continue
                dr = best["com"] - c1["com"]
                accum[c1["comp"]].append(float(np.dot(dr, dr)))
        for comp, vals in accum.items():
            if len(vals) >= min_samples:
                msd_rows.append(
                    {
                        "i": comp[0],
                        "j": comp[1],
                        "lag_frames": lag,
                        "dt_ps": dt_ps,
                        "n_samples": len(vals),
                        "msd_A2": float(np.mean(vals)),
                    }
                )

    msd_df = pd.DataFrame(msd_rows)
    dij = {}
    if msd_df.empty:
        return dij, msd_df
    for (i, j), g in msd_df.groupby(["i", "j"]):
        g = g.sort_values("dt_ps")
        if len(g) >= 3:
            slope = float(np.polyfit(g["dt_ps"].to_numpy(), g["msd_A2"].to_numpy(), 1)[0])
            d = max(0.0, slope / 6.0 * 1e-4)
        elif len(g) >= 1:
            d_vals = g["msd_A2"].to_numpy() / (6.0 * g["dt_ps"].to_numpy()) * 1e-4
            d = float(np.mean(d_vals))
        else:
            continue
        dij[(int(i), int(j))] = d
    return dij, msd_df


def sigma_full_cne(pop, dij, volume_nm3: float, temp_k: float, z: float = 1.0):
    pref = (E_CHG**2) / (KB * temp_k * (volume_nm3 * 1e-21))
    sigma = 0.0
    rows = []
    covered_q2 = 0.0
    total_q2 = 0.0
    for (i, j), n_ij in sorted(pop.items()):
        q = (i - j) * z
        if q == 0:
            continue
        q2n = q * q * n_ij
        total_q2 += q2n
        d = dij.get((i, j), math.nan)
        term = math.nan
        if math.isfinite(d):
            term = pref * q2n * d
            sigma += term
            covered_q2 += q2n
        rows.append(
            {
                "i": i,
                "j": j,
                "q": q,
                "pop_per_frame": n_ij,
                "q2_pop": q2n,
                "D_ij_cm2_s": d,
                "sigma_term_S_cm": term,
            }
        )
    return sigma, covered_q2 / total_q2 if total_q2 > 0 else math.nan, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--traj-id", type=int, action="append", required=True)
    ap.add_argument("--cutoff-nm", type=float, default=0.28)
    ap.add_argument("--begin-ns", type=float, default=0.0)
    ap.add_argument("--end-ns", type=float, default=100.0)
    ap.add_argument("--stride-frames", type=int, default=1)
    ap.add_argument("--lags", default="1,2,5,10,25,50,100")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--temp-k", type=float, default=353.0)
    args = ap.parse_args()

    root = args.root.resolve()
    refs = ref_table(root)
    out_dir = root / "results" / "full_cne_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)
    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    summary = []

    for tid in args.traj_id:
        run = root / "runs" / f"Traj_{tid}"
        prod = run / "md" / "production"
        gro = prod / "production.gro"
        xtc = prod / "production.xtc"
        tpr = prod / "production.tpr"
        print(f"[full-cNE] Traj_{tid}: loading/clustering", flush=True)
        clusters, times, pop = collect_cluster_trace(
            tpr=tpr,
            xtc=xtc,
            gro=gro,
            cutoff_nm=args.cutoff_nm,
            begin_ns=args.begin_ns,
            end_ns=args.end_ns,
            stride_frames=args.stride_frames,
        )
        print(f"[full-cNE] Traj_{tid}: frames={len(clusters)}, estimating D_ij", flush=True)
        dij, msd_df = estimate_dij(
            clusters,
            times,
            lags=lags,
            min_overlap=args.min_overlap,
            min_samples=args.min_samples,
        )
        volume = gro_volume_nm3(gro)
        sigma, coverage, term_df = sigma_full_cne(pop, dij, volume, args.temp_k)
        row = refs.loc[refs["Trajectory ID"].astype(int) == tid].iloc[0]
        ref = float(row["reference_CONDUCTIVITY_S_cm"])
        raw = float(row["sigma_cNE_raw_S_cm"])
        htp = float(row["sigma_cNE_htpmd_S_cm"])
        signed = math.log10(sigma / ref) if sigma > 0 and ref > 0 else math.nan
        summary.append(
            {
                "Trajectory ID": tid,
                "sample_group": row["sample_group"],
                "reference_CONDUCTIVITY_S_cm": ref,
                "sigma_full_cNE_overlap_S_cm": sigma,
                "ma_loge_full_cNE_overlap": abs(signed) if math.isfinite(signed) else math.nan,
                "signed_log10_error_full_cNE_overlap": signed,
                "fold_error_full_cNE_overlap": max(sigma / ref, ref / sigma) if sigma > 0 and ref > 0 else math.nan,
                "q2_coverage_with_Dij": coverage,
                "n_Dij": len(dij),
                "frames": len(clusters),
                "cutoff_nm": args.cutoff_nm,
                "min_overlap": args.min_overlap,
                "raw_cNE_S_cm": raw,
                "htpmd_cNE_S_cm": htp,
            }
        )
        msd_df.to_csv(out_dir / f"Traj_{tid}_dij_msd.csv", index=False)
        term_df.sort_values("q2_pop", ascending=False).to_csv(
            out_dir / f"Traj_{tid}_terms.csv", index=False
        )
        print(
            f"[full-cNE] Traj_{tid}: sigma={sigma:.6e} S/cm, coverage={coverage:.3f}, n_Dij={len(dij)}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
