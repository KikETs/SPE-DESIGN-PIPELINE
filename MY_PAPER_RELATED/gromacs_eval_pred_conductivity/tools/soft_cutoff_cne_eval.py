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


def infer_resnames(u: mda.Universe) -> tuple[str, str]:
    resnames = {r.resname for r in u.residues}
    if "LI" not in resnames:
        raise RuntimeError(f"LI resname not found: {sorted(resnames)}")
    if "TFSI" in resnames:
        return "LI", "TFSI"
    for rn in sorted(resnames):
        if rn != "LI":
            return "LI", rn
    raise RuntimeError(f"anion resname not found: {sorted(resnames)}")


def accumulate_counts(counts, parent, n_li, core_is_n, max_cluster):
    comp_li = defaultdict(int)
    comp_an = defaultdict(int)
    for i in range(n_li):
        comp_li[uf_find(parent, i)] += 1
    for core_pos, is_n in enumerate(core_is_n):
        if is_n:
            comp_an[uf_find(parent, n_li + core_pos)] += 1
    for root in set(comp_li) | set(comp_an):
        i = comp_li.get(root, 0)
        j = comp_an.get(root, 0)
        if i < max_cluster and j < max_cluster:
            counts[(i, j)] += 1.0


def sigma_from_counts(counts, norm, d_li, d_an, volume_nm3, temp_k, max_cluster):
    pref = (E_CHG**2) / (KB * temp_k * (volume_nm3 * 1e-21))
    sigma = 0.0
    for (i, j), n in counts.items():
        if i == j:
            continue
        if i >= max_cluster or j >= max_cluster:
            continue
        q = float(i - j)
        dij = d_li if i > j else d_an
        sigma += pref * q * q * (n / norm) * dij
    return float(sigma)


def gro_volume_nm3(gro: Path) -> float:
    vals = [float(x) for x in gro.read_text().splitlines()[-1].split()]
    return vals[0] * vals[1] * vals[2]


def smooth_switch_prob(dist_a, r_on_a, r_off_a):
    p = np.zeros_like(dist_a, dtype=np.float64)
    p[dist_a <= r_on_a] = 1.0
    mid = (dist_a > r_on_a) & (dist_a < r_off_a)
    x = (dist_a[mid] - r_on_a) / (r_off_a - r_on_a)
    p[mid] = 1.0 - (3.0 * x * x - 2.0 * x * x * x)
    return p


def evaluate_traj(root: Path, tid: int, args, ref_row, summary_row):
    run = root / "runs" / f"Traj_{tid}"
    prod = run / "md" / "production"
    gro = prod / "production.gro"
    xtc = prod / "production.xtc"
    u = mda.Universe(str(gro), str(xtc))
    cation_res, anion_res = infer_resnames(u)
    sel_li = u.select_atoms(f"resname {cation_res}")
    sel_core = u.select_atoms(f"resname {anion_res} and (name O* or name S* or name N*)")
    sel_anion_n = u.select_atoms(f"resname {anion_res} and name N*")
    if len(sel_li) == 0 or len(sel_core) == 0 or len(sel_anion_n) == 0:
        raise RuntimeError(
            f"empty selection Traj_{tid}: LI={len(sel_li)} core={len(sel_core)} N={len(sel_anion_n)}"
        )

    n_li = len(sel_li)
    n_tot = n_li + len(sel_core)
    core_is_n = np.isin(sel_core.indices, sel_anion_n.indices)
    hard_cutoffs_a = np.array(args.hard_cutoffs_nm, dtype=np.float64) * 10.0
    r_on_a = args.soft_on_nm * 10.0
    r_off_a = args.soft_off_nm * 10.0
    max_search_a = max(float(np.max(hard_cutoffs_a)), r_off_a)
    rng = np.random.default_rng(args.seed + tid)

    hard_counts = defaultdict(float)
    soft_counts = defaultdict(float)
    n_frames = 0

    for ts in u.trajectory:
        if ts.time < args.begin_ns * 1000.0 or ts.time > args.end_ns * 1000.0:
            continue
        coords = np.vstack([sel_li.positions, sel_core.positions]).astype(np.float32)
        box = ts.dimensions.astype(np.float32)
        pairs = FastNS(max_search_a, coords, box, pbc=True).self_search().get_pairs()
        if len(pairs) == 0:
            n_frames += 1
            continue
        delta = coords[pairs[:, 0]] - coords[pairs[:, 1]]
        box_lengths = box[:3].astype(np.float32)
        delta -= box_lengths * np.round(delta / box_lengths)
        dist = np.sqrt(np.sum(delta * delta, axis=1))

        order = np.argsort(dist)
        pairs_sorted = pairs[order]
        dist_sorted = dist[order]
        parent, rank = uf_init(n_tot)
        cursor = 0
        for cutoff_a in hard_cutoffs_a:
            limit = int(np.searchsorted(dist_sorted, cutoff_a, side="right"))
            for a, b in pairs_sorted[cursor:limit]:
                uf_union(parent, rank, int(a), int(b))
            cursor = limit
            accumulate_counts(hard_counts, parent, n_li, core_is_n, args.max_cluster)

        prob = smooth_switch_prob(dist, r_on_a, r_off_a)
        for _ in range(args.soft_samples):
            mask = rng.random(len(prob)) < prob
            parent, rank = uf_init(n_tot)
            for a, b in pairs[mask]:
                uf_union(parent, rank, int(a), int(b))
            accumulate_counts(soft_counts, parent, n_li, core_is_n, args.max_cluster)

        n_frames += 1

    d_li = float(summary_row["D_Li_cm2s"])
    d_an = float(summary_row["D_an_cm2s"])
    volume_nm3 = float(summary_row.get("V_nm3", gro_volume_nm3(gro)))
    temp_k = float(args.temp_k)
    ref = float(ref_row["CONDUCTIVITY"])
    sweep_sigma = sigma_from_counts(
        hard_counts, n_frames * len(hard_cutoffs_a), d_li, d_an, volume_nm3, temp_k, args.max_cluster
    )
    soft_sigma = sigma_from_counts(
        soft_counts, n_frames * args.soft_samples, d_li, d_an, volume_nm3, temp_k, args.max_cluster
    )
    sweep_signed = math.log10(sweep_sigma / ref)
    soft_signed = math.log10(soft_sigma / ref)
    return {
        "Trajectory ID": tid,
        "ref": ref,
        "sweep_sigma": sweep_sigma,
        "sweep_MALogE": abs(sweep_signed),
        "sweep_signed": sweep_signed,
        "soft_sigma": soft_sigma,
        "soft_MALogE": abs(soft_signed),
        "soft_signed": soft_signed,
        "frames": n_frames,
        "soft_samples": args.soft_samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--traj-id", type=int, action="append", required=True)
    ap.add_argument("--max-cluster", type=int, default=50)
    ap.add_argument("--begin-ns", type=float, default=0.0)
    ap.add_argument("--end-ns", type=float, default=100.0)
    ap.add_argument("--hard-cutoffs-nm", default="0.30,0.32,0.34,0.36,0.38")
    ap.add_argument("--soft-on-nm", type=float, default=0.32)
    ap.add_argument("--soft-off-nm", type=float, default=0.40)
    ap.add_argument("--soft-samples", type=int, default=4)
    ap.add_argument("--temp-k", type=float, default=353.0)
    ap.add_argument("--seed", type=int, default=20260505)
    ap.add_argument("--summary-suffix", default="latest_max50")
    args = ap.parse_args()
    args.hard_cutoffs_nm = [float(x) for x in args.hard_cutoffs_nm.split(",") if x.strip()]

    root = args.root.resolve()
    agg = pd.read_csv(root / "runs" / "simulation-trajectory-aggregate.csv")
    agg["Trajectory ID"] = agg["Trajectory ID"].astype(int)
    out = []
    for tid in args.traj_id:
        summary_path = (
            root
            / "runs"
            / f"Traj_{tid}"
            / "analysis"
            / f"conductivity_summary_htpmd_ref_{args.summary_suffix}.csv"
        )
        summary = pd.read_csv(summary_path).iloc[0]
        ref_row = agg.loc[agg["Trajectory ID"] == tid].iloc[0]
        print(f"[soft-cutoff] Traj_{tid}", flush=True)
        out.append(evaluate_traj(root, tid, args, ref_row, summary))

    df = pd.DataFrame(out)
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "soft_cutoff_cne_eval_latest.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
