from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd


KB = 1.380649e-23
E_CHG = 1.602176634e-19


def production_finished(run_dir: Path) -> bool:
    prod = run_dir / "md" / "production"
    log = prod / "production.log"
    return (
        (prod / "production.gro").exists()
        and (prod / "production.xtc").exists()
        and (prod / "production.tpr").exists()
        and log.exists()
        and ("Finished mdrun" in log.read_text(errors="ignore")[-5000:])
    )


def read_nst(run_dir: Path) -> int | None:
    mdp = run_dir / "md" / "production" / "mdout.mdp"
    if not mdp.exists():
        return None
    for line in mdp.read_text(errors="ignore").splitlines():
        text = line.strip()
        if text.startswith("nstxout-compressed"):
            try:
                return int(text.split("=", 1)[1].split()[0])
            except Exception:
                return None
    return None


def gro_indices(gro: Path) -> tuple[list[int], list[int], str]:
    lines = gro.read_text(errors="ignore").splitlines()
    nat = int(lines[1].strip())
    li: list[int] = []
    an_n: list[int] = []
    n_name = "N1"
    for line in lines[2 : 2 + nat]:
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        idx = int(line[15:20])
        if resname == "LI" and atomname.upper().startswith("LI"):
            li.append(idx)
        if resname == "TFSI" and atomname.upper().startswith("N"):
            an_n.append(idx)
            n_name = atomname
    return li, an_n, n_name


def write_index(path: Path, groups: dict[str, list[int]]) -> None:
    chunks: list[str] = []
    for name, values in groups.items():
        chunks.append(f"[ {name} ]")
        for start in range(0, len(values), 15):
            chunks.append(" ".join(str(v) for v in values[start : start + 15]))
    path.write_text("\n".join(chunks) + "\n")


def run_msd(
    gmx: str,
    prod: Path,
    outdir: Path,
    group: str,
    outname: str,
    trestart_ps: float,
) -> Path:
    out_xvg = outdir / outname
    log_path = outdir / f"{outname}.log"
    cmd = [
        gmx,
        "msd",
        "-f",
        str(prod / "production.xtc"),
        "-s",
        str(prod / "production.tpr"),
        "-n",
        str(outdir / "index.ndx"),
        "-o",
        str(out_xvg),
        "-trestart",
        f"{trestart_ps:g}",
    ]
    proc = subprocess.run(
        cmd,
        input=f"{group}\n",
        text=True,
        cwd=outdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode != 0 or not out_xvg.exists():
        raise RuntimeError(f"gmx msd failed for {group}; see {log_path}")
    return out_xvg


def xvg_to_df(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except Exception:
            pass
    if not rows:
        raise RuntimeError(f"No numeric data in {path}")
    return pd.DataFrame(rows, columns=["t_ps", "msd_nm2"])


def diffusion_from_xvg(path: Path, begin_ns: float = 0.0, end_ns: float | None = None) -> tuple[float, float]:
    df = xvg_to_df(path)
    t_ns = df["t_ps"].to_numpy() / 1000.0
    y = df["msd_nm2"].to_numpy()
    if end_ns is None:
        end_ns = float(np.nanmax(t_ns))
    mask = (t_ns >= begin_ns) & (t_ns <= end_ns)
    if int(mask.sum()) < 5:
        mask = np.isfinite(t_ns) & np.isfinite(y)
    slope = np.polyfit(t_ns[mask], y[mask], 1)[0]
    return float((slope / 6.0) * 1e-5), float(end_ns)


def first_last_diffusivities(prod: Path) -> tuple[float, float, float]:
    import MDAnalysis as mda

    def center_of_mass(coords: np.ndarray, masses: np.ndarray | None) -> np.ndarray:
        if masses is None:
            return coords.mean(axis=0)
        return (coords * masses[:, None]).sum(axis=0) / masses.sum()

    u = mda.Universe(str(prod / "production.gro"), str(prod / "production.xtc"))
    sel_li = u.select_atoms("resname LI")
    sel_n = u.select_atoms("resname TFSI and name N*")
    if len(sel_li) == 0 or len(sel_n) == 0:
        raise RuntimeError(f"empty diffusion selections LI={len(sel_li)} TFSI_N={len(sel_n)}")

    masses = None
    try:
        masses = u.atoms.masses.astype(float)
        if np.allclose(masses, 0.0):
            masses = None
    except Exception:
        masses = None

    u.trajectory[0]
    pos_prev = u.atoms.positions.astype(float).copy()
    pos_unwrap = pos_prev.copy()
    com0 = center_of_mass(pos_unwrap, masses)
    li_start = pos_unwrap[sel_li.indices] - com0
    an_start = pos_unwrap[sel_n.indices] - com0
    n_frames = len(u.trajectory)

    for iframe, ts in enumerate(u.trajectory[1:], start=1):
        cur = u.atoms.positions.astype(float)
        box = ts.dimensions[:3].astype(float)
        delta = cur - pos_prev
        delta -= box * np.round(delta / box)
        pos_unwrap += delta
        pos_prev = cur
        if iframe % 10000 == 0:
            print(f"first-last unwrap scanned {iframe + 1} frames", flush=True)

    com_end = center_of_mass(pos_unwrap, masses)
    total_time_ps = float(u.trajectory.dt) * (n_frames - 1)
    if total_time_ps <= 0:
        raise RuntimeError("invalid trajectory time for first-last diffusivity")
    li_disp = (pos_unwrap[sel_li.indices] - com_end) - li_start
    an_disp = (pos_unwrap[sel_n.indices] - com_end) - an_start
    d_li = float(np.mean(np.sum(li_disp * li_disp, axis=1)) / (6.0 * total_time_ps) * 1e-4)
    d_an = float(np.mean(np.sum(an_disp * an_disp, axis=1)) / (6.0 * total_time_ps) * 1e-4)
    return d_li, d_an, total_time_ps / 1000.0


def add_pop_from_codes(pop: np.ndarray, codes: np.ndarray, n_li: int, n_an: int) -> None:
    if codes.size == 0:
        pop[1, 0] += n_li
        pop[0, 1] += n_an
        return
    n_tot = n_li + n_an
    parent = np.arange(n_tot, dtype=np.int32)
    rank = np.zeros(n_tot, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    li = (codes // n_an).astype(np.int32, copy=False)
    an = (codes % n_an).astype(np.int32, copy=False)
    for li_i, an_i in zip(li, an):
        union(int(li_i), n_li + int(an_i))

    comp_li: dict[int, int] = defaultdict(int)
    comp_an: dict[int, int] = defaultdict(int)
    active_li = np.unique(li)
    active_an = np.unique(an)
    for li_i in active_li:
        comp_li[find(int(li_i))] += 1
    for an_i in active_an:
        comp_an[find(n_li + int(an_i))] += 1
    for root in set(comp_li) | set(comp_an):
        i = comp_li.get(root, 0)
        j = comp_an.get(root, 0)
        if i < pop.shape[0] and j < pop.shape[1]:
            pop[i, j] += 1.0
    free_li = n_li - len(active_li)
    free_an = n_an - len(active_an)
    if free_li:
        pop[1, 0] += free_li
    if free_an:
        pop[0, 1] += free_an


def sigma_from_pop(pop: np.ndarray, d_li: float, d_an: float, v_nm3: float, max_cluster: int, temp_k: float) -> float:
    pref = (E_CHG**2) / (KB * temp_k * (v_nm3 * 1e-21))
    sigma = 0.0
    for i in range(min(max_cluster, pop.shape[0])):
        for j in range(min(max_cluster, pop.shape[1])):
            nij = float(pop[i, j])
            if nij == 0.0 or i == j:
                continue
            q = i - j
            d_eff = d_li if i > j else d_an
            sigma += pref * (q * q) * nij * d_eff
    return float(sigma)


def analyze_contacts(
    tid: int,
    prod: Path,
    outdir: Path,
    d_li: float,
    d_an: float,
    v_nm3: float,
    sigma_ref: float,
    sample_group: str,
    cutoff_nm: float,
    thresholds_ps: list[float],
    max_cluster: int,
    temp_k: float,
) -> tuple[list[dict], dict]:
    import MDAnalysis as mda
    from MDAnalysis.lib.nsgrid import FastNS

    u = mda.Universe(str(prod / "production.gro"), str(prod / "production.xtc"))
    sel_li = u.select_atoms("resname LI")
    sel_o = u.select_atoms("resname TFSI and name O*")
    sel_n = u.select_atoms("resname TFSI and name N*")
    if len(sel_li) == 0 or len(sel_o) == 0 or len(sel_n) == 0:
        raise RuntimeError(f"Traj_{tid}: empty selection LI={len(sel_li)} O={len(sel_o)} N={len(sel_n)}")

    n_li = len(sel_li)
    n_an = len(sel_n)
    pop_shape = (min(n_li + 1, max_cluster), min(n_an + 1, max_cluster))
    an_map = {residx: i for i, residx in enumerate(list(sel_n.resindices))}
    o_to_an = np.array([an_map[atom.resindex] for atom in sel_o], dtype=np.int32)
    cutoff_a = cutoff_nm * 10.0
    dt_ps = float(u.trajectory.dt)
    if not np.isfinite(dt_ps) or dt_ps <= 0:
        dt_ps = 1.0

    frame_codes: list[np.ndarray] = []
    active: dict[int, tuple[int, int]] = {}
    run_records: list[tuple[int, int, int, int, float]] = []
    start_time = time.time()
    for iframe, ts in enumerate(u.trajectory):
        coords = np.vstack([sel_li.positions, sel_o.positions]).astype(np.float32)
        pairs = FastNS(cutoff_a, coords, ts.dimensions.astype(np.float32), pbc=True).self_search().get_pairs()
        if pairs.size:
            a = pairs[:, 0]
            b = pairs[:, 1]
            mask = ((a < n_li) & (b >= n_li)) | ((b < n_li) & (a >= n_li))
            pair_contacts = pairs[mask]
            if pair_contacts.size:
                li_idx = np.where(pair_contacts[:, 0] < n_li, pair_contacts[:, 0], pair_contacts[:, 1])
                o_idx = np.where(
                    pair_contacts[:, 0] >= n_li,
                    pair_contacts[:, 0] - n_li,
                    pair_contacts[:, 1] - n_li,
                )
                an_idx = o_to_an[o_idx.astype(np.int32)]
                codes = np.unique(li_idx.astype(np.int32) * n_an + an_idx.astype(np.int32))
            else:
                codes = np.empty(0, dtype=np.int32)
        else:
            codes = np.empty(0, dtype=np.int32)
        frame_codes.append(codes.astype(np.int32, copy=False))

        current = set(int(code) for code in codes)
        for code in list(active.keys()):
            if code not in current:
                first, last = active.pop(code)
                run_records.append((code, first, last, last - first + 1, (last - first) * dt_ps))
        for code in current:
            if code in active:
                first, _ = active[code]
                active[code] = (first, iframe)
            else:
                active[code] = (iframe, iframe)
        if (iframe + 1) % 10000 == 0:
            print(f"Traj_{tid}: contacts scanned {iframe + 1} frames", flush=True)

    for code, (first, last) in active.items():
        run_records.append((code, first, last, last - first + 1, (last - first) * dt_ps))

    n_frames = len(frame_codes)
    pops: dict[str, np.ndarray] = {"instant_molecular_edges": np.zeros(pop_shape, dtype=float)}
    for threshold in thresholds_ps:
        pops[f"persist_span_ge_{threshold:g}ps"] = np.zeros(pop_shape, dtype=float)

    starts = {threshold: [[] for _ in range(n_frames)] for threshold in thresholds_ps}
    ends = {threshold: [[] for _ in range(n_frames + 1)] for threshold in thresholds_ps}
    for code, first, last, _length, tau_span in run_records:
        for threshold in thresholds_ps:
            if tau_span >= threshold:
                starts[threshold][first].append(code)
                if last + 1 <= n_frames:
                    ends[threshold][last + 1].append(code)

    active_bool = {threshold: np.zeros(n_li * n_an, dtype=bool) for threshold in thresholds_ps}
    for iframe, codes in enumerate(frame_codes):
        add_pop_from_codes(pops["instant_molecular_edges"], codes, n_li, n_an)
        for threshold in thresholds_ps:
            if ends[threshold][iframe]:
                active_bool[threshold][np.array(ends[threshold][iframe], dtype=np.int32)] = False
            if starts[threshold][iframe]:
                active_bool[threshold][np.array(starts[threshold][iframe], dtype=np.int32)] = True
            filt = codes[active_bool[threshold][codes]] if codes.size else codes
            add_pop_from_codes(pops[f"persist_span_ge_{threshold:g}ps"], filt, n_li, n_an)
        if (iframe + 1) % 10000 == 0:
            print(f"Traj_{tid}: clustered {iframe + 1} frames", flush=True)

    cne_rows = []
    ii = np.arange(pop_shape[0], dtype=float)[:, None]
    jj = np.arange(pop_shape[1], dtype=float)[None, :]
    for variant, pop_raw in pops.items():
        pop = pop_raw / float(n_frames)
        sigma = sigma_from_pop(pop, d_li, d_an, v_nm3, max_cluster, temp_k)
        threshold = 0.0 if variant == "instant_molecular_edges" else float(variant.split("_ge_")[1].replace("ps", ""))
        cne_rows.append(
            {
                "Trajectory ID": tid,
                "sample_group": sample_group,
                "variant": variant,
                "cutoff_nm": cutoff_nm,
                "persistence_threshold_ps": threshold,
                "max_cluster": max_cluster,
                "frames": n_frames,
                "dt_ps": dt_ps,
                "N_LI": n_li,
                "N_AN": n_an,
                "D_Li_cm2s": d_li,
                "D_an_cm2s": d_an,
                "V_nm3": v_nm3,
                "sigma_ref_S_cm": sigma_ref,
                "sigma_cNE_S_cm": sigma,
                "pred_ref_ratio": sigma / sigma_ref if sigma_ref > 0 else np.nan,
                "ma_loge_sigma": abs(math.log10(sigma / sigma_ref)) if sigma > 0 and sigma_ref > 0 else np.nan,
                "alpha_Li_sum": float((ii * pop).sum()),
                "alpha_AN_sum": float((jj * pop).sum()),
                "scan_minutes": (time.time() - start_time) / 60.0,
            }
        )

    life = pd.DataFrame(
        run_records,
        columns=["contact_code", "start_frame", "end_frame", "n_frames_contact", "tau_span_ps"],
    )
    raw_path = outdir / f"Traj_{tid}_contact_lifetimes.csv.gz"
    life.to_csv(raw_path, index=False, compression="gzip")
    taus = life["tau_span_ps"].to_numpy(float) if not life.empty else np.array([], dtype=float)
    if taus.size:
        tau_i = np.rint(taus / dt_ps).astype(int)
        uniq, counts = np.unique(tau_i, return_counts=True)
        pd.DataFrame(
            {"tau_span_ps": uniq.astype(float) * dt_ps, "count": counts, "P_tau": counts / counts.sum()}
        ).to_csv(outdir / f"Traj_{tid}_P_tau.csv", index=False)
        max_i = int(uniq.max())
        grid = np.arange(max_i + 1)
        sorted_i = np.sort(tau_i)
        n_ge = len(sorted_i) - np.searchsorted(sorted_i, grid, side="left")
        pd.DataFrame({"t_ps": grid.astype(float) * dt_ps, "n_ge_t": n_ge, "S_t": n_ge / len(sorted_i)}).to_csv(
            outdir / f"Traj_{tid}_survival_S_t.csv", index=False
        )
        tau_sum = float(taus.sum())
        sp = {
            "Trajectory ID": tid,
            "sample_group": sample_group,
            "cutoff_nm": cutoff_nm,
            "dt_ps": dt_ps,
            "n_contact_runs": int(taus.size),
            "median_tau_ps": float(np.median(taus)),
            "mean_tau_ps": float(np.mean(taus)),
            "p90_tau_ps": float(np.percentile(taus, 90)),
            "p95_tau_ps": float(np.percentile(taus, 95)),
            "p99_tau_ps": float(np.percentile(taus, 99)),
            "max_tau_ps": float(np.max(taus)),
            "run_frac_ge_5ps": float(np.mean(taus >= 5.0)),
            "run_frac_ge_10ps": float(np.mean(taus >= 10.0)),
            "run_frac_ge_20ps": float(np.mean(taus >= 20.0)),
            "contact_time_frac_in_runs_ge_20ps": float(taus[taus >= 20.0].sum() / tau_sum) if tau_sum > 0 else np.nan,
            "raw_lifetime_csv_gz": str(raw_path),
        }
    else:
        sp = {
            "Trajectory ID": tid,
            "sample_group": sample_group,
            "cutoff_nm": cutoff_nm,
            "dt_ps": dt_ps,
            "n_contact_runs": 0,
        }
    return cne_rows, sp


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = percentile / 100.0 * weights.sum()
    idx = np.searchsorted(np.cumsum(weights), cutoff, side="left")
    return float(values[min(idx, len(values) - 1)])


def threshold_suffix(thresholds_ps: list[float]) -> str:
    parts = ["0"]
    parts.extend(f"{threshold:g}".replace(".", "p") for threshold in thresholds_ps)
    return "_".join(parts) + "ps"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--targets", nargs="+", type=int, required=True)
    parser.add_argument("--label", default="current_completed")
    parser.add_argument("--cutoff-nm", type=float, default=0.28)
    parser.add_argument("--thresholds-ps", nargs="*", type=float, default=[20.0])
    parser.add_argument("--max-cluster", type=int, default=101)
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--trestart-ps", type=float, default=1000.0)
    parser.add_argument("--diffusion-mode", choices=["msd", "first_last"], default="first_last")
    parser.add_argument("--gmx", default=os.environ.get("GMX", "gmx"))
    args = parser.parse_args()

    base = args.base.resolve()
    runs = base / "runs"
    results = base / "results"
    outdir = results / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(results / "sample_manifest.csv").set_index("Trajectory ID")
    suffix = threshold_suffix(args.thresholds_ps)

    valid = []
    skipped = []
    for tid in args.targets:
        run_dir = runs / f"Traj_{tid}"
        if production_finished(run_dir) and read_nst(run_dir) == 500:
            valid.append(tid)
        else:
            skipped.append({"Trajectory ID": tid, "finished": production_finished(run_dir), "nstxout_compressed": read_nst(run_dir)})
    print(f"valid targets: {valid}", flush=True)
    if skipped:
        pd.DataFrame(skipped).to_csv(outdir / "skipped_not_complete.csv", index=False)
        print(f"skipped: {skipped}", flush=True)

    all_cne: list[dict] = []
    all_sp: list[dict] = []
    runtime_rows: list[dict] = []
    for tid in valid:
        t0 = time.time()
        run_dir = runs / f"Traj_{tid}"
        prod = run_dir / "md" / "production"
        msd_dir = outdir / f"Traj_{tid}_msd"
        msd_dir.mkdir(parents=True, exist_ok=True)
        if args.diffusion_mode == "msd":
            li_idx, an_n_idx, n_name = gro_indices(prod / "production.gro")
            write_index(msd_dir / "index.ndx", {"LI": li_idx, "TFSI_N": an_n_idx})
            print(f"Traj_{tid}: MSD start LI={len(li_idx)} TFSI_N={len(an_n_idx)} N={n_name}", flush=True)
            li_xvg = run_msd(args.gmx, prod, msd_dir, "LI", "msd_li.xvg", args.trestart_ps)
            an_xvg = run_msd(args.gmx, prod, msd_dir, "TFSI_N", "msd_tfsi_N.xvg", args.trestart_ps)
            d_li, end_ns = diffusion_from_xvg(li_xvg)
            d_an, _ = diffusion_from_xvg(an_xvg, end_ns=end_ns)
        else:
            print(f"Traj_{tid}: first-last diffusivity start", flush=True)
            d_li, d_an, end_ns = first_last_diffusivities(prod)
        box_vals = [float(x) for x in (prod / "production.gro").read_text(errors="ignore").splitlines()[-1].split()[:3]]
        v_nm3 = box_vals[0] * box_vals[1] * box_vals[2]
        sample = manifest.loc[tid].to_dict()
        print(
            f"Traj_{tid}: D_Li={d_li:.3e} D_TFSI={d_an:.3e} V={v_nm3:.2f} end={end_ns:.2f} ns; cNE/SP start",
            flush=True,
        )
        cne_rows, sp = analyze_contacts(
            tid,
            prod,
            outdir,
            d_li,
            d_an,
            v_nm3,
            float(sample["CONDUCTIVITY"]),
            str(sample.get("sample_group", "")),
            args.cutoff_nm,
            args.thresholds_ps,
            args.max_cluster,
            args.temperature_k,
        )
        all_cne.extend(cne_rows)
        all_sp.append(sp)
        runtime_rows.append({"Trajectory ID": tid, "analysis_minutes": (time.time() - t0) / 60.0, "analysis_end_ns": end_ns})
        pd.DataFrame(all_cne).sort_values(["Trajectory ID", "persistence_threshold_ps"]).to_csv(
            outdir / f"cne_persistence_core_{suffix}.csv", index=False
        )
        pd.DataFrame(all_sp).sort_values("Trajectory ID").to_csv(outdir / "contact_lifetime_summary.csv", index=False)
        pd.DataFrame(runtime_rows).to_csv(outdir / "analysis_runtime.csv", index=False)
        print(f"Traj_{tid}: done in {(time.time() - t0) / 60.0:.1f} min", flush=True)

    if all_sp:
        combined: dict[int, int] = defaultdict(int)
        for tid in valid:
            p_tau = outdir / f"Traj_{tid}_P_tau.csv"
            if not p_tau.exists():
                continue
            h = pd.read_csv(p_tau)
            for tau, count in zip(h["tau_span_ps"], h["count"]):
                combined[int(round(float(tau)))] += int(count)
        if combined:
            keys = np.array(sorted(combined), dtype=int)
            counts = np.array([combined[int(key)] for key in keys], dtype=np.int64)
            total = int(counts.sum())
            tau_total = float((keys * counts).sum())
            ge20 = keys >= 20
            combined_row = {
                "Trajectory ID": "combined",
                "sample_group": "combined",
                "cutoff_nm": args.cutoff_nm,
                "dt_ps": 1.0,
                "n_contact_runs": total,
                "median_tau_ps": weighted_percentile(keys.astype(float), counts, 50.0),
                "mean_tau_ps": tau_total / total if total else np.nan,
                "p90_tau_ps": weighted_percentile(keys.astype(float), counts, 90.0),
                "p95_tau_ps": weighted_percentile(keys.astype(float), counts, 95.0),
                "p99_tau_ps": weighted_percentile(keys.astype(float), counts, 99.0),
                "max_tau_ps": float(keys.max()),
                "run_frac_ge_5ps": float(counts[keys >= 5].sum() / total),
                "run_frac_ge_10ps": float(counts[keys >= 10].sum() / total),
                "run_frac_ge_20ps": float(counts[keys >= 20].sum() / total),
                "contact_time_frac_in_runs_ge_20ps": float((keys[ge20] * counts[ge20]).sum() / tau_total) if tau_total else np.nan,
            }
            pd.concat([pd.DataFrame(all_sp), pd.DataFrame([combined_row])], ignore_index=True).to_csv(
                outdir / "contact_lifetime_summary_with_combined.csv", index=False
            )

    print(f"SAVED {outdir}", flush=True)


if __name__ == "__main__":
    main()
