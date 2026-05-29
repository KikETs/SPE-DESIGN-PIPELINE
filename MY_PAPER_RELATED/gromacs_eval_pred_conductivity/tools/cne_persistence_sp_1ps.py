from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import math
import os
import time

import numpy as np
import pandas as pd


KB = 1.380649e-23
E_CHG = 1.602176634e-19
TEMP_K = 353.0


def read_sample_manifest(results_dir: Path) -> dict[int, dict]:
    path = results_dir / "sample_manifest.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        try:
            tid = int(row["Trajectory ID"])
        except Exception:
            continue
        out[tid] = {
            "sample_group": row.get("sample_group", np.nan),
            "sigma_ref_S_cm": float(row.get("CONDUCTIVITY", np.nan)),
            "tplus_ref": float(row.get("Transference Number", np.nan)),
            "smiles": row.get("SMILES", ""),
        }
    return out


def read_mdp_nst(run_dir: Path) -> int | None:
    mdp = run_dir / "md" / "production" / "mdout.mdp"
    if not mdp.exists():
        return None
    for line in mdp.read_text(errors="ignore").splitlines():
        text = line.strip()
        if text.startswith("nstxout-compressed"):
            try:
                return int(text.split("=")[1].split()[0])
            except Exception:
                return None
    return None


def threshold_suffix(thresholds_ps: list[float]) -> str:
    parts = ["0"]
    parts.extend(f"{threshold:g}".replace(".", "p") for threshold in thresholds_ps)
    return "_".join(parts) + "ps"


def production_finished(run_dir: Path) -> bool:
    gro = run_dir / "md" / "production" / "production.gro"
    log = run_dir / "md" / "production" / "production.log"
    if not gro.exists() or not log.exists():
        return False
    tail = log.read_text(errors="ignore")[-5000:]
    return "Finished mdrun" in tail or "Writing final coordinates." in tail


def read_inputs(run_dir: Path) -> dict:
    path = run_dir / "analysis" / "conductivity_summary_htpmd_ref.csv"
    if not path.exists():
        path = run_dir / "analysis" / "conductivity_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing conductivity summary: {run_dir}")
    row = pd.read_csv(path).iloc[0]

    def get(*names: str, default=np.nan):
        for name in names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return default

    return {
        "D_Li_cm2s": float(get("D_Li_cm2s", "D_Li(cm^2/s)")),
        "D_an_cm2s": float(get("D_an_cm2s", "D_an(cm^2/s)")),
        "V_nm3": float(get("V_nm3", "V(nm^3)")),
        "input_summary": str(path),
    }


def sigma_from_pop(
    pop: np.ndarray,
    d_li: float,
    d_an: float,
    volume_nm3: float,
    max_cluster: int,
    temp_k: float = TEMP_K,
) -> float:
    v_cm3 = volume_nm3 * 1e-21
    pref = (E_CHG**2) / (KB * temp_k * v_cm3)
    sigma = 0.0
    imax = min(max_cluster, pop.shape[0])
    jmax = min(max_cluster, pop.shape[1])
    for i in range(imax):
        for j in range(jmax):
            nij = float(pop[i, j])
            if nij == 0.0 or i == j:
                continue
            q = i - j
            d_eff = d_li if i > j else d_an
            sigma += pref * (q * q) * nij * d_eff
    return float(sigma)


def add_pop_from_codes(pop: np.ndarray, codes: np.ndarray, n_li: int, n_an: int) -> None:
    if codes.size == 0:
        pop[1, 0] += n_li
        pop[0, 1] += n_an
        return

    n_tot = n_li + n_an
    parent = np.arange(n_tot, dtype=np.int16)
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

    li = (codes // n_an).astype(np.int16, copy=False)
    an = (codes % n_an).astype(np.int16, copy=False)
    for li_i, an_i in zip(li, an):
        union(int(li_i), n_li + int(an_i))

    active_li = np.unique(li)
    active_an = np.unique(an)
    comp_li: dict[int, int] = defaultdict(int)
    comp_an: dict[int, int] = defaultdict(int)
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


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    if len(values) == 0 or weights.sum() == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = percentile / 100.0 * weights.sum()
    idx = np.searchsorted(np.cumsum(weights), cutoff, side="left")
    return float(values[min(idx, len(values) - 1)])


def analyze_one(
    tid: int,
    base: Path,
    cutoff_nm: float,
    thresholds_ps: list[float],
    max_cluster: int,
    sample_map: dict[int, dict],
    out_sp_dir: Path,
) -> dict:
    import MDAnalysis as mda
    from MDAnalysis.lib.nsgrid import FastNS

    start_time = time.time()
    run_dir = base / "runs" / f"Traj_{tid}"
    gro = run_dir / "md" / "production" / "production.gro"
    xtc = run_dir / "md" / "production" / "production.xtc"
    if not gro.exists() or not xtc.exists():
        raise FileNotFoundError(f"Traj_{tid}: missing production.gro/xtc")

    nst = read_mdp_nst(run_dir)
    if nst != 500:
        raise RuntimeError(f"Traj_{tid}: output stride is not 1 ps (nstxout-compressed={nst})")

    inputs = read_inputs(run_dir)
    sample = sample_map.get(tid, {})
    sigma_ref = float(sample.get("sigma_ref_S_cm", np.nan))
    group = sample.get("sample_group", np.nan)

    u = mda.Universe(str(gro), str(xtc))
    sel_li = u.select_atoms("resname LI")
    sel_o = u.select_atoms("resname TFSI and name O*")
    sel_n = u.select_atoms("resname TFSI and name N*")
    if len(sel_li) == 0 or len(sel_o) == 0 or len(sel_n) == 0:
        raise RuntimeError(
            f"Traj_{tid}: empty selection LI={len(sel_li)} O={len(sel_o)} N={len(sel_n)}"
        )

    n_li = len(sel_li)
    n_an = len(sel_n)
    pop_shape = (min(n_li + 1, max_cluster), min(n_an + 1, max_cluster))
    an_map = {residx: i for i, residx in enumerate(list(sel_n.resindices))}
    o_to_an = np.array([an_map[atom.resindex] for atom in sel_o], dtype=np.int16)
    cutoff_a = cutoff_nm * 10.0

    frame_codes: list[np.ndarray] = []
    active: dict[int, tuple[int, int]] = {}
    run_records: list[tuple[int, int, int, int, float]] = []
    dt_ps = float(u.trajectory.dt)
    if not np.isfinite(dt_ps) or dt_ps <= 0:
        dt_ps = 1.0

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
            print(f"Traj_{tid}: scanned {iframe + 1} frames", flush=True)

    n_frames = len(frame_codes)
    for code, (first, last) in active.items():
        run_records.append((code, first, last, last - first + 1, (last - first) * dt_ps))

    pops: dict[str, np.ndarray] = {
        "instant_molecular_edges": np.zeros(pop_shape, dtype=np.float64)
    }
    for threshold in thresholds_ps:
        pops[f"persist_span_ge_{threshold:g}ps"] = np.zeros(pop_shape, dtype=np.float64)

    starts = {thr: [[] for _ in range(n_frames)] for thr in thresholds_ps}
    ends = {thr: [[] for _ in range(n_frames + 1)] for thr in thresholds_ps}
    for code, first, last, _length, tau_span in run_records:
        for threshold in thresholds_ps:
            if tau_span >= threshold:
                starts[threshold][first].append(code)
                if last + 1 <= n_frames:
                    ends[threshold][last + 1].append(code)

    active_bool = {thr: np.zeros(n_li * n_an, dtype=bool) for thr in thresholds_ps}
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
    for variant, pop in pops.items():
        pop /= float(n_frames)
        sigma = sigma_from_pop(pop, inputs["D_Li_cm2s"], inputs["D_an_cm2s"], inputs["V_nm3"], max_cluster)
        ma_loge = abs(math.log10(sigma / sigma_ref)) if sigma > 0 and sigma_ref > 0 else np.nan
        ii = np.arange(pop.shape[0], dtype=float)[:, None]
        jj = np.arange(pop.shape[1], dtype=float)[None, :]
        cne_rows.append(
            {
                "Trajectory ID": tid,
                "sample_group": group,
                "variant": variant,
                "cutoff_nm": cutoff_nm,
                "persistence_threshold_ps": (
                    0.0 if variant == "instant_molecular_edges" else float(variant.split("_ge_")[1].replace("ps", ""))
                ),
                "max_cluster": max_cluster,
                "frames": n_frames,
                "dt_ps": dt_ps,
                "N_LI": n_li,
                "N_AN": n_an,
                "D_Li_cm2s": inputs["D_Li_cm2s"],
                "D_an_cm2s": inputs["D_an_cm2s"],
                "V_nm3": inputs["V_nm3"],
                "sigma_ref_S_cm": sigma_ref,
                "sigma_cNE_S_cm": sigma,
                "pred_ref_ratio": sigma / sigma_ref if sigma_ref > 0 else np.nan,
                "ma_loge_sigma": ma_loge,
                "alpha_Li_sum": float((ii * pop).sum()),
                "alpha_AN_sum": float((jj * pop).sum()),
                "input_summary": inputs["input_summary"],
            }
        )

    out_sp_dir.mkdir(parents=True, exist_ok=True)
    life = pd.DataFrame(
        run_records,
        columns=["contact_code", "start_frame", "end_frame", "n_frames_contact", "tau_span_ps"],
    )
    if not life.empty:
        codes_arr = life["contact_code"].to_numpy(dtype=np.int64)
        life["li_index"] = (codes_arr // n_an).astype(int)
        life["anion_index"] = (codes_arr % n_an).astype(int)
        life["start_ps"] = life["start_frame"].astype(float) * dt_ps
        life["end_ps"] = life["end_frame"].astype(float) * dt_ps
    raw_path = out_sp_dir / f"Traj_{tid}_contact_lifetimes.csv.gz"
    life.to_csv(raw_path, index=False, compression="gzip")

    taus = life["tau_span_ps"].to_numpy(float) if not life.empty else np.array([], dtype=float)
    if taus.size:
        tau_int = np.rint(taus / dt_ps).astype(int)
        uniq, counts = np.unique(tau_int, return_counts=True)
        p_tau = pd.DataFrame(
            {"tau_span_ps": uniq.astype(float) * dt_ps, "count": counts, "P_tau": counts / counts.sum()}
        )
        p_tau.to_csv(out_sp_dir / f"Traj_{tid}_P_tau.csv", index=False)
        max_i = int(tau_int.max())
        grid_i = np.arange(0, max_i + 1, dtype=int)
        sorted_i = np.sort(tau_int)
        n_ge = len(sorted_i) - np.searchsorted(sorted_i, grid_i, side="left")
        survival = pd.DataFrame(
            {"t_ps": grid_i.astype(float) * dt_ps, "n_ge_t": n_ge, "S_t": n_ge / len(sorted_i)}
        )
        survival.to_csv(out_sp_dir / f"Traj_{tid}_survival_S_t.csv", index=False)

        tau_sum = float(np.sum(taus))
        summary = {
            "Trajectory ID": tid,
            "sample_group": group,
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
            "contact_time_frac_in_runs_ge_20ps": float(np.sum(taus[taus >= 20.0]) / tau_sum) if tau_sum > 0 else np.nan,
            "raw_lifetime_csv_gz": str(raw_path),
        }
        hist_pairs = list(zip(uniq.astype(int).tolist(), counts.astype(int).tolist()))
    else:
        summary = {
            "Trajectory ID": tid,
            "sample_group": group,
            "cutoff_nm": cutoff_nm,
            "dt_ps": dt_ps,
            "n_contact_runs": 0,
            "median_tau_ps": np.nan,
            "mean_tau_ps": np.nan,
            "p90_tau_ps": np.nan,
            "p95_tau_ps": np.nan,
            "p99_tau_ps": np.nan,
            "max_tau_ps": np.nan,
            "run_frac_ge_5ps": np.nan,
            "run_frac_ge_10ps": np.nan,
            "run_frac_ge_20ps": np.nan,
            "contact_time_frac_in_runs_ge_20ps": np.nan,
            "raw_lifetime_csv_gz": str(raw_path),
        }
        hist_pairs = []

    per_run_cne = run_dir / "analysis_cutoff_0p28_1ps" / f"cne_persistence_core_{threshold_suffix(thresholds_ps)}.csv"
    per_run_cne.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cne_rows).to_csv(per_run_cne, index=False)

    best = min(cne_rows, key=lambda row: row["ma_loge_sigma"] if np.isfinite(row["ma_loge_sigma"]) else 999.0)
    print(
        f"Traj_{tid}: done in {(time.time() - start_time) / 60:.1f} min; "
        f"best {best['variant']} MALogE={best['ma_loge_sigma']:.3f}",
        flush=True,
    )
    return {"tid": tid, "cne_rows": cne_rows, "sp_summary": summary, "hist_pairs": hist_pairs}


def combine_sp_from_hist(
    hist_by_tid: dict[int, list[tuple[int, int]]],
    summaries: list[dict],
    out_dir: Path,
    label: str,
    cutoff_nm: float,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_summary = pd.DataFrame(summaries).sort_values("Trajectory ID") if summaries else pd.DataFrame()
    if hist_by_tid:
        combined: dict[int, int] = defaultdict(int)
        for pairs in hist_by_tid.values():
            for tau_i, count in pairs:
                combined[int(tau_i)] += int(count)
        keys = np.array(sorted(combined), dtype=int)
        counts = np.array([combined[int(key)] for key in keys], dtype=np.int64)
        p_tau = pd.DataFrame({"tau_span_ps": keys.astype(float), "count": counts, "P_tau": counts / counts.sum()})
        p_tau.to_csv(out_dir / f"{label}_P_tau.csv", index=False)

        max_i = int(keys.max()) if len(keys) else 0
        grid = np.arange(0, max_i + 1, dtype=int)
        count_by_tau = np.zeros(max_i + 1, dtype=np.int64)
        count_by_tau[keys] = counts
        n_ge = np.cumsum(count_by_tau[::-1])[::-1]
        survival = pd.DataFrame({"t_ps": grid.astype(float), "n_ge_t": n_ge, "S_t": n_ge / n_ge[0] if n_ge[0] else np.nan})
        survival.to_csv(out_dir / f"{label}_survival_S_t.csv", index=False)

        total_runs = int(counts.sum())
        total_tau = float((keys * counts).sum())
        ge20 = keys >= 20
        combined_summary = {
            "Trajectory ID": "combined",
            "sample_group": "combined",
            "cutoff_nm": cutoff_nm,
            "dt_ps": 1.0,
            "n_contact_runs": total_runs,
            "median_tau_ps": weighted_percentile(keys.astype(float), counts, 50.0),
            "mean_tau_ps": float(total_tau / total_runs) if total_runs else np.nan,
            "p90_tau_ps": weighted_percentile(keys.astype(float), counts, 90.0),
            "p95_tau_ps": weighted_percentile(keys.astype(float), counts, 95.0),
            "p99_tau_ps": weighted_percentile(keys.astype(float), counts, 99.0),
            "max_tau_ps": float(keys.max()) if len(keys) else np.nan,
            "run_frac_ge_5ps": float(counts[keys >= 5].sum() / counts.sum()) if counts.sum() else np.nan,
            "run_frac_ge_10ps": float(counts[keys >= 10].sum() / counts.sum()) if counts.sum() else np.nan,
            "run_frac_ge_20ps": float(counts[keys >= 20].sum() / counts.sum()) if counts.sum() else np.nan,
            "contact_time_frac_in_runs_ge_20ps": float((keys[ge20] * counts[ge20]).sum() / total_tau) if total_tau > 0 else np.nan,
            "raw_lifetime_csv_gz": "",
        }
        df_with_combined = pd.concat([df_summary, pd.DataFrame([combined_summary])], ignore_index=True)
    else:
        df_with_combined = df_summary

    df_summary.to_csv(out_dir / "contact_lifetime_summary.csv", index=False)
    df_with_combined.to_csv(out_dir / "contact_lifetime_summary_with_combined.csv", index=False)
    return df_with_combined


def load_prev_sp_hist(prev_dir: Path) -> tuple[dict[int, list[tuple[int, int]]], list[dict]]:
    hist_by_tid: dict[int, list[tuple[int, int]]] = {}
    summaries: list[dict] = []
    if not prev_dir.exists():
        return hist_by_tid, summaries
    summary_path = prev_dir / "contact_lifetime_summary.csv"
    if summary_path.exists():
        try:
            summaries = pd.read_csv(summary_path).to_dict("records")
        except Exception:
            summaries = []
    for path in prev_dir.glob("Traj_*_P_tau.csv"):
        try:
            tid = int(path.name.split("_")[1])
            df = pd.read_csv(path)
            tau_col = "tau_span_ps" if "tau_span_ps" in df.columns else "tau_ps"
            hist_by_tid[tid] = [
                (int(round(tau)), int(count))
                for tau, count in zip(df[tau_col], df["count"])
            ]
        except Exception:
            continue
    return hist_by_tid, summaries


def default_targets(base: Path, already_done: set[int]) -> list[int]:
    rows = []
    for gro in (base / "runs").glob("Traj_*/md/production/production.gro"):
        tid = int(gro.parts[-4].split("_")[1])
        if tid in already_done:
            continue
        run_dir = base / "runs" / f"Traj_{tid}"
        if read_mdp_nst(run_dir) == 500 and production_finished(run_dir):
            rows.append((gro.stat().st_mtime, tid))
    return [tid for _mtime, tid in sorted(rows)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--targets", nargs="*", type=int, default=None)
    parser.add_argument("--already-done", nargs="*", type=int, default=[27655, 27659, 27650])
    parser.add_argument("--cutoff-nm", type=float, default=0.28)
    parser.add_argument("--thresholds-ps", nargs="*", type=float, default=[20.0])
    parser.add_argument("--max-cluster", type=int, default=101)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--label", type=str, default="remaining13")
    parser.add_argument("--combine-previous-latest3", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)
    sample_map = read_sample_manifest(results)
    suffix = threshold_suffix(args.thresholds_ps)
    out_sp_dir = results / f"contact_lifetime_cutoff0p28_1ps_{args.label}"
    out_cne = results / f"persistence_filter_cutoff0p28_1ps_{args.label}_core_{suffix}.csv"

    if args.targets is None:
        targets = default_targets(base, set(args.already_done))
    else:
        targets = args.targets

    valid_targets = []
    for tid in targets:
        run_dir = base / "runs" / f"Traj_{tid}"
        if read_mdp_nst(run_dir) == 500 and production_finished(run_dir):
            valid_targets.append(tid)
        else:
            print(f"skip Traj_{tid}: nst={read_mdp_nst(run_dir)}, finished={production_finished(run_dir)}", flush=True)

    print(f"Analyzing {len(valid_targets)} one-ps completed runs: {valid_targets}", flush=True)
    cne_rows: list[dict] = []
    sp_summaries: list[dict] = []
    hist_by_tid: dict[int, list[tuple[int, int]]] = {}
    errors = []

    workers = max(1, min(args.workers, os.cpu_count() or 1, len(valid_targets) or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                analyze_one,
                tid,
                base,
                args.cutoff_nm,
                args.thresholds_ps,
                args.max_cluster,
                sample_map,
                out_sp_dir,
            ): tid
            for tid in valid_targets
        }
        for future in as_completed(futures):
            tid = futures[future]
            try:
                result = future.result()
                cne_rows.extend(result["cne_rows"])
                sp_summaries.append(result["sp_summary"])
                hist_by_tid[tid] = result["hist_pairs"]
                print(f"COLLECTED Traj_{tid}", flush=True)
            except Exception as exc:
                errors.append({"Trajectory ID": tid, "error": repr(exc)})
                print(f"ERROR Traj_{tid}: {exc!r}", flush=True)

    cne_df = pd.DataFrame(cne_rows)
    if not cne_df.empty:
        cne_df = cne_df.sort_values(["Trajectory ID", "persistence_threshold_ps"])
        cne_df.to_csv(out_cne, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(results / f"persistence_filter_cutoff0p28_1ps_{args.label}_errors.csv", index=False)

    sp_df = combine_sp_from_hist(hist_by_tid, sp_summaries, out_sp_dir, f"{args.label}_combined", args.cutoff_nm)

    if args.combine_previous_latest3:
        prev_cne_path = results / "persistence_filter_cutoff0p28_1ps_latest3_with_10ps_20ps.csv"
        combined_label = "completed16"
        out_cne_all = results / f"persistence_filter_cutoff0p28_1ps_{combined_label}_core_{suffix}.csv"
        all_cne = cne_df.copy()
        if prev_cne_path.exists():
            prev = pd.read_csv(prev_cne_path)
            keep = ["instant_molecular_edges"] + [f"persist_span_ge_{threshold:g}ps" for threshold in args.thresholds_ps]
            prev = prev[prev["variant"].isin(keep)].copy()
            prev = prev.rename(
                columns={
                    "group": "sample_group",
                    "reference_CONDUCTIVITY_S_cm": "sigma_ref_S_cm",
                    "MALogE": "ma_loge_sigma",
                }
            )
            if "persistence_threshold_ps" not in prev.columns:
                prev["persistence_threshold_ps"] = prev["variant"].map(
                    lambda variant: 0.0
                    if variant == "instant_molecular_edges"
                    else float(str(variant).split("_ge_")[1].replace("ps", ""))
                )
            if "max_cluster" not in prev.columns:
                prev["max_cluster"] = args.max_cluster
            all_cne = pd.concat([prev, cne_df], ignore_index=True, sort=False)
        if not all_cne.empty:
            all_cne = all_cne.sort_values(["Trajectory ID", "persistence_threshold_ps"])
            all_cne.to_csv(out_cne_all, index=False)
            by_variant = (
                all_cne.groupby("variant", dropna=False)
                .agg(
                    n=("Trajectory ID", "count"),
                    mean_MALogE=("ma_loge_sigma", "mean"),
                    median_MALogE=("ma_loge_sigma", "median"),
                    mean_sigma=("sigma_cNE_S_cm", "mean"),
                )
                .reset_index()
            )
            by_variant.to_csv(results / f"persistence_filter_cutoff0p28_1ps_{combined_label}_core_{suffix}_summary_by_variant.csv", index=False)

        prev_hist, prev_summaries = load_prev_sp_hist(results / "contact_lifetime_cutoff0p28_1ps_latest3")
        hist_all = dict(prev_hist)
        hist_all.update(hist_by_tid)
        summaries_all = prev_summaries + sp_summaries
        combine_sp_from_hist(
            hist_all,
            summaries_all,
            results / f"contact_lifetime_cutoff0p28_1ps_{combined_label}",
            f"{combined_label}_combined",
            args.cutoff_nm,
        )

    print("\nWROTE:")
    print(f"  {out_cne}")
    print(f"  {out_sp_dir / 'contact_lifetime_summary_with_combined.csv'}")
    if args.combine_previous_latest3:
        print(f"  {out_cne_all}")
        print(f"  {results / 'contact_lifetime_cutoff0p28_1ps_completed16' / 'contact_lifetime_summary_with_combined.csv'}")
    if not cne_df.empty:
        print("\nCNE by variant:")
        print(
            cne_df.groupby("variant")
            .agg(n=("Trajectory ID", "count"), mean_MALogE=("ma_loge_sigma", "mean"), median_MALogE=("ma_loge_sigma", "median"))
            .reset_index()
            .to_string(index=False)
        )
    if not sp_df.empty:
        print("\nSP summary tail:")
        print(sp_df.tail(8).to_string(index=False))
    if errors:
        print("\nErrors:")
        print(pd.DataFrame(errors).to_string(index=False))


if __name__ == "__main__":
    main()
