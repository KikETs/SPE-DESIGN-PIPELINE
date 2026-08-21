from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import math
import warnings

import numpy as np
import pandas as pd


KB = 1.380649e-23
E_CHG = 1.602176634e-19
DEFAULT_TAU_PS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]


@dataclass(frozen=True)
class TrajMeta:
    traj_id: int
    group: str
    sigma_ref: float
    sigma_ne: float
    d_li_cm2s: float
    d_an_cm2s: float
    v_nm3: float
    n_li: int
    n_an: int
    gro: Path
    xtc: Path


def parse_tau_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def safe_log10_ratio(pred: float, ref: float) -> float:
    if pred > 0 and ref > 0 and np.isfinite(pred) and np.isfinite(ref):
        return float(abs(math.log10(pred / ref)))
    return float("nan")


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    m = xx.notna() & yy.notna()
    if int(m.sum()) < 3:
        return np.nan
    xx = xx[m]
    yy = yy[m]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return np.nan
    return float(xx.corr(yy, method=method))


def sigma_from_pop(
    pop: np.ndarray,
    d_li_cm2s: float,
    d_an_cm2s: float,
    volume_nm3: float,
    temperature_k: float,
    max_cluster: int,
) -> float:
    """cNE0 conductivity with formal cluster net charge z_ij = i - j.

    alpha_ij is the average number of clusters per frame, not a normalized
    fraction. Positively charged clusters use Li diffusivity; negatively charged
    clusters use anion diffusivity; neutral clusters contribute zero.
    """
    v_cm3 = volume_nm3 * 1e-21
    pref = (E_CHG**2) / (KB * temperature_k * v_cm3)
    sigma = 0.0
    imax = min(int(max_cluster), pop.shape[0])
    jmax = min(int(max_cluster), pop.shape[1])
    for i in range(imax):
        for j in range(jmax):
            nij = float(pop[i, j])
            if nij == 0.0 or i == j:
                continue
            q = i - j
            d_eff = d_li_cm2s if q > 0 else d_an_cm2s
            sigma += pref * (q * q) * nij * d_eff
    return float(sigma)


def add_pop_from_codes(pop: np.ndarray, codes: np.ndarray, n_li: int, n_an: int) -> int:
    """Add one frame's cluster counts from Li-anion contact codes.

    codes encode molecular contacts as li_index * n_an + anion_index.
    Returns the largest cluster size observed in this frame.
    """
    if codes.size == 0:
        pop[1, 0] += n_li
        pop[0, 1] += n_an
        return 1

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

    active_li = np.unique(li)
    active_an = np.unique(an)
    comp_li: dict[int, int] = {}
    comp_an: dict[int, int] = {}
    for li_i in active_li:
        root = find(int(li_i))
        comp_li[root] = comp_li.get(root, 0) + 1
    for an_i in active_an:
        root = find(n_li + int(an_i))
        comp_an[root] = comp_an.get(root, 0) + 1

    largest = 1
    for root in set(comp_li) | set(comp_an):
        i = comp_li.get(root, 0)
        j = comp_an.get(root, 0)
        largest = max(largest, i + j)
        if i < pop.shape[0] and j < pop.shape[1]:
            pop[i, j] += 1.0

    free_li = n_li - len(active_li)
    free_an = n_an - len(active_an)
    if free_li:
        pop[1, 0] += free_li
    if free_an:
        pop[0, 1] += free_an
    return int(largest)


def pop_descriptors(pop: np.ndarray, n_li: int, n_an: int) -> dict[str, float]:
    ii = np.arange(pop.shape[0], dtype=int)[:, None]
    jj = np.arange(pop.shape[1], dtype=int)[None, :]
    neutral = float(pop[ii == jj].sum())
    charged = float(pop[ii != jj].sum())
    nonzero = np.argwhere(pop > 0)
    largest = int((nonzero[:, 0] + nonzero[:, 1]).max()) if len(nonzero) else 0

    def val(i: int, j: int) -> float:
        if i < pop.shape[0] and j < pop.shape[1]:
            return float(pop[i, j])
        return 0.0

    free_cat = val(1, 0)
    free_an = val(0, 1)
    return {
        "neutral_pop": neutral,
        "charged_pop": charged,
        "free_cation_count": free_cat,
        "free_anion_count": free_an,
        "free_cation_fraction": free_cat / n_li if n_li else np.nan,
        "free_anion_fraction": free_an / n_an if n_an else np.nan,
        "largest_cluster_size": largest,
        "pop_1_0": val(1, 0),
        "pop_0_1": val(0, 1),
        "pop_1_1": val(1, 1),
        "pop_2_1": val(2, 1),
        "pop_1_2": val(1, 2),
        "pop_2_2": val(2, 2),
        "alpha_Li_sum": float((ii * pop).sum()),
        "alpha_AN_sum": float((jj * pop).sum()),
    }


def read_htpmd_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis summary: {path}")
    row = pd.read_csv(path).iloc[0]

    def get(name: str, default=np.nan) -> float:
        return float(pd.to_numeric(row.get(name, default), errors="coerce"))

    return {
        "sigma_ne": get("sigma_NE_htpmd_S_cm"),
        "d_li_cm2s": get("D_Li_cm2s"),
        "d_an_cm2s": get("D_an_cm2s"),
        "v_nm3": get("V_nm3"),
        "n_li": int(get("N_LI")),
        "n_an": int(get("N_AN")),
    }


def load_valid_metadata(base: Path) -> list[TrajMeta]:
    per_path = base / "results" / "per_traj_eval.csv"
    if not per_path.exists():
        raise FileNotFoundError(f"Missing required file: {per_path}")
    per = pd.read_csv(per_path)
    required = {
        "Trajectory ID",
        "sample_group",
        "status",
        "CONDUCTIVITY",
        "sigma_NE_htpmd_S_cm_pred",
    }
    missing = required - set(per.columns)
    if missing:
        raise KeyError(f"per_traj_eval.csv missing columns: {sorted(missing)}")

    metas: list[TrajMeta] = []
    for _, row in per.iterrows():
        if str(row.get("status")) != "ok":
            continue
        traj_id = int(row["Trajectory ID"])
        sigma_ref = float(pd.to_numeric(row["CONDUCTIVITY"], errors="coerce"))
        sigma_ne_per = float(pd.to_numeric(row["sigma_NE_htpmd_S_cm_pred"], errors="coerce"))
        if not (np.isfinite(sigma_ref) and sigma_ref > 0 and np.isfinite(sigma_ne_per) and sigma_ne_per > 0):
            continue
        run_dir = base / "runs" / f"Traj_{traj_id}"
        analysis_summary = run_dir / "analysis" / "conductivity_summary_htpmd_ref.csv"
        gro = run_dir / "md" / "production" / "production.gro"
        xtc = run_dir / "md" / "production" / "production.xtc"
        if not gro.exists() or not xtc.exists() or not analysis_summary.exists():
            warnings.warn(f"Skipping Traj_{traj_id}: missing gro/xtc/analysis summary")
            continue
        htp = read_htpmd_summary(analysis_summary)
        metas.append(
            TrajMeta(
                traj_id=traj_id,
                group=str(row["sample_group"]),
                sigma_ref=sigma_ref,
                sigma_ne=float(htp["sigma_ne"]) if np.isfinite(htp["sigma_ne"]) else sigma_ne_per,
                d_li_cm2s=float(htp["d_li_cm2s"]),
                d_an_cm2s=float(htp["d_an_cm2s"]),
                v_nm3=float(htp["v_nm3"]),
                n_li=int(htp["n_li"]),
                n_an=int(htp["n_an"]),
                gro=gro,
                xtc=xtc,
            )
        )
    if not metas:
        raise RuntimeError("No valid trajectories found for persistence sweep.")
    return metas


def compute_one_sweep(
    meta: TrajMeta,
    tau_ps_list: list[float],
    cutoff_nm: float,
    frame_dt_ps: float,
    max_cluster: int,
    temperature_k: float,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    cache_path = cache_dir / f"Traj_{meta.traj_id}_persistence_sweep.csv"
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        cached_taus = set(round(float(x), 8) for x in cached.get("tau_ps", []))
        if set(round(float(x), 8) for x in tau_ps_list).issubset(cached_taus):
            return cached

    import MDAnalysis as mda
    from MDAnalysis.lib.nsgrid import FastNS

    u = mda.Universe(str(meta.gro), str(meta.xtc))
    sel_li = u.select_atoms("resname LI")
    sel_o = u.select_atoms("resname TFSI and name O*")
    sel_n = u.select_atoms("resname TFSI and name N*")
    if len(sel_li) == 0 or len(sel_o) == 0 or len(sel_n) == 0:
        raise RuntimeError(
            f"Traj_{meta.traj_id}: empty selection LI={len(sel_li)} O={len(sel_o)} N={len(sel_n)}"
        )
    n_li = len(sel_li)
    n_an = len(sel_n)
    if n_li != meta.n_li or n_an != meta.n_an:
        warnings.warn(f"Traj_{meta.traj_id}: selection counts differ from summary: {n_li}/{n_an} vs {meta.n_li}/{meta.n_an}")

    traj_dt = float(u.trajectory.dt)
    if np.isfinite(traj_dt) and abs(traj_dt - frame_dt_ps) > 0.05:
        warnings.warn(f"Traj_{meta.traj_id}: trajectory dt={traj_dt:g} ps; tau_span uses frame_dt_ps={frame_dt_ps:g} ps")

    anion_resindex_to_i = {int(residx): i for i, residx in enumerate(list(sel_n.resindices))}
    o_to_an = np.array([anion_resindex_to_i[int(atom.resindex)] for atom in sel_o], dtype=np.int32)
    cutoff_a = float(cutoff_nm) * 10.0

    frame_codes: list[np.ndarray] = []
    active: dict[int, tuple[int, int]] = {}
    run_records: list[tuple[int, int, int, int, float]] = []

    for _iframe, ts in enumerate(u.trajectory):
        coords = np.vstack([sel_li.positions, sel_o.positions]).astype(np.float32)
        pairs = FastNS(cutoff_a, coords, np.asarray(ts.dimensions, dtype=np.float32), pbc=True).self_search().get_pairs()
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

        iframe = len(frame_codes)
        frame_codes.append(codes.astype(np.int32, copy=False))
        current = set(int(code) for code in codes)
        for code in list(active.keys()):
            if code not in current:
                first, last = active.pop(code)
                # tau_span = (n_consecutive_frames - 1) * dt
                run_records.append((code, first, last, last - first + 1, (last - first) * frame_dt_ps))
        for code in current:
            if code in active:
                first, _ = active[code]
                active[code] = (first, iframe)
            else:
                active[code] = (iframe, iframe)

    n_frames = len(frame_codes)
    if n_frames == 0:
        raise RuntimeError(f"Traj_{meta.traj_id}: no trajectory frames")
    for code, (first, last) in active.items():
        run_records.append((code, first, last, last - first + 1, (last - first) * frame_dt_ps))

    taus = np.array([x[4] for x in run_records], dtype=float)
    life = {
        "n_contact_runs": int(taus.size),
        "median_contact_lifetime_ps": float(np.median(taus)) if taus.size else np.nan,
        "mean_contact_lifetime_ps": float(np.mean(taus)) if taus.size else np.nan,
        "p90_contact_lifetime_ps": float(np.percentile(taus, 90)) if taus.size else np.nan,
        "p95_contact_lifetime_ps": float(np.percentile(taus, 95)) if taus.size else np.nan,
        "p99_contact_lifetime_ps": float(np.percentile(taus, 99)) if taus.size else np.nan,
        "S20": float(np.mean(taus >= 20.0)) if taus.size else np.nan,
        "frac_runs_ge20": float(np.mean(taus >= 20.0)) if taus.size else np.nan,
        "contact_time_fraction_ge20": float(np.sum(taus[taus >= 20.0]) / np.sum(taus)) if taus.size and np.sum(taus) > 0 else np.nan,
    }

    tau_ps_list = sorted(float(x) for x in tau_ps_list)
    starts = {tau: [[] for _ in range(n_frames)] for tau in tau_ps_list if tau > 0}
    ends = {tau: [[] for _ in range(n_frames + 1)] for tau in tau_ps_list if tau > 0}
    for code, first, last, _length, tau_span in run_records:
        for tau in starts:
            if tau_span >= tau:
                starts[tau][first].append(code)
                if last + 1 <= n_frames:
                    ends[tau][last + 1].append(code)

    pop_shape = (max(2, min(max_cluster, n_li + 1)), max(2, min(max_cluster, n_an + 1)))
    pops = {tau: np.zeros(pop_shape, dtype=np.float64) for tau in tau_ps_list}
    largest_by_tau = {tau: 0 for tau in tau_ps_list}
    active_bool = {tau: np.zeros(n_li * n_an, dtype=bool) for tau in tau_ps_list if tau > 0}

    for iframe, codes in enumerate(frame_codes):
        if 0.0 in pops:
            largest_by_tau[0.0] = max(largest_by_tau[0.0], add_pop_from_codes(pops[0.0], codes, n_li, n_an))
        for tau in active_bool:
            if ends[tau][iframe]:
                active_bool[tau][np.array(ends[tau][iframe], dtype=np.int32)] = False
            if starts[tau][iframe]:
                active_bool[tau][np.array(starts[tau][iframe], dtype=np.int32)] = True
            filtered = codes[active_bool[tau][codes]] if codes.size else codes
            largest_by_tau[tau] = max(largest_by_tau[tau], add_pop_from_codes(pops[tau], filtered, n_li, n_an))

    rows = []
    for tau, pop in pops.items():
        pop /= float(n_frames)
        sigma = sigma_from_pop(pop, meta.d_li_cm2s, meta.d_an_cm2s, meta.v_nm3, temperature_k, max_cluster)
        desc = pop_descriptors(pop, n_li, n_an)
        desc["largest_cluster_size"] = max(desc["largest_cluster_size"], largest_by_tau[tau])
        pred_ref = sigma / meta.sigma_ref if meta.sigma_ref > 0 else np.nan
        ne_ref = meta.sigma_ne / meta.sigma_ref if meta.sigma_ref > 0 else np.nan
        s_tau = sigma / meta.sigma_ne if meta.sigma_ne > 0 else np.nan
        s_target = meta.sigma_ref / meta.sigma_ne if meta.sigma_ne > 0 else np.nan
        rows.append(
            {
                "traj_id": meta.traj_id,
                "Trajectory ID": meta.traj_id,
                "group": meta.group,
                "sample_group": meta.group,
                "valid": True,
                "cutoff_nm": cutoff_nm,
                "frame_dt_ps": frame_dt_ps,
                "tau_ps": tau,
                "n_frames": n_frames,
                "sigma_cNE_tau": sigma,
                "sigma_NE": meta.sigma_ne,
                "sigma_ref": meta.sigma_ref,
                "pred_ref": pred_ref,
                "NE_ref": ne_ref,
                "s_tau": s_tau,
                "s_target": s_target,
                "abs_log_error": safe_log10_ratio(sigma, meta.sigma_ref),
                "D_Li_cm2s": meta.d_li_cm2s,
                "D_an_cm2s": meta.d_an_cm2s,
                "V_nm3": meta.v_nm3,
                "N_LI": n_li,
                "N_AN": n_an,
                **desc,
                **life,
            }
        )

    out = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def lifetime_stats_from_analysis(run_analysis_dir: Path) -> dict[str, float]:
    paths = sorted(run_analysis_dir.glob("contact_lifetime_summary_cutoff0p28_ge20ps.csv"))
    if not paths:
        return {
            "n_contact_runs": np.nan,
            "median_contact_lifetime_ps": np.nan,
            "mean_contact_lifetime_ps": np.nan,
            "p90_contact_lifetime_ps": np.nan,
            "p95_contact_lifetime_ps": np.nan,
            "p99_contact_lifetime_ps": np.nan,
            "S20": np.nan,
            "frac_runs_ge20": np.nan,
            "contact_time_fraction_ge20": np.nan,
        }
    row = pd.read_csv(paths[-1]).iloc[0]

    def get(*names: str, default=np.nan) -> float:
        for name in names:
            if name in row and pd.notna(row[name]):
                return float(pd.to_numeric(row[name], errors="coerce"))
        return float(default)

    s20 = get("contact_lifetime_run_frac_ge_threshold", "run_frac_ge_20ps")
    return {
        "n_contact_runs": get("contact_lifetime_n_runs", "n_contact_runs"),
        "median_contact_lifetime_ps": get("contact_lifetime_median_ps", "median_tau_ps"),
        "mean_contact_lifetime_ps": get("contact_lifetime_mean_ps", "mean_tau_ps"),
        "p90_contact_lifetime_ps": get("contact_lifetime_p90_ps", "p90_tau_ps"),
        "p95_contact_lifetime_ps": get("contact_lifetime_p95_ps", "p95_tau_ps"),
        "p99_contact_lifetime_ps": get("contact_lifetime_p99_ps", "p99_tau_ps"),
        "S20": s20,
        "frac_runs_ge20": s20,
        "contact_time_fraction_ge20": get("contact_lifetime_time_frac_ge_threshold", "contact_time_frac_in_runs_ge_20ps"),
    }


def row_from_pop(
    meta: TrajMeta,
    tau_ps: float,
    pop: np.ndarray,
    cutoff_nm: float,
    frame_dt_ps: float,
    max_cluster: int,
    temperature_k: float,
    lifetime_stats: dict[str, float],
    source: str,
) -> dict[str, float | str | bool | int]:
    sigma = sigma_from_pop(pop, meta.d_li_cm2s, meta.d_an_cm2s, meta.v_nm3, temperature_k, max_cluster)
    desc = pop_descriptors(pop, meta.n_li, meta.n_an)
    return {
        "traj_id": meta.traj_id,
        "Trajectory ID": meta.traj_id,
        "group": meta.group,
        "sample_group": meta.group,
        "valid": True,
        "missing_reason": "",
        "source": source,
        "cutoff_nm": cutoff_nm,
        "frame_dt_ps": frame_dt_ps,
        "tau_ps": float(tau_ps),
        "n_frames": np.nan,
        "sigma_cNE_tau": sigma,
        "sigma_NE": meta.sigma_ne,
        "sigma_ref": meta.sigma_ref,
        "pred_ref": sigma / meta.sigma_ref if meta.sigma_ref > 0 else np.nan,
        "NE_ref": meta.sigma_ne / meta.sigma_ref if meta.sigma_ref > 0 else np.nan,
        "s_tau": sigma / meta.sigma_ne if meta.sigma_ne > 0 else np.nan,
        "s_target": meta.sigma_ref / meta.sigma_ne if meta.sigma_ne > 0 else np.nan,
        "abs_log_error": safe_log10_ratio(sigma, meta.sigma_ref),
        "D_Li_cm2s": meta.d_li_cm2s,
        "D_an_cm2s": meta.d_an_cm2s,
        "V_nm3": meta.v_nm3,
        "N_LI": meta.n_li,
        "N_AN": meta.n_an,
        **desc,
        **lifetime_stats,
    }


def missing_tau_row(
    meta: TrajMeta,
    tau_ps: float,
    cutoff_nm: float,
    frame_dt_ps: float,
    lifetime_stats: dict[str, float],
    reason: str,
) -> dict[str, float | str | bool | int]:
    return {
        "traj_id": meta.traj_id,
        "Trajectory ID": meta.traj_id,
        "group": meta.group,
        "sample_group": meta.group,
        "valid": False,
        "missing_reason": reason,
        "source": "missing",
        "cutoff_nm": cutoff_nm,
        "frame_dt_ps": frame_dt_ps,
        "tau_ps": float(tau_ps),
        "n_frames": np.nan,
        "sigma_cNE_tau": np.nan,
        "sigma_NE": meta.sigma_ne,
        "sigma_ref": meta.sigma_ref,
        "pred_ref": np.nan,
        "NE_ref": meta.sigma_ne / meta.sigma_ref if meta.sigma_ref > 0 else np.nan,
        "s_tau": np.nan,
        "s_target": meta.sigma_ref / meta.sigma_ne if meta.sigma_ne > 0 else np.nan,
        "abs_log_error": np.nan,
        "D_Li_cm2s": meta.d_li_cm2s,
        "D_an_cm2s": meta.d_an_cm2s,
        "V_nm3": meta.v_nm3,
        "N_LI": meta.n_li,
        "N_AN": meta.n_an,
        "neutral_pop": np.nan,
        "charged_pop": np.nan,
        "free_cation_count": np.nan,
        "free_anion_count": np.nan,
        "free_cation_fraction": np.nan,
        "free_anion_fraction": np.nan,
        "largest_cluster_size": np.nan,
        "pop_1_0": np.nan,
        "pop_0_1": np.nan,
        "pop_1_1": np.nan,
        "pop_2_1": np.nan,
        "pop_1_2": np.nan,
        "pop_2_2": np.nan,
        "alpha_Li_sum": np.nan,
        "alpha_AN_sum": np.nan,
        **lifetime_stats,
    }


def compute_existing_available_sweep(
    metas: list[TrajMeta],
    tau_ps_list: list[float],
    cutoff_nm: float,
    frame_dt_ps: float,
    max_cluster: int,
    temperature_k: float,
    out_dir: Path,
) -> pd.DataFrame:
    """Fast path using already persisted pop matrices.

    This gives exact tau=0 and tau=20 ps for all current runs. Other tau values
    require trajectory rescan and are marked invalid unless exact cached rows are
    already available in this output directory.
    """
    rows = []
    exact_cache = {}
    for cache in (out_dir / "per_traj_cache").glob("Traj_*_persistence_sweep.csv"):
        try:
            df = pd.read_csv(cache)
            for _, r in df.iterrows():
                exact_cache[(int(r["traj_id"]), float(r["tau_ps"]))] = r.to_dict()
        except Exception:
            continue

    for meta in metas:
        analysis_dir = meta.gro.parents[2] / "analysis"
        lifetime = lifetime_stats_from_analysis(analysis_dir)
        pop0_path = analysis_dir / "pop_mat.npy"
        pop20_path = analysis_dir / "pop_mat_persist_cutoff0p28_ge20ps.npy"
        for tau in tau_ps_list:
            key = (meta.traj_id, float(tau))
            if key in exact_cache:
                rows.append(exact_cache[key])
                continue
            if np.isclose(tau, 0.0) and pop0_path.exists():
                rows.append(
                    row_from_pop(
                        meta,
                        tau,
                        np.load(pop0_path),
                        cutoff_nm,
                        frame_dt_ps,
                        max_cluster,
                        temperature_k,
                        lifetime,
                        "analysis/pop_mat.npy",
                    )
                )
            elif np.isclose(tau, 20.0) and pop20_path.exists():
                rows.append(
                    row_from_pop(
                        meta,
                        tau,
                        np.load(pop20_path),
                        cutoff_nm,
                        frame_dt_ps,
                        max_cluster,
                        temperature_k,
                        lifetime,
                        "analysis/pop_mat_persist_cutoff0p28_ge20ps.npy",
                    )
                )
            else:
                rows.append(
                    missing_tau_row(
                        meta,
                        tau,
                        cutoff_nm,
                        frame_dt_ps,
                        lifetime,
                        "not precomputed; run this script with --mode exact to rescan trajectories",
                    )
                )
    return pd.DataFrame(rows)


def summarize_group_tau(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, tau), sub in sweep[sweep["valid"]].groupby(["group", "tau_ps"], sort=True):
        rows.append(
            {
                "group": group,
                "tau_ps": tau,
                "n_valid": int(len(sub)),
                "mean_MAlogE": float(sub["abs_log_error"].mean(skipna=True)),
                "median_MAlogE": float(sub["abs_log_error"].median(skipna=True)),
                "median_pred_ref": float(sub["pred_ref"].median(skipna=True)),
                "mean_pred_ref": float(sub["pred_ref"].mean(skipna=True)),
                "median_s_tau": float(sub["s_tau"].median(skipna=True)),
                "median_s_target": float(sub["s_target"].median(skipna=True)),
                "median_NE_ref": float(sub["NE_ref"].median(skipna=True)),
                "median_neutral_pop": float(sub["neutral_pop"].median(skipna=True)),
                "median_charged_pop": float(sub["charged_pop"].median(skipna=True)),
                "median_free_cation_fraction": float(sub["free_cation_fraction"].median(skipna=True)),
                "median_free_anion_fraction": float(sub["free_anion_fraction"].median(skipna=True)),
                "median_largest_cluster_size": float(sub["largest_cluster_size"].median(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def tau_star_table(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = sweep[sweep["valid"] & np.isfinite(sweep["abs_log_error"])].copy()
    for traj_id, sub in valid.groupby("traj_id", sort=True):
        best = sub.sort_values(["abs_log_error", "tau_ps"]).iloc[0]
        row20 = sub[np.isclose(sub["tau_ps"], 20.0)]
        row0 = sub[np.isclose(sub["tau_ps"], 0.0)]
        r20 = row20.iloc[0] if len(row20) else best
        r0 = row0.iloc[0] if len(row0) else best
        rows.append(
            {
                "traj_id": int(traj_id),
                "system_id": f"Traj_{int(traj_id)}",
                "group": best["group"],
                "tau_star": float(best["tau_ps"]),
                "best_MAlogE": float(best["abs_log_error"]),
                "sigma_ref": float(best["sigma_ref"]),
                "sigma_NE": float(best["sigma_NE"]),
                "NE_ref": float(best["NE_ref"]),
                "s_target": float(best["s_target"]),
                "cNE0_NE": float(r0["s_tau"]),
                "cNE20_NE": float(r20["s_tau"]),
                "cNE20_ref": float(r20["pred_ref"]),
                "median_contact_lifetime_ps": float(best["median_contact_lifetime_ps"]),
                "p90_contact_lifetime_ps": float(best["p90_contact_lifetime_ps"]),
                "p95_contact_lifetime_ps": float(best["p95_contact_lifetime_ps"]),
                "S20": float(best["S20"]),
                "frac_runs_ge20": float(best["frac_runs_ge20"]),
                "contact_time_fraction_ge20": float(best["contact_time_fraction_ge20"]),
                "neutral_pop_20ps": float(r20["neutral_pop"]),
                "charged_pop_20ps": float(r20["charged_pop"]),
                "largest_cluster_size_20ps": float(r20["largest_cluster_size"]),
                "free_cation_fraction_20ps": float(r20["free_cation_fraction"]),
                "free_anion_fraction_20ps": float(r20["free_anion_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def descriptor_correlations(sweep: pd.DataFrame, tau_star: pd.DataFrame) -> pd.DataFrame:
    descriptors = [
        "median_contact_lifetime_ps",
        "p90_contact_lifetime_ps",
        "p95_contact_lifetime_ps",
        "S20",
        "frac_runs_ge20",
        "contact_time_fraction_ge20",
        "neutral_pop_20ps",
        "charged_pop_20ps",
        "cNE20_NE",
        "cNE0_NE",
        "cNE5_NE",
        "cNE10_NE",
        "largest_cluster_size_20ps",
        "free_cation_fraction_20ps",
        "free_anion_fraction_20ps",
    ]
    wide = tau_star.copy()
    for tau in [0.0, 5.0, 10.0, 20.0]:
        tmp = sweep[np.isclose(sweep["tau_ps"], tau)][["traj_id", "s_tau"]].rename(columns={"s_tau": f"cNE{int(tau)}_NE"})
        wide = wide.merge(tmp, on="traj_id", how="left", suffixes=("", "_dup"))
        dup = f"cNE{int(tau)}_NE_dup"
        if dup in wide:
            wide[f"cNE{int(tau)}_NE"] = wide[f"cNE{int(tau)}_NE"].fillna(wide[dup])
            wide = wide.drop(columns=[dup])

    rows = []
    for col in descriptors:
        if col not in wide.columns:
            continue
        x = pd.to_numeric(wide[col], errors="coerce")
        y = pd.to_numeric(wide["tau_star"], errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) < 3:
            continue
        pearson = safe_corr(x[m], y[m], method="pearson")
        spearman = safe_corr(x[m], y[m], method="spearman")
        rows.append({"descriptor": col, "n": int(m.sum()), "pearson_r": pearson, "spearman_r": spearman})
    return pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)


def try_import_sklearn():
    try:
        from sklearn.linear_model import HuberRegressor, Ridge
        from sklearn.metrics import r2_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        return {
            "available": True,
            "Ridge": Ridge,
            "HuberRegressor": HuberRegressor,
            "r2_score": r2_score,
            "StratifiedKFold": StratifiedKFold,
            "SimpleImputer": SimpleImputer,
            "make_pipeline": make_pipeline,
            "StandardScaler": StandardScaler,
        }
    except Exception:
        return {"available": False}


def build_hybrid_feature_table(sweep: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base_cols = [
        "traj_id",
        "group",
        "sigma_ref",
        "sigma_NE",
        "NE_ref",
        "median_contact_lifetime_ps",
        "p90_contact_lifetime_ps",
        "S20",
        "contact_time_fraction_ge20",
    ]
    base = sweep[np.isclose(sweep["tau_ps"], 20.0)][base_cols + [
        "neutral_pop",
        "charged_pop",
        "largest_cluster_size",
        "free_cation_fraction",
        "free_anion_fraction",
    ]].copy()
    base = base.rename(
        columns={
            "neutral_pop": "neutral_pop_20ps",
            "charged_pop": "charged_pop_20ps",
            "largest_cluster_size": "largest_cluster_size_20ps",
            "free_cation_fraction": "free_cation_fraction_20ps",
            "free_anion_fraction": "free_anion_fraction_20ps",
        }
    )
    wide = base.drop_duplicates("traj_id").copy()
    wide["log10_sigma_ref"] = np.log10(wide["sigma_ref"])
    wide["log10_sigma_NE"] = np.log10(wide["sigma_NE"])

    for tau in sorted(sweep["tau_ps"].unique()):
        sub = sweep[np.isclose(sweep["tau_ps"], tau)][["traj_id", "sigma_cNE_tau", "s_tau"]].copy()
        tau_label = f"{tau:g}".replace(".", "p")
        sub = sub.rename(columns={"sigma_cNE_tau": f"sigma_cNE_tau{tau_label}", "s_tau": f"cNE{tau_label}_NE"})
        wide = wide.merge(sub, on="traj_id", how="left")
        wide[f"log10_sigma_cNE_tau{tau_label}"] = np.log10(wide[f"sigma_cNE_tau{tau_label}"])

    features = [
        "log10_sigma_NE",
        "neutral_pop_20ps",
        "charged_pop_20ps",
        "p90_contact_lifetime_ps",
        "S20",
        "contact_time_fraction_ge20",
    ]
    for tau in sorted(sweep["tau_ps"].unique()):
        tau_label = f"{tau:g}".replace(".", "p")
        features.append(f"log10_sigma_cNE_tau{tau_label}")
        features.append(f"cNE{tau_label}_NE")
    features = [f for f in features if f in wide.columns]
    min_finite = max(3, int(math.ceil(0.8 * len(wide)))) if len(wide) else 3
    features = [f for f in features if pd.to_numeric(wide[f], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum() >= min_finite]
    return wide, features


def metrics_for_predictions(df: pd.DataFrame, pred_col: str, ref_col: str = "log10_sigma_ref") -> dict[str, float]:
    m = df[pred_col].notna() & df[ref_col].notna()
    if int(m.sum()) == 0:
        return {}
    err = df.loc[m, pred_col] - df.loc[m, ref_col]
    pred_ref = 10 ** err
    out = {
        "n": int(m.sum()),
        "MAlogE": float(np.abs(err).mean()),
        "median_abs_log_error": float(np.abs(err).median()),
        "median_pred_ref": float(np.median(pred_ref)),
        "spearman_r": safe_corr(df.loc[m, pred_col], df.loc[m, ref_col], method="spearman"),
        "pearson_r": safe_corr(df.loc[m, pred_col], df.loc[m, ref_col], method="pearson"),
    }
    if int(m.sum()) >= 3:
        ss_res = float(np.sum(err**2))
        yy = df.loc[m, ref_col]
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        out["R2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        out["R2"] = np.nan
    return out


def ridge_fit_predict_numpy(x_train, y_train, x_test, alpha: float = 1.0):
    mu = np.nanmean(x_train, axis=0)
    sigma = np.nanstd(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    xtr = np.where(np.isfinite(x_train), x_train, mu)
    xte = np.where(np.isfinite(x_test), x_test, mu)
    xtr = (xtr - mu) / sigma
    xte = (xte - mu) / sigma
    xtr_i = np.column_stack([np.ones(len(xtr)), xtr])
    xte_i = np.column_stack([np.ones(len(xte)), xte])
    reg = np.eye(xtr_i.shape[1]) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(xtr_i.T @ xtr_i + reg, xtr_i.T @ y_train)
    return xte_i @ coef, coef, mu, sigma


def cross_validated_hybrid(
    sweep: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wide, features = build_hybrid_feature_table(sweep)
    wide = wide.replace([np.inf, -np.inf], np.nan)
    valid = wide[wide["log10_sigma_ref"].notna()].copy().reset_index(drop=True)
    x = valid[features].to_numpy(float)
    y = valid["log10_sigma_ref"].to_numpy(float)
    groups = valid["group"].astype(str).to_numpy()
    traj_ids = valid["traj_id"].to_numpy(int)

    sklearn = try_import_sklearn()
    pred_rows = []
    coef_rows = []

    baseline_cols = {"NE": "log10_sigma_NE"}
    for tau in sorted(sweep["tau_ps"].unique()):
        tau_label = f"{tau:g}".replace(".", "p")
        col = f"log10_sigma_cNE_tau{tau_label}"
        if col in valid.columns and pd.to_numeric(valid[col], errors="coerce").notna().sum() >= 3:
            baseline_cols[f"cNE_tau{tau_label}"] = col

    def add_pred_rows(split_name: str, test_idx: np.ndarray, model_name: str, pred_log: np.ndarray) -> None:
        for idx, pred in zip(test_idx, pred_log):
            pred_rows.append(
                {
                    "validation_scheme": split_name,
                    "model": model_name,
                    "traj_id": int(traj_ids[idx]),
                    "group": groups[idx],
                    "log10_sigma_ref": float(y[idx]),
                    "log10_sigma_pred": float(pred),
                    "sigma_ref": float(10 ** y[idx]),
                    "sigma_pred": float(10 ** pred),
                    "pred_ref": float(10 ** (pred - y[idx])),
                    "abs_log_error": float(abs(pred - y[idx])),
                }
            )

    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(valid))
    for i in all_idx:
        train = all_idx[all_idx != i]
        test = np.array([i])
        splits.append(("LOOCV", train, test))

    unique_groups = sorted(set(groups))
    for g in unique_groups:
        test = np.where(groups == g)[0]
        train = np.where(groups != g)[0]
        if len(train) and len(test):
            splits.append((f"leave_one_group_out:{g}", train, test))

    if sklearn["available"] and len(valid) >= 9 and len(unique_groups) >= 2:
        n_splits = min(5, min(np.bincount(pd.Categorical(groups).codes)))
        if n_splits >= 2:
            skf = sklearn["StratifiedKFold"](n_splits=n_splits, shuffle=True, random_state=7)
            for fold, (train, test) in enumerate(skf.split(x, groups), start=1):
                splits.append((f"stratified_kfold:{fold}", train, test))

    for split_name, train_idx, test_idx in splits:
        train_df = valid.iloc[train_idx]
        test_df = valid.iloc[test_idx]

        for model, col in baseline_cols.items():
            add_pred_rows(split_name, test_idx, model, test_df[col].to_numpy(float))

        # Best single global tau is chosen on the training fold only.
        tau_perf = []
        for model, col in baseline_cols.items():
            if not model.startswith("cNE_tau"):
                continue
            e = np.abs(train_df[col].to_numpy(float) - train_df["log10_sigma_ref"].to_numpy(float))
            mean_err = float(np.nanmean(e)) if np.isfinite(e).any() else float("nan")
            if np.isfinite(mean_err):
                tau_perf.append((mean_err, model, col))
        if tau_perf:
            _err, best_name, best_col = sorted(tau_perf)[0]
            add_pred_rows(split_name, test_idx, f"best_global_tau_train_selected:{best_name}", test_df[best_col].to_numpy(float))

        if sklearn["available"]:
            pipe = sklearn["make_pipeline"](
                sklearn["SimpleImputer"](strategy="median"),
                sklearn["StandardScaler"](),
                sklearn["Ridge"](alpha=1.0),
            )
            pipe.fit(x[train_idx], y[train_idx])
            add_pred_rows(split_name, test_idx, "hybrid_ridge", pipe.predict(x[test_idx]))
            ridge = pipe.named_steps["ridge"]
            coef_rows.append({"validation_scheme": split_name, "model": "hybrid_ridge", "feature": "intercept", "coef": float(ridge.intercept_)})
            for feature, coef in zip(features, ridge.coef_):
                coef_rows.append({"validation_scheme": split_name, "model": "hybrid_ridge", "feature": feature, "coef": float(coef)})

            if len(train_idx) >= 10:
                try:
                    huber_pipe = sklearn["make_pipeline"](
                        sklearn["SimpleImputer"](strategy="median"),
                        sklearn["StandardScaler"](),
                        sklearn["HuberRegressor"](alpha=1e-3, max_iter=10000),
                    )
                    huber_pipe.fit(x[train_idx], y[train_idx])
                    add_pred_rows(split_name, test_idx, "hybrid_huber", huber_pipe.predict(x[test_idx]))
                except Exception as exc:
                    warnings.warn(f"Huber failed for {split_name}: {exc}")
        else:
            pred, coef, _mu, _sigma = ridge_fit_predict_numpy(x[train_idx], y[train_idx], x[test_idx], alpha=1.0)
            add_pred_rows(split_name, test_idx, "hybrid_ridge_numpy", pred)
            coef_rows.append({"validation_scheme": split_name, "model": "hybrid_ridge_numpy", "feature": "intercept", "coef": float(coef[0])})
            for feature, value in zip(features, coef[1:]):
                coef_rows.append({"validation_scheme": split_name, "model": "hybrid_ridge_numpy", "feature": feature, "coef": float(value)})

    pred_df = pd.DataFrame(pred_rows)
    summary_rows = []
    for (scheme, model), sub in pred_df.groupby(["validation_scheme", "model"], sort=True):
        d = metrics_for_predictions(sub.rename(columns={"log10_sigma_pred": "pred"}), "pred")
        if not d:
            continue
        d.update({"validation_scheme": scheme, "model": model})
        summary_rows.append(d)
        for group, gsub in sub.groupby("group"):
            gd = metrics_for_predictions(gsub.rename(columns={"log10_sigma_pred": "pred"}), "pred")
            if not gd:
                continue
            gd.update({"validation_scheme": scheme, "model": model, "group": group})
            summary_rows.append(gd)
    summary_df = pd.DataFrame(summary_rows)
    coef_df = pd.DataFrame(coef_rows)

    pred_df.to_csv(out_dir / "hybrid_model_cv_predictions.csv", index=False)
    summary_df.to_csv(out_dir / "hybrid_model_cv_summary.csv", index=False)
    coef_df.to_csv(out_dir / "hybrid_model_coefficients.csv", index=False)
    return pred_df, summary_df, coef_df


def plot_outputs(sweep: pd.DataFrame, tau_star: pd.DataFrame, corr: pd.DataFrame, hybrid_pred: pd.DataFrame, out_dir: Path, hybrid_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tau_order = sorted(sweep["tau_ps"].unique())
    groups = sorted(sweep["group"].dropna().unique())

    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 3.2 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        sub = sweep[sweep["group"] == group]
        for _tid, ts in sub.groupby("traj_id"):
            ts = ts.sort_values("tau_ps")
            ax.plot(ts["tau_ps"], ts["s_tau"], color="0.7", alpha=0.35, linewidth=0.8)
        med = sub.groupby("tau_ps")["s_tau"].median().reindex(tau_order)
        ax.plot(tau_order, med.values, marker="o", linewidth=2.4, color="black", label="group median")
        target = sub.groupby("traj_id")["s_target"].first().median()
        ax.axhline(target, color="tab:red", linestyle="--", linewidth=1.8, label="median s_target")
        ax.set_title(group)
        ax.set_ylabel("s_tau = cNE_tau / NE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("persistence threshold tau (ps)")
    fig.tight_layout()
    fig.savefig(plot_dir / "suppression_curves_by_group.png", dpi=220)
    plt.close(fig)

    summary = summarize_group_tau(sweep)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group in groups:
        sub = summary[summary["group"] == group].sort_values("tau_ps")
        ax.plot(sub["tau_ps"], sub["median_pred_ref"], marker="o", label=group)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("persistence threshold tau (ps)")
    ax.set_ylabel("median pred/ref")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "median_pred_ref_vs_tau_by_group.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group in groups:
        sub = summary[summary["group"] == group].sort_values("tau_ps")
        ax.plot(sub["tau_ps"], sub["median_MAlogE"], marker="o", label=group)
    ax.set_xlabel("persistence threshold tau (ps)")
    ax.set_ylabel("median absolute log error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "MAlogE_vs_tau_by_group.png", dpi=220)
    plt.close(fig)

    scatter_specs = [
        ("p90_contact_lifetime_ps", "tau_star_vs_p90_contact_lifetime.png"),
        ("S20", "tau_star_vs_S20.png"),
        ("contact_time_fraction_ge20", "tau_star_vs_contact_time_fraction_ge20.png"),
        ("cNE20_NE", "tau_star_vs_cNE20_NE.png"),
        ("neutral_pop_20ps", "tau_star_vs_neutral_pop_20ps.png"),
    ]
    for col, name in scatter_specs:
        if col not in tau_star.columns:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for group in sorted(tau_star["group"].unique()):
            sub = tau_star[tau_star["group"] == group]
            ax.scatter(sub[col], sub["tau_star"], label=group, alpha=0.75)
        ax.set_xlabel(col)
        ax.set_ylabel("tau_star diagnostic (ps)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=220)
        plt.close(fig)

    hybrid_plot_dir = hybrid_dir / "plots"
    hybrid_plot_dir.mkdir(parents=True, exist_ok=True)
    loocv = hybrid_pred[hybrid_pred["validation_scheme"] == "LOOCV"].copy()
    preferred = ["NE", "cNE_tau0", "cNE_tau5", "cNE_tau20", "hybrid_ridge", "hybrid_huber", "hybrid_ridge_numpy"]
    models = [m for m in preferred if m in set(loocv["model"])]
    fig, axes = plt.subplots(1, max(1, len(models)), figsize=(4.2 * max(1, len(models)), 4.0), squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = loocv[loocv["model"] == model]
        ax.scatter(sub["log10_sigma_ref"], sub["log10_sigma_pred"], c="tab:blue", alpha=0.7, s=18)
        lo = min(sub["log10_sigma_ref"].min(), sub["log10_sigma_pred"].min())
        hi = max(sub["log10_sigma_ref"].max(), sub["log10_sigma_pred"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("log10 reference")
        ax.set_ylabel("log10 prediction")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(hybrid_plot_dir / "predicted_vs_reference_log_conductivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_models = [m for m in ["NE", "cNE_tau0", "cNE_tau5", "cNE_tau20", "hybrid_ridge", "hybrid_huber", "hybrid_ridge_numpy"] if m in set(loocv["model"])]
    data = [loocv[loocv["model"] == m]["pred_ref"].to_numpy(float) for m in plot_models]
    ax.boxplot(data, tick_labels=plot_models, showfliers=False)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("pred/ref")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(hybrid_plot_dir / "pred_ref_distribution_by_model.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    model = "hybrid_ridge" if "hybrid_ridge" in set(loocv["model"]) else ("hybrid_ridge_numpy" if "hybrid_ridge_numpy" in set(loocv["model"]) else plot_models[-1])
    sub = loocv[loocv["model"] == model].copy()
    sub["residual_log10"] = sub["log10_sigma_pred"] - sub["log10_sigma_ref"]
    labels = sorted(sub["group"].unique())
    ax.boxplot([sub[sub["group"] == g]["residual_log10"].to_numpy(float) for g in labels], tick_labels=labels, showfliers=True)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("log10(pred/ref)")
    ax.set_title(model)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(hybrid_plot_dir / "hybrid_residual_by_group.png", dpi=220)
    plt.close(fig)


def write_collective_todo(base: Path) -> None:
    out_dir = base / "results" / "collective_einstein"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "TODO_collective_conductivity.md"
    md.write_text(
        """# Collective Einstein Conductivity TODO

This repository currently has wrapped GROMACS `production.xtc` trajectories and final `production.gro` files. A defensible collective Einstein conductivity calculation should not be faked from wrapped coordinates.

Required inputs:

- Unwrapped ion trajectories for Li and TFSI center/reference atoms.
- Formal ion charges: Li = +1, TFSI = -1.
- Time-dependent box volume or a clearly justified average volume.
- Temperature in K.
- Unit conversion from GROMACS coordinates/time to SI conductivity.
- Block averaging or bootstrap uncertainty estimate.

Formula target:

```text
sigma_collective = e^2 / (6 V k_B T) * d/dt < | sum_i z_i Delta r_i(t) |^2 >
```

Implementation notes:

- Use formal charges in the transport formula, not scaled MD partial charges.
- Remove center-of-mass drift consistently.
- Use unwrapped coordinates. Reconstructing unwrapped coordinates from XTC may be possible only if frame spacing is fine enough and molecules do not jump ambiguously across PBC.
- Report block-to-block variability; do not use a single noisy long-time slope without diagnostics.
""",
        encoding="utf-8",
    )


def make_report(
    out_dir: Path,
    hybrid_dir: Path,
    sweep: pd.DataFrame,
    group_tau: pd.DataFrame,
    tau_star: pd.DataFrame,
    hybrid_summary: pd.DataFrame,
) -> None:
    best_by_group = group_tau.loc[group_tau.groupby("group")["mean_MAlogE"].idxmin()].sort_values("group")
    loocv = hybrid_summary[hybrid_summary["validation_scheme"] == "LOOCV"].copy()
    compare_models = ["NE", "cNE_tau0", "cNE_tau5", "cNE_tau20", "hybrid_ridge", "hybrid_huber", "hybrid_ridge_numpy"]
    overall_mask = loocv["group"].isna() if "group" in loocv.columns else pd.Series(True, index=loocv.index)
    compare = loocv[overall_mask & loocv["model"].isin(compare_models) & loocv["MAlogE"].notna()].copy()
    best_global = loocv[loocv["model"].str.startswith("best_global_tau_train_selected")].copy()
    if "group" in best_global.columns:
        best_global = best_global[best_global["group"].isna()]
    best_global = best_global[best_global["MAlogE"].notna()]
    if len(best_global):
        bg = best_global.sort_values("MAlogE").head(1).copy()
        bg["model"] = "best_global_tau_train_selected"
        compare = pd.concat([compare, bg], ignore_index=True)
    compare = compare.sort_values("MAlogE") if len(compare) else compare
    coverage = (
        sweep.groupby("tau_ps")
        .agg(n_rows=("traj_id", "count"), n_valid=("valid", "sum"))
        .reset_index()
        .sort_values("tau_ps")
    )

    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_No data available._"
        use = df[cols].copy()
        for c in use.columns:
            if pd.api.types.is_float_dtype(use[c]):
                use[c] = use[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        return use.to_markdown(index=False)

    report = f"""# Persistence Sweep and Hybrid Conductivity Analysis

## Main conclusion

The 20 ps persistence threshold is not universal. It is useful as a low-conductivity diagnostic, but it cannot simultaneously explain bottom, middle, and top conductivity regimes.

Current behavior:

- bottom: long-lived Li-TFSI association dominates. 20 ps persistence gives near-correct suppression.
- middle: 20 ps overcorrects and underpredicts conductivity.
- top: long-lived clusters are mostly absent. 20 ps persistence leaves cNE close to NE and overpredicts conductivity.

The 20 ps threshold should be interpreted as an operational persistence criterion supported by the contact-lifetime distribution, not as a universal ion-pair lifetime.

## Best tau by group

Diagnostic only. These values use reference conductivity and must not be used as production thresholds.

{md_table(best_by_group, ["group", "tau_ps", "n_valid", "median_MAlogE", "median_pred_ref", "median_s_tau", "median_s_target", "median_NE_ref"])}

## Tau coverage

Rows with `n_valid = 0` require exact trajectory rescan. Run:

```bash
python scripts/analysis/persistence_sweep_and_hybrid.py --mode exact --workers <N>
```

{md_table(coverage, ["tau_ps", "n_rows", "n_valid"])}

## Model comparison

LOOCV comparison. `best_global_tau_train_selected` selects one tau using the training fold only.

{md_table(compare, ["model", "n", "MAlogE", "median_abs_log_error", "median_pred_ref", "spearman_r", "pearson_r", "R2"])}

## Key plots

- [Suppression curves by group](plots/suppression_curves_by_group.png)
- [Median pred/ref vs tau](plots/median_pred_ref_vs_tau_by_group.png)
- [MAlogE vs tau](plots/MAlogE_vs_tau_by_group.png)
- [tau_star vs p90 contact lifetime](plots/tau_star_vs_p90_contact_lifetime.png)
- [tau_star vs S20](plots/tau_star_vs_S20.png)
- [Hybrid predicted vs reference](../hybrid_correction/plots/predicted_vs_reference_log_conductivity.png)
- [Hybrid residual by group](../hybrid_correction/plots/hybrid_residual_by_group.png)

## Recommended manuscript strategy

- NE is retained as an upper-biased baseline because it ignores correlated ion motion.
- Static cNE is retained as a structurally motivated lower/correction baseline, but it can overcorrect.
- 20 ps persistence-cNE is retained as a physically interpretable low-conductivity diagnostic.
- A single distance- or persistence-based cNE definition is insufficient across conductivity regimes.
- For overall screening-level validation, use a cross-validated hybrid correction or a future collective Einstein estimator.

Suggested wording:

> The 20 ps threshold should be interpreted as an operational persistence criterion supported by the contact-lifetime distribution, not as a universal ion-pair lifetime.

> A single distance- or persistence-based cNE definition is insufficient across conductivity regimes.

> Persistence-filtered cNE is retained as a physically interpretable diagnostic, while cross-validated hybrid correction is used for screening-level validation.

## Output files

- `persistence_sweep_per_system.csv`
- `persistence_sweep_summary_by_group.csv`
- `group_tau_summary.csv`
- `tau_star_per_system.csv`
- `tau_star_descriptor_correlations.csv`
- `../hybrid_correction/hybrid_model_cv_predictions.csv`
- `../hybrid_correction/hybrid_model_cv_summary.csv`
- `../hybrid_correction/hybrid_model_coefficients.csv`
"""
    (out_dir / "README_analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--taus", default=",".join(f"{x:g}" for x in DEFAULT_TAU_PS))
    parser.add_argument("--cutoff-nm", type=float, default=0.28)
    parser.add_argument("--frame-dt-ps", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--max-cluster", type=int, default=101)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["existing", "exact"],
        default="existing",
        help="existing uses precomputed tau=0/tau=20 pop matrices; exact rescans trajectories for every tau.",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    tau_ps_list = parse_tau_list(args.taus)
    out_dir = base / "results" / "persistence_sweep_cutoff0p28_1ps"
    hybrid_dir = base / "results" / "hybrid_correction"
    cache_dir = out_dir / "per_traj_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metas = load_valid_metadata(base)
    if args.limit:
        metas = metas[: args.limit]
    print(f"[persistence-sweep] valid trajectories: {len(metas)}; mode={args.mode}", flush=True)

    errors: list[dict[str, str]] = []
    if args.mode == "existing":
        sweep = compute_existing_available_sweep(
            metas,
            tau_ps_list,
            args.cutoff_nm,
            args.frame_dt_ps,
            args.max_cluster,
            args.temperature_k,
            out_dir,
        )
    else:
        frames: list[pd.DataFrame] = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {
                ex.submit(
                    compute_one_sweep,
                    meta,
                    tau_ps_list,
                    args.cutoff_nm,
                    args.frame_dt_ps,
                    args.max_cluster,
                    args.temperature_k,
                    cache_dir,
                    args.force,
                ): meta
                for meta in metas
            }
            for i, fut in enumerate(as_completed(futs), start=1):
                meta = futs[fut]
                try:
                    df = fut.result()
                    frames.append(df)
                    print(f"[persistence-sweep] {i}/{len(futs)} Traj_{meta.traj_id} ok", flush=True)
                except Exception as exc:
                    errors.append({"traj_id": meta.traj_id, "group": meta.group, "error": repr(exc)})
                    print(f"[persistence-sweep] {i}/{len(futs)} Traj_{meta.traj_id} failed: {exc}", flush=True)
        if not frames:
            raise RuntimeError("No persistence sweep results were generated.")
        sweep = pd.concat(frames, ignore_index=True)
    sweep = sweep.sort_values(["group", "traj_id", "tau_ps"])
    sweep.to_csv(out_dir / "persistence_sweep_per_system.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "persistence_sweep_errors.csv", index=False)

    group_tau = summarize_group_tau(sweep)
    group_tau.to_csv(out_dir / "group_tau_summary.csv", index=False)
    group_tau.to_csv(out_dir / "persistence_sweep_summary_by_group.csv", index=False)

    tau_star = tau_star_table(sweep)
    tau_star.to_csv(out_dir / "tau_star_per_system.csv", index=False)

    corr = descriptor_correlations(sweep, tau_star)
    corr.to_csv(out_dir / "tau_star_descriptor_correlations.csv", index=False)

    hybrid_pred, hybrid_summary, _coef = cross_validated_hybrid(sweep, hybrid_dir)
    plot_outputs(sweep, tau_star, corr, hybrid_pred, out_dir, hybrid_dir)
    write_collective_todo(base)
    make_report(out_dir, hybrid_dir, sweep, group_tau, tau_star, hybrid_summary)

    print(f"[done] {out_dir / 'persistence_sweep_per_system.csv'}", flush=True)
    print(f"[done] {hybrid_dir / 'hybrid_model_cv_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
