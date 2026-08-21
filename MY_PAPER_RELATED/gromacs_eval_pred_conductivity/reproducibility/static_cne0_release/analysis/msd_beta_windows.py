#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_xvg(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["time_ps", "msd_nm2"])
    return df[(df["time_ps"] > 0.0) & (df["msd_nm2"] > 0.0)].reset_index(drop=True)


def parse_record(path: Path, runs_dir: Path, dataset: str) -> dict:
    rel = path.relative_to(runs_dir)
    traj_id = rel.parts[0].replace("Traj_", "")
    species = "Li" if path.name == "msd_li.xvg" else "TFSI_N"
    if dataset == "pred_replica":
        replica = next(part for part in rel.parts if part.startswith("replica_"))
        replica_index = int(replica.split("_", 1)[1])
    else:
        replica = "ref"
        replica_index = 1
    file_prefix = "pred_runs" if dataset == "pred_replica" else "ref_runs"
    return {
        "dataset": dataset,
        "trajectory_id": int(traj_id) if traj_id.isdigit() else traj_id,
        "replica": replica,
        "replica_index": replica_index,
        "species": species,
        "msd_file": f"{file_prefix}/{rel.as_posix()}",
    }


def local_beta(
    df: pd.DataFrame,
    *,
    n_grid: int,
    window_points: int,
    min_time_ps: float,
    max_time_ps: float | None,
) -> pd.DataFrame:
    work = df[(df["time_ps"] >= min_time_ps)].copy()
    if max_time_ps is not None:
        work = work[work["time_ps"] <= max_time_ps]
    if len(work) < window_points:
        return pd.DataFrame(columns=["time_ps", "msd_nm2", "beta"])

    log_t = np.log(work["time_ps"].to_numpy(dtype=float))
    log_msd = np.log(work["msd_nm2"].to_numpy(dtype=float))
    order = np.argsort(log_t)
    log_t = log_t[order]
    log_msd = log_msd[order]
    uniq_t, uniq_idx = np.unique(log_t, return_index=True)
    uniq_msd = log_msd[uniq_idx]

    grid_log_t = np.linspace(float(uniq_t[0]), float(uniq_t[-1]), int(n_grid))
    grid_log_msd = np.interp(grid_log_t, uniq_t, uniq_msd)
    n = len(grid_log_t)
    half = max(1, int(window_points) // 2)

    beta = np.full(n, np.nan, dtype=float)
    r2 = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 5:
            continue
        x = grid_log_t[lo:hi]
        y = grid_log_msd[lo:hi]
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        beta[i] = float(slope)
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return pd.DataFrame({
        "time_ps": np.exp(grid_log_t),
        "msd_nm2": np.exp(grid_log_msd),
        "beta": beta,
        "fit_r2": r2,
    })


def contiguous_windows(curve: pd.DataFrame, beta_low: float, beta_high: float, min_duration_ps: float) -> list[dict]:
    ok = curve["beta"].between(beta_low, beta_high) & np.isfinite(curve["beta"])
    windows = []
    start_idx = None
    for i, is_ok in enumerate(ok.to_numpy()):
        if is_ok and start_idx is None:
            start_idx = i
        if (not is_ok or i == len(ok) - 1) and start_idx is not None:
            end_idx = i if is_ok and i == len(ok) - 1 else i - 1
            part = curve.iloc[start_idx:end_idx + 1]
            start_ps = float(part["time_ps"].iloc[0])
            end_ps = float(part["time_ps"].iloc[-1])
            duration_ps = end_ps - start_ps
            if duration_ps >= min_duration_ps:
                windows.append({
                    "window_start_ps": start_ps,
                    "window_end_ps": end_ps,
                    "window_duration_ps": duration_ps,
                    "window_start_ns": start_ps / 1000.0,
                    "window_end_ns": end_ps / 1000.0,
                    "window_duration_ns": duration_ps / 1000.0,
                    "beta_mean": float(part["beta"].mean()),
                    "beta_median": float(part["beta"].median()),
                    "beta_min": float(part["beta"].min()),
                    "beta_max": float(part["beta"].max()),
                    "fit_r2_mean": float(part["fit_r2"].mean()),
                    "n_beta_points": int(len(part)),
                })
            start_idx = None
    windows.sort(key=lambda row: (row["window_duration_ps"], row["fit_r2_mean"]), reverse=True)
    return windows


def collect_pred_msd_records(pred_runs_dir: Path) -> list[tuple[Path, dict]]:
    records = []
    for li_path in sorted(pred_runs_dir.glob("Traj_*/analysis/replica_*/msd_li.xvg")):
        for path in [li_path, li_path.parent / "msd_tfsi_N.xvg"]:
            if path.exists():
                records.append((path, parse_record(path, pred_runs_dir, "pred_replica")))
    return records


def collect_ref_msd_records(ref_runs_dir: Path) -> list[tuple[Path, dict]]:
    records = []
    for li_path in sorted(ref_runs_dir.glob("Traj_*/analysis/msd_li.xvg")):
        for path in [li_path, li_path.parent / "msd_tfsi_N.xvg"]:
            if path.exists():
                records.append((path, parse_record(path, ref_runs_dir, "ref_traj")))
    return records


def find_default_ref_runs_dir(base_dir: Path) -> Path | None:
    candidates = [
        base_dir.parent.parent.parent / "gromacs" / "eval_top10_bottom10_stratified100" / "runs",
        base_dir.parent.parent / "gromacs" / "eval_top10_bottom10_stratified100" / "runs",
        Path("gromacs") / "eval_top10_bottom10_stratified100" / "runs",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def validate_record_counts(
    records: list[tuple[Path, dict]],
    *,
    dataset: str,
    expected_per_species: int,
    allow_mismatch: bool,
) -> None:
    counts = {
        species: sum(1 for _, meta in records if meta["species"] == species)
        for species in ["Li", "TFSI_N"]
    }
    if allow_mismatch or expected_per_species <= 0:
        return
    bad = {species: count for species, count in counts.items() if count != expected_per_species}
    if bad:
        detail = ", ".join(f"{species}={count}" for species, count in bad.items())
        raise SystemExit(
            f"{dataset} MSD count mismatch: expected {expected_per_species} per species, got {detail}"
        )


def load_group_labels(base_dir: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for csv_path in [
        base_dir / "github_results" / "sample_manifest_summary.csv",
        base_dir / "results" / "sample_manifest.csv",
    ]:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "Trajectory ID" not in df.columns or "sample_group" not in df.columns:
            continue
        for _, row in df.iterrows():
            labels[int(row["Trajectory ID"])] = str(row["sample_group"])
    return labels


def plot_results(curves: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ["ref_traj", "pred_replica"]:
        for species in ["Li", "TFSI_N"]:
            sub = curves[(curves["dataset"] == dataset) & (curves["species"] == species)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
            hue = "sample_group" if dataset == "pred_replica" and "sample_group" in sub.columns else None
            sns.lineplot(
                data=sub,
                x="time_ns",
                y="beta",
                hue=hue,
                units="series_id",
                estimator=None,
                linewidth=0.7,
                alpha=0.35,
                ax=ax,
            )
            ax.axhspan(0.9, 1.1, color="#2ca25f", alpha=0.14, label="0.9 <= beta <= 1.1")
            ax.axhline(1.0, color="#1b7837", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_ylim(-0.25, 2.25)
            ax.set_xlabel("lag time (ns)")
            ax.set_ylabel("local slope beta = d log(MSD) / d log(t)")
            ax.set_title(f"{dataset}: {species} local MSD exponent")
            if hue is None:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:1], labels[:1], loc="best")
            else:
                ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"beta_curves_{dataset}_{species}.png")
            plt.close(fig)

    top = windows.sort_values(["dataset", "species", "window_duration_ns"], ascending=[True, True, False])
    for dataset in ["ref_traj", "pred_replica"]:
        for species in ["Li", "TFSI_N"]:
            sub = top[(top["dataset"] == dataset) & (top["species"] == species)].head(20)
            if sub.empty:
                continue
            label = sub["series_label"].astype(str)
            fig, ax = plt.subplots(figsize=(10, max(4.5, 0.32 * len(sub))), dpi=180)
            y = np.arange(len(sub))
            ax.barh(y, sub["window_duration_ns"], color="#3182bd")
            ax.set_yticks(y)
            ax.set_yticklabels(label)
            ax.invert_yaxis()
            ax.set_xlabel("longest beta≈1 window duration (ns)")
            ax.set_title(f"{dataset}: {species} beta≈1 window")
            fig.tight_layout()
            fig.savefig(fig_dir / f"beta_window_lengths_{dataset}_{species}.png")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract local MSD log-log beta and beta≈1 windows.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--pred-runs-dir", type=Path, default=None)
    parser.add_argument("--ref-runs-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-grid", type=int, default=500)
    parser.add_argument("--window-points", type=int, default=31)
    parser.add_argument("--min-time-ps", type=float, default=10.0)
    parser.add_argument("--max-time-ps", type=float, default=None)
    parser.add_argument("--beta-low", type=float, default=0.9)
    parser.add_argument("--beta-high", type=float, default=1.1)
    parser.add_argument("--min-duration-ps", type=float, default=500.0)
    parser.add_argument("--expected-pred-replicas", type=int, default=180)
    parser.add_argument("--expected-ref-trajs", type=int, default=108)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    pred_runs_dir = (args.pred_runs_dir or base_dir / "runs").resolve()
    ref_runs_dir = args.ref_runs_dir.resolve() if args.ref_runs_dir else find_default_ref_runs_dir(base_dir)
    if ref_runs_dir is None:
        raise SystemExit(
            "Reference runs directory was not found. Pass --ref-runs-dir pointing to "
            "eval_top10_bottom10_stratified100/runs."
        )
    out_dir = (args.out_dir or base_dir / "github_results" / "msd_beta").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_group_labels = load_group_labels(base_dir)
    ref_group_labels = load_group_labels(ref_runs_dir.parent)
    curve_frames = []
    window_rows = []
    all_window_rows = []
    pred_records = collect_pred_msd_records(pred_runs_dir)
    ref_records = collect_ref_msd_records(ref_runs_dir)
    validate_record_counts(
        pred_records,
        dataset="pred_replica",
        expected_per_species=args.expected_pred_replicas,
        allow_mismatch=args.allow_count_mismatch,
    )
    validate_record_counts(
        ref_records,
        dataset="ref_traj",
        expected_per_species=args.expected_ref_trajs,
        allow_mismatch=args.allow_count_mismatch,
    )
    records = ref_records + pred_records

    for path, meta in records:
        df = read_xvg(path)
        curve = local_beta(
            df,
            n_grid=args.n_grid,
            window_points=args.window_points,
            min_time_ps=args.min_time_ps,
            max_time_ps=args.max_time_ps,
        )
        traj_id = int(meta["trajectory_id"]) if str(meta["trajectory_id"]).isdigit() else meta["trajectory_id"]
        if meta["dataset"] == "pred_replica":
            sample_group = pred_group_labels.get(traj_id, "pred")
        else:
            sample_group = ref_group_labels.get(traj_id, "ref")
        series_id = f"{meta['dataset']}_Traj_{traj_id}_{meta['replica']}_{meta['species']}"
        series_label = f"Traj_{traj_id}:{meta['replica']}:{meta['species']}"
        if not curve.empty:
            curve.insert(0, "series_id", series_id)
            curve.insert(1, "dataset", meta["dataset"])
            curve.insert(2, "trajectory_id", traj_id)
            curve.insert(3, "sample_group", sample_group)
            curve.insert(4, "replica", meta["replica"])
            curve.insert(5, "replica_index", meta["replica_index"])
            curve.insert(6, "species", meta["species"])
            curve["time_ns"] = curve["time_ps"] / 1000.0
            curve_frames.append(curve)

        windows = contiguous_windows(curve, args.beta_low, args.beta_high, args.min_duration_ps)
        for window_rank, window in enumerate(windows, start=1):
            all_window_rows.append({
                **meta,
                "sample_group": sample_group,
                "series_id": series_id,
                "series_label": series_label,
                "window_rank": window_rank,
                **window,
            })
        row = {
            **meta,
            "sample_group": sample_group,
            "series_id": series_id,
            "series_label": series_label,
            "n_msd_points": int(len(df)),
            "time_min_ps": float(df["time_ps"].min()) if not df.empty else math.nan,
            "time_max_ps": float(df["time_ps"].max()) if not df.empty else math.nan,
            "beta_low": args.beta_low,
            "beta_high": args.beta_high,
            "min_duration_ps": args.min_duration_ps,
            "n_windows": len(windows),
        }
        if windows:
            row.update(windows[0])
        else:
            row.update({
                "window_start_ps": math.nan,
                "window_end_ps": math.nan,
                "window_duration_ps": 0.0,
                "window_start_ns": math.nan,
                "window_end_ns": math.nan,
                "window_duration_ns": 0.0,
                "beta_mean": math.nan,
                "beta_median": math.nan,
                "beta_min": math.nan,
                "beta_max": math.nan,
                "fit_r2_mean": math.nan,
                "n_beta_points": 0,
            })
        window_rows.append(row)

    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    windows_df = pd.DataFrame(window_rows)
    all_windows_df = pd.DataFrame(all_window_rows)
    curves.to_csv(out_dir / "msd_beta_curves.csv", index=False)
    windows_df.to_csv(out_dir / "msd_beta_windows.csv", index=False)
    all_windows_df.to_csv(out_dir / "msd_beta_all_windows.csv", index=False)

    summary = (
        windows_df.groupby(["dataset", "species"], dropna=False)
        .agg(
            n_series=("series_id", "count"),
            n_with_beta_window=("n_windows", lambda x: int((x > 0).sum())),
            median_window_duration_ns=("window_duration_ns", "median"),
            mean_window_duration_ns=("window_duration_ns", "mean"),
            median_beta_mean=("beta_mean", "median"),
            mean_beta_mean=("beta_mean", "mean"),
            median_window_start_ns=("window_start_ns", "median"),
            median_window_end_ns=("window_end_ns", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "msd_beta_summary.csv", index=False)

    plot_results(curves, windows_df, out_dir)

    readme = [
        "# MSD log-log beta analysis",
        "",
        "Local anomalous-diffusion exponent beta was estimated by rolling linear fits of log(MSD) vs log(t).",
        "",
        "Settings:",
        f"- log-spaced grid points: {args.n_grid}",
        f"- rolling fit window points: {args.window_points}",
        f"- minimum lag time: {args.min_time_ps:g} ps",
        f"- beta≈1 threshold: {args.beta_low:g} <= beta <= {args.beta_high:g}",
        f"- minimum accepted beta≈1 window length: {args.min_duration_ps:g} ps",
        "",
        "Outputs:",
        "- `msd_beta_windows.csv`: longest beta≈1 interval for each trajectory/replica/species.",
        "- `msd_beta_all_windows.csv`: every accepted beta≈1 interval for each trajectory/replica/species.",
        "- `msd_beta_curves.csv`: local beta curve for every analyzed series.",
        "- `msd_beta_summary.csv`: dataset/species-level summary.",
        "- `figures/`: beta curves and longest-window bar plots.",
        "",
        "Summary:",
        "",
        summary.to_markdown(index=False, floatfmt=".6g"),
        "",
        "Input counts:",
        f"- reference trajectories: {sum(1 for _, meta in ref_records if meta['species'] == 'Li')} per species",
        f"- predicted production replicas: {sum(1 for _, meta in pred_records if meta['species'] == 'Li')} per species",
        "- predicted single-run `Traj_*/analysis/msd_*.xvg` files are intentionally excluded.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme))

    print(f"Reference runs: {ref_runs_dir}")
    print(f"Predicted runs: {pred_runs_dir}")
    print(f"Input MSD files: {len(records)}")
    print(f"Series with beta curves: {len(windows_df)}")
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
