#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


_FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_COND_PATTERNS = [
    re.compile(rf"static conductivity[^0-9+-]*{_FLOAT_RE}", re.IGNORECASE),
    re.compile(rf"\bconductivity\b[^0-9+-]*{_FLOAT_RE}", re.IGNORECASE),
]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _parse_conductivity_from_log(path: Path) -> float:
    txt = _read_text(path)
    if not txt:
        return float("nan")
    for pat in _COND_PATTERNS:
        matches = pat.findall(txt)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                pass
    return float("nan")


def _load_ref_table(root: Path) -> pd.DataFrame:
    candidates = [
        root / "results" / "per_traj_eval.csv",
        root / "results" / "run_results.csv",
        root.parent / "simulation-trajectory-aggregate.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Trajectory ID" in df.columns:
            return df
    return pd.DataFrame(columns=["Trajectory ID"])


def _sigma_to_s_cm(value: float, unit: str) -> float:
    if not np.isfinite(value):
        return float("nan")
    unit = unit.strip().lower()
    if unit in {"s_per_cm", "s/cm"}:
        return float(value)
    if unit in {"s_per_m", "s/m"}:
        return float(value) / 100.0
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect per-trajectory GK analysis results into results/*.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--sigma-unit", default="s_per_m", choices=["s_per_m", "s_per_cm"], help="Unit assumed for conductivity parsed from gmx current logs")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    runs_dir = root / "runs"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ref_df = _load_ref_table(root)
    if not ref_df.empty:
        ref_df = ref_df.copy()
        ref_df["Trajectory ID"] = pd.to_numeric(ref_df["Trajectory ID"], errors="coerce").astype("Int64")

    rows: list[dict] = []
    for summary_path in sorted(runs_dir.glob("Traj_*/analysis_gk/gk_summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        run_dir = summary_path.parents[1]
        try:
            traj_id = int(run_dir.name.split("_", 1)[1])
        except Exception:
            continue

        inputs = summary.get("inputs", {})
        for mode_entry in summary.get("modes", []):
            log_path = Path(mode_entry.get("log_path", "")) if mode_entry.get("log_path") else None
            sigma_raw = _parse_conductivity_from_log(log_path) if log_path and log_path.exists() else float("nan")
            sigma_s_cm = _sigma_to_s_cm(sigma_raw, args.sigma_unit)
            rows.append(
                {
                    "Trajectory ID": traj_id,
                    "mode": mode_entry.get("mode"),
                    "gk_status": "ok" if int(mode_entry.get("returncode", 1)) == 0 else "failed",
                    "gk_returncode": int(mode_entry.get("returncode", -1)),
                    "gk_group": inputs.get("group"),
                    "gk_begin_ns": float(inputs.get("begin_ps", 0.0)) / 1000.0,
                    "gk_end_ns": None if inputs.get("end_ps") is None else float(inputs.get("end_ps")) / 1000.0,
                    "gk_sample_dt_ps": inputs.get("sample_dt_ps"),
                    "gk_temperature_k": inputs.get("temperature_k"),
                    "gk_sigma_raw": sigma_raw,
                    "gk_sigma_s_cm": sigma_s_cm,
                    "gk_summary_json": str(summary_path),
                    "gk_log_path": str(log_path) if log_path else "",
                    "gk_current_xvg": mode_entry.get("outputs", {}).get("current_xvg"),
                    "gk_dsp_xvg": mode_entry.get("outputs", {}).get("dsp_xvg"),
                    "gk_md_xvg": mode_entry.get("outputs", {}).get("md_xvg"),
                    "gk_mj_xvg": mode_entry.get("outputs", {}).get("mj_xvg"),
                    "gk_caf_xvg": mode_entry.get("outputs", {}).get("caf_xvg"),
                    "gk_mc_xvg": mode_entry.get("outputs", {}).get("mc_xvg"),
                    "gk_error": mode_entry.get("error", ""),
                }
            )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "Trajectory ID",
                "mode",
                "gk_status",
                "gk_returncode",
                "gk_group",
                "gk_begin_ns",
                "gk_end_ns",
                "gk_sample_dt_ps",
                "gk_temperature_k",
                "gk_sigma_raw",
                "gk_sigma_s_cm",
                "gk_summary_json",
                "gk_log_path",
                "gk_current_xvg",
                "gk_dsp_xvg",
                "gk_md_xvg",
                "gk_mj_xvg",
                "gk_caf_xvg",
                "gk_mc_xvg",
                "gk_error",
            ]
        )

    if not ref_df.empty:
        keep_cols = [c for c in ["Trajectory ID", "sample_group", "CONDUCTIVITY"] if c in ref_df.columns]
        out_df = out_df.merge(ref_df[keep_cols], on="Trajectory ID", how="left")

    per_traj_csv = results_dir / "gk_results_per_traj.csv"
    out_df.to_csv(per_traj_csv, index=False)

    summary_rows = []
    if not out_df.empty and "sample_group" in out_df.columns:
        for (mode, group), sub in out_df.groupby(["mode", "sample_group"], dropna=False):
            ok = sub[sub["gk_status"] == "ok"].copy()
            sigma = pd.to_numeric(ok.get("gk_sigma_s_cm"), errors="coerce")
            ref = pd.to_numeric(ok.get("CONDUCTIVITY"), errors="coerce")
            mask = np.isfinite(sigma) & np.isfinite(ref) & (sigma > 0) & (ref > 0)
            ma_loge = float(np.mean(np.abs(np.log10(sigma[mask]) - np.log10(ref[mask])))) if mask.any() else float("nan")
            summary_rows.append(
                {
                    "mode": mode,
                    "sample_group": group,
                    "n_total": int(len(sub)),
                    "n_ok": int((sub["gk_status"] == "ok").sum()),
                    "n_sigma_parsed": int(mask.sum()),
                    "ma_loge_sigma_gk": ma_loge,
                }
            )
        if summary_rows:
            summ_df = pd.DataFrame(summary_rows).sort_values(["mode", "sample_group"])
        else:
            summ_df = pd.DataFrame(columns=["mode", "sample_group", "n_total", "n_ok", "n_sigma_parsed", "ma_loge_sigma_gk"])
    else:
        summ_df = pd.DataFrame(columns=["mode", "sample_group", "n_total", "n_ok", "n_sigma_parsed", "ma_loge_sigma_gk"])

    summary_csv = results_dir / "gk_results_summary.csv"
    summ_df.to_csv(summary_csv, index=False)

    print(f"Saved: {per_traj_csv}")
    print(f"Saved: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
