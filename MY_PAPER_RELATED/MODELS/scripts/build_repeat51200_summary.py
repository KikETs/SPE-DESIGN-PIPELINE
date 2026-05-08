#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SUFFIX = "repeat5_51200"
CONDITION_ORDER = ["HIGH", "LOW"]
COND_DIR_MAP = {"LOW": "condz_low", "HIGH": "condz_high"}


def safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return np.nan
    return float(a / b)


def safe_std(s: pd.Series) -> float:
    if len(s) <= 1:
        return 0.0
    v = float(s.std(ddof=1))
    return 0.0 if np.isnan(v) else v


def safe_var(s: pd.Series) -> float:
    if len(s) <= 1:
        return 0.0
    v = float(s.var(ddof=1))
    return 0.0 if np.isnan(v) else v


def bold_md(s: str) -> str:
    return f"**{s}**"


def pm_fmt(mean_v: float, std_v: float) -> str:
    if pd.isna(mean_v):
        return "NaN"
    if pd.isna(std_v):
        return f"{mean_v:.3f} ± NaN"
    return f"{mean_v:.3f} ± {std_v:.3f}"


def md_to_tex_bold(s: str) -> str:
    if s.startswith("**") and s.endswith("**"):
        return f"\\textbf{{{s[2:-2]}}}"
    return s


def best_mask(values: np.ndarray, condition: str, metric_key: str) -> np.ndarray:
    arr = values.astype(float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=bool)
    if not finite.any():
        return out

    if metric_key == "fcd_mean":
        target = np.nanmin(arr)
        return np.isclose(arr, target, rtol=1e-9, atol=1e-12)

    if metric_key in ("mean_log10_cond_mean", "hit_1e4_mean", "hit_1e3_mean"):
        target = np.nanmax(arr) if condition == "HIGH" else np.nanmin(arr)
        return np.isclose(arr, target, rtol=1e-9, atol=1e-12)

    target = np.nanmax(arr)
    return np.isclose(arr, target, rtol=1e-9, atol=1e-12)


def find_fcd_root(start: Path) -> Path:
    for base in [start.resolve()] + list(start.resolve().parents):
        cand = base / "MY_PAPER_RELATED" / "MODELS" / "FCD_runs"
        if cand.exists():
            return cand
        cand2 = base / "MODELS" / "FCD_runs"
        if cand2.exists():
            return cand2
    raise FileNotFoundError("Could not locate MY_PAPER_RELATED/MODELS/FCD_runs")


def infer_mingpt_cond_map(by_repeat_csv: Path) -> dict[int, str]:
    if not by_repeat_csv.exists():
        return {0: "LOW", 1: "HIGH"}
    df = pd.read_csv(by_repeat_csv)
    df = df[df["model"] == "minGPT"].copy()
    cond_map: dict[int, str] = {}
    for _, row in df.iterrows():
        src = str(row.get("source_csv", ""))
        m = re.search(r"generated_cond(\d+)_repeat", src)
        if m is None:
            continue
        cid = int(m.group(1))
        cond = str(row.get("condition", "")).upper()
        if cond in ("LOW", "HIGH"):
            cond_map[cid] = cond
    if cond_map:
        return cond_map
    return {0: "LOW", 1: "HIGH"}


def collect_repeat_totals_non_mingpt(fcd_root: Path) -> list[dict]:
    rows: list[dict] = []
    model_dirs = []
    for p in sorted(fcd_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        if p.name.startswith("minGPT"):
            continue
        model_dirs.append(p)

    for model_dir in model_dirs:
        model_name = model_dir.name
        for cond_label, cond_dir in COND_DIR_MAP.items():
            cdir = model_dir / cond_dir
            if not cdir.exists():
                continue
            repeat_dirs = sorted([p for p in cdir.glob("repeat_*") if p.is_dir()])
            for rep_dir in repeat_dirs:
                files = sorted(rep_dir.glob(f"fcd_summary_{cond_dir}_repeat_*.csv"))
                if not files:
                    files = sorted(rep_dir.glob("fcd_summary_*_repeat_*.csv"))
                if not files:
                    continue
                sfile = files[0]
                sdf = pd.read_csv(sfile)
                need = {"batch_size", "valid", "valid_unique", "valid_unique_novel"}
                if not need.issubset(set(sdf.columns)):
                    continue

                total_samples = float(sdf["batch_size"].astype(float).sum())
                valid_total = float(sdf["valid"].astype(float).sum())
                valid_unique_total = float(sdf["valid_unique"].astype(float).sum())
                valid_unique_novel_total = float(sdf["valid_unique_novel"].astype(float).sum())

                all_novel_unique = np.nan
                if "all_novel_unique" in sdf.columns:
                    v = sdf["all_novel_unique"].dropna()
                    if not v.empty:
                        all_novel_unique = float(v.iloc[0])

                fcd = np.nan
                if "fcd_all_novel_vs_original" in sdf.columns:
                    v = sdf["fcd_all_novel_vs_original"].dropna()
                    if not v.empty:
                        fcd = float(v.iloc[0])

                rows.append(
                    {
                        "model": model_name,
                        "condition": cond_label,
                        "repeat": rep_dir.name,
                        "total_samples": total_samples,
                        "valid_total": valid_total,
                        "valid_unique_total": valid_unique_total,
                        "valid_unique_novel_total": valid_unique_novel_total,
                        "all_novel_unique": all_novel_unique,
                        "fcd": fcd,
                        "source": str(sfile),
                    }
                )
    return rows


def collect_repeat_totals_mingpt(fcd_root: Path, cond_map: dict[int, str]) -> list[dict]:
    rows: list[dict] = []
    mdir = fcd_root / "minGPT_cond_repeat5_of_50_results"
    gm_path = mdir / "generation_metrics.csv"
    rm_path = mdir / "repeat_metrics.csv"
    if not gm_path.exists() or not rm_path.exists():
        return rows

    gm = pd.read_csv(gm_path)
    rm = pd.read_csv(rm_path)

    for cid in sorted(gm["target_conductivity"].dropna().unique()):
        cid_int = int(cid)
        cond = cond_map.get(cid_int, f"COND_{cid_int}")
        gm_c = gm[gm["target_conductivity"] == cid_int].copy()
        repeats = sorted(gm_c["repeat"].dropna().astype(int).unique().tolist())
        for rep in repeats:
            gm_r = gm_c[gm_c["repeat"].astype(int) == rep].copy()
            if gm_r.empty:
                continue
            n = gm_r["num_samples"].astype(float).to_numpy()
            valid = gm_r["validity"].astype(float).to_numpy()
            uniq = gm_r["uniqueness"].astype(float).to_numpy()
            novel = gm_r["novelty"].astype(float).to_numpy()

            total_samples = float(n.sum())
            valid_total = float((valid * n).sum())
            valid_unique_total = float((valid * n * uniq).sum())
            valid_unique_novel_total = float((valid * n * uniq * novel).sum())

            rm_r = rm[(rm["target_conductivity"] == cid_int) & (rm["repeat"].astype(int) == rep)]
            all_novel_unique = np.nan
            fcd = np.nan
            if not rm_r.empty:
                all_novel_unique = float(rm_r["novel_unique_count_repeat"].iloc[0])
                fcd = float(rm_r["fcd_vs_full_reference"].iloc[0])

            rows.append(
                {
                    "model": "minGPT",
                    "condition": cond,
                    "repeat": f"repeat_{rep:02d}",
                    "total_samples": total_samples,
                    "valid_total": valid_total,
                    "valid_unique_total": valid_unique_total,
                    "valid_unique_novel_total": valid_unique_novel_total,
                    "all_novel_unique": all_novel_unique,
                    "fcd": fcd,
                    "source": str(mdir),
                }
            )
    return rows


def build_paper_tables(full_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("valid_mean", "valid_std", "Valid"),
        ("valid_unique_mean", "valid_unique_std", "Valid-Unique"),
        ("valid_unique_novel_mean", "valid_unique_novel_std", "Valid-Unique-Novel"),
        ("all_novel_unique_mean", "all_novel_unique_std", "All-Novel-Unique"),
        ("fcd_mean", "fcd_std", "FCD"),
        ("mean_log10_cond_mean", "mean_log10_cond_std", "Pred log10(cond)"),
        ("hit_1e4_mean", "hit_1e4_std", "Hit@1e-4"),
        ("hit_1e3_mean", "hit_1e3_std", "Hit@1e-3"),
    ]

    sections = []
    for cond in CONDITION_ORDER:
        sub = full_df[full_df["condition"] == cond].copy().reset_index(drop=True)
        num_cols = ["model"]
        for m, s, _ in metrics:
            num_cols.extend([m, s])
        numeric = sub[num_cols].copy()

        display = pd.DataFrame({"Model": sub["model"]})
        for m, s, label in metrics:
            values = sub[m].to_numpy(dtype=float)
            stds = sub[s].to_numpy(dtype=float)
            text_vals = [pm_fmt(v, st) for v, st in zip(values, stds)]
            mask = best_mask(values, cond, m)
            text_vals = [bold_md(t) if b else t for t, b in zip(text_vals, mask)]
            display[label] = text_vals

        num_path = out_dir / f"paper_table_{cond.lower()}_numeric_{SUFFIX}.csv"
        disp_path = out_dir / f"paper_table_{cond.lower()}_display_{SUFFIX}.csv"
        md_path = out_dir / f"paper_table_{cond.lower()}_{SUFFIX}.md"
        tex_path = out_dir / f"paper_table_{cond.lower()}_{SUFFIX}.tex"

        numeric.to_csv(num_path, index=False)
        display.to_csv(disp_path, index=False)
        md_text = f"# {cond} Condition\n\n" + display.to_markdown(index=False) + "\n"
        md_path.write_text(md_text, encoding="utf-8")

        tex_df = display.copy()
        for c in tex_df.columns:
            tex_df[c] = tex_df[c].astype(str).map(md_to_tex_bold)
        tex_path.write_text(tex_df.to_latex(index=False, escape=False), encoding="utf-8")

        sections.append(md_text.rstrip())

    all_md = "\n\n\n".join(sections) + "\n"
    (out_dir / f"paper_tables_all_conditions_{SUFFIX}.md").write_text(all_md, encoding="utf-8")


def main() -> None:
    cwd = Path.cwd()
    fcd_root = find_fcd_root(cwd)
    src_eval = fcd_root / "_conductivity_eval"
    out_dir = fcd_root / f"_conductivity_eval_{SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_summary_path = src_eval / "conductivity_eval_summary.csv"
    by_repeat_path = src_eval / "conductivity_eval_by_repeat.csv"
    full_template_path = src_eval / "all_models_full_eval_with_polybert.csv"
    if not cond_summary_path.exists() or not full_template_path.exists():
        raise FileNotFoundError("Missing source _conductivity_eval files")

    cond_map = infer_mingpt_cond_map(by_repeat_path)

    rows = []
    rows.extend(collect_repeat_totals_non_mingpt(fcd_root))
    rows.extend(collect_repeat_totals_mingpt(fcd_root, cond_map))
    if not rows:
        raise RuntimeError("No repeat-level records were collected.")

    repeat_df = pd.DataFrame(rows)
    repeat_df["valid_rate"] = repeat_df.apply(
        lambda r: safe_div(r["valid_total"], r["total_samples"]), axis=1
    )
    repeat_df["unique_over_valid"] = repeat_df.apply(
        lambda r: safe_div(r["valid_unique_total"], r["valid_total"]), axis=1
    )
    repeat_df["novel_over_unique"] = repeat_df.apply(
        lambda r: safe_div(r["valid_unique_novel_total"], r["valid_unique_total"]), axis=1
    )

    summary_df = (
        repeat_df.groupby(["model", "condition"], as_index=False)
        .agg(
            n_run=("repeat", "nunique"),
            valid_mean=("valid_total", "mean"),
            valid_std=("valid_total", safe_std),
            valid_var=("valid_total", safe_var),
            valid_unique_mean=("valid_unique_total", "mean"),
            valid_unique_std=("valid_unique_total", safe_std),
            valid_unique_var=("valid_unique_total", safe_var),
            valid_unique_novel_mean=("valid_unique_novel_total", "mean"),
            valid_unique_novel_std=("valid_unique_novel_total", safe_std),
            valid_unique_novel_var=("valid_unique_novel_total", safe_var),
            valid_rate_mean=("valid_rate", "mean"),
            valid_rate_var=("valid_rate", safe_var),
            unique_over_valid_mean=("unique_over_valid", "mean"),
            unique_over_valid_var=("unique_over_valid", safe_var),
            novel_over_unique_mean=("novel_over_unique", "mean"),
            novel_over_unique_var=("novel_over_unique", safe_var),
            repeat_evals=("repeat", "nunique"),
            all_novel_unique_mean=("all_novel_unique", "mean"),
            all_novel_unique_std=("all_novel_unique", safe_std),
            all_novel_unique_var=("all_novel_unique", safe_var),
            fcd_mean=("fcd", "mean"),
            fcd_std=("fcd", safe_std),
            fcd_var=("fcd", safe_var),
        )
        .reset_index(drop=True)
    )

    cond_summary = pd.read_csv(cond_summary_path)
    cond_cols = [
        "model",
        "condition",
        "repeats",
        "n_samples_mean",
        "mean_log10_cond_mean",
        "mean_log10_cond_std",
        "median_log10_cond_mean",
        "median_log10_cond_std",
        "q90_log10_cond_mean",
        "q90_log10_cond_std",
        "hit_1e4_mean",
        "hit_1e4_std",
        "hit_1e3_mean",
        "hit_1e3_std",
    ]
    cond_summary = cond_summary[cond_cols].copy()
    cond_summary["mean_log10_cond_var"] = cond_summary["mean_log10_cond_std"] ** 2
    cond_summary["median_log10_cond_var"] = cond_summary["median_log10_cond_std"] ** 2
    cond_summary["q90_log10_cond_var"] = cond_summary["q90_log10_cond_std"] ** 2
    cond_summary["hit_1e4_var"] = cond_summary["hit_1e4_std"] ** 2
    cond_summary["hit_1e3_var"] = cond_summary["hit_1e3_std"] ** 2

    full = summary_df.merge(cond_summary, on=["model", "condition"], how="left")
    full["repeats"] = full["repeats"].fillna(full["n_run"])

    template_cols = pd.read_csv(full_template_path, nrows=1).columns.tolist()
    for col in template_cols:
        if col not in full.columns:
            full[col] = np.nan
    extra_cols = [c for c in full.columns if c.endswith("_var") and c not in template_cols]
    full = full[template_cols + extra_cols].copy()

    cat = pd.CategoricalDtype(categories=CONDITION_ORDER, ordered=True)
    full["condition"] = full["condition"].astype(cat)
    full = full.sort_values(["condition", "model"]).reset_index(drop=True)
    full["condition"] = full["condition"].astype(str)

    # save raw repeat-level and summary-level files
    repeat_out = out_dir / f"repeat_level_counts_{SUFFIX}.csv"
    summary_out = out_dir / f"repeat_level_summary_{SUFFIX}.csv"
    repeat_df.to_csv(repeat_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    full_out = out_dir / f"all_models_full_eval_with_polybert_{SUFFIX}.csv"
    raw_out = out_dir / f"paper_table_all_conditions_raw_{SUFFIX}.csv"
    score_out = out_dir / f"all_models_scorecard_{SUFFIX}.csv"
    full.to_csv(full_out, index=False)
    full.to_csv(raw_out, index=False)

    score_cols = [
        "model",
        "condition",
        "valid_mean",
        "valid_unique_mean",
        "valid_unique_novel_mean",
        "all_novel_unique_mean",
        "fcd_mean",
        "mean_log10_cond_mean",
        "hit_1e4_mean",
        "hit_1e3_mean",
    ]
    full[score_cols].to_csv(score_out, index=False)

    build_paper_tables(full, out_dir)

    print("saved:", repeat_out)
    print("saved:", summary_out)
    print("saved:", full_out)
    print("saved:", raw_out)
    print("saved:", score_out)
    print("saved tables in:", out_dir)


if __name__ == "__main__":
    main()
