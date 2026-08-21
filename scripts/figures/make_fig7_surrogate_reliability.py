#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"

WEIGHTED_DIR = ROOT / "MY_PAPER_RELATED" / "revised" / "polybert_weighted_evidence"
OOF_CSV = WEIGHTED_DIR / "source_data" / "polybert_run" / "oof_predictions.csv"
MODEL_SELECTION_CSV = WEIGHTED_DIR / "tables" / "weighted_model_selection.csv"
CONDUCTIVITY_SUMMARY_CSV = (
    ROOT
    / "MY_PAPER_RELATED"
    / "MODELS"
    / "FCD_runs"
    / "_conductivity_eval"
    / "conductivity_eval_summary.csv"
)
PROPOSED_PRED_CACHE = (
    ROOT
    / "figures"
    / "generated_pool_enrichment"
    / "cache"
    / "proposed_transcvae_polybert_predictions.csv"
)
PROPOSED_PRECOMPUTED = {
    "LOW": (
        ROOT
        / "MY_PAPER_RELATED"
        / "polybert_con"
        / "TransCVAE"
        / "all_novel_smiles_condz_low_with_pred_conductivity.csv"
    ),
    "HIGH": (
        ROOT
        / "MY_PAPER_RELATED"
        / "polybert_con"
        / "TransCVAE"
        / "all_novel_smiles_condz_high_with_pred_conductivity.csv"
    ),
}

BASELINE_ID = "baseline_unweighted__ridge_alpha_1"
WEIGHTED_ID = "smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100"


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.9,
            "svg.fonttype": "path",
        }
    )


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_tif_600dpi(fig, tif_path: Path) -> None:
    tmp_png = tif_path.with_suffix(".tmp.png")
    fig.savefig(tmp_png, dpi=600, bbox_inches="tight", facecolor="white")
    with Image.open(tmp_png) as image:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        image.save(tif_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")
    tmp_png.unlink(missing_ok=True)


def load_proposed_histogram_predictions() -> pd.DataFrame:
    if PROPOSED_PRED_CACHE.exists():
        pred = pd.read_csv(PROPOSED_PRED_CACHE).copy()
        required = {"condition", "target_setting", "pred_log10_sigma"}
        missing = required.difference(pred.columns)
        if missing:
            raise ValueError(f"{PROPOSED_PRED_CACHE} is missing columns: {sorted(missing)}")
        pred["pred_log10_sigma"] = pd.to_numeric(pred["pred_log10_sigma"], errors="coerce")
        return pred.dropna(subset=["pred_log10_sigma"]).copy()

    rows = []
    for condition, path in PROPOSED_PRECOMPUTED.items():
        require(path)
        frame = pd.read_csv(path).copy()
        if "pred_log10_cond" not in frame.columns:
            raise ValueError(f"{path} is missing pred_log10_cond")
        tmp = pd.DataFrame(
            {
                "condition": condition,
                "target_setting": "lower target" if condition == "LOW" else "top-tail target",
                "pred_log10_sigma": pd.to_numeric(frame["pred_log10_cond"], errors="coerce"),
            }
        )
        rows.append(tmp.dropna(subset=["pred_log10_sigma"]))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    setup_matplotlib()

    for path in (OOF_CSV, MODEL_SELECTION_CSV, CONDUCTIVITY_SUMMARY_CSV):
        require(path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(OOF_CSV)
    model_selection = pd.read_csv(MODEL_SELECTION_CSV)
    conductivity_summary = pd.read_csv(CONDUCTIVITY_SUMMARY_CSV)

    required_oof = {"log10_cond", "pred_log10_cond"}
    missing_oof = required_oof.difference(oof.columns)
    if missing_oof:
        raise ValueError(f"{OOF_CSV} is missing columns: {sorted(missing_oof)}")

    model_selection = model_selection.set_index("model_id", drop=False)
    for model_id in (BASELINE_ID, WEIGHTED_ID):
        if model_id not in model_selection.index:
            raise ValueError(f"{model_id} not found in {MODEL_SELECTION_CSV}")

    baseline = model_selection.loc[BASELINE_ID]
    weighted = model_selection.loc[WEIGHTED_ID]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.45, 2.70),
        gridspec_kw={"width_ratios": [1.08, 1.02, 0.86], "wspace": 0.34},
    )
    fig.patch.set_facecolor("white")

    # A. OOF reliability for the baseline PolyBERT-Ridge model used for screening.
    ax = axes[0]
    x = pd.to_numeric(oof["log10_cond"], errors="coerce")
    y = pd.to_numeric(oof["pred_log10_cond"], errors="coerce")
    valid = x.notna() & y.notna()
    x = x.loc[valid]
    y = y.loc[valid]

    ax.scatter(x, y, s=5, color="#1f77b4", alpha=0.23, linewidths=0)
    lo = float(np.floor(min(x.min(), y.min()) * 2) / 2)
    hi = float(np.ceil(max(x.max(), y.max()) * 2) / 2)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.25", zorder=2)
    ax.axhline(-4.0, ls=":", lw=1.0, color="0.35")
    ax.axvline(-4.0, ls=":", lw=1.0, color="0.35")

    metrics_text = (
        f"MAE = {baseline['mae_log10']:.3f}\n"
        f"RMSE = {baseline['rmse_log10']:.3f}\n"
        f"R$^2$ = {baseline['r2']:.3f}\n"
        f"Pearson r = {baseline['pearson']:.3f}\n"
        f"Spearman $\\rho$ = {baseline['spearman']:.3f}"
    )
    ax.text(
        0.03,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox=dict(facecolor="white", edgecolor="0.78", boxstyle="round,pad=0.25"),
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("HTP-MD reference\nlog$_{10}(\\sigma$ / S cm$^{-1})$", fontsize=9.0)
    ax.set_ylabel("PolyBERT-Ridge predicted\nlog$_{10}(\\sigma$ / S cm$^{-1})$", fontsize=9.0)
    ax.tick_params(axis="both", labelsize=8.0, width=0.9, length=4)
    add_panel_label(ax, "A")
    despine(ax)

    # B. Screening-relevant tradeoff between baseline and weighted Ridge variants.
    ax = axes[1]
    metrics = [
        ("Recall", "recall_at_1e4"),
        ("Precision", "precision_at_1e4"),
        ("R$^2$", "r2"),
    ]
    labels = [m[0] for m in metrics]
    base_vals = [float(baseline[m[1]]) for m in metrics]
    weight_vals = [float(weighted[m[1]]) for m in metrics]
    x_pos = np.arange(len(metrics))
    width = 0.36
    bars_a = ax.bar(
        x_pos - width / 2,
        base_vals,
        width=width,
        color="#2c7fb8",
        label="Baseline",
    )
    bars_b = ax.bar(
        x_pos + width / 2,
        weight_vals,
        width=width,
        color="#ff7f0e",
        label="Weighted",
    )
    for bars in (bars_a, bars_b):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.018,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Metric value", fontsize=9.0)
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    ax.tick_params(axis="y", labelsize=8.0, width=0.9, length=4)
    ax.tick_params(axis="x", width=0.9, length=4)
    add_panel_label(ax, "B")
    despine(ax)

    # C. Generated-pool predicted-conductivity distribution.
    # The histogram shows the cached candidate-level prediction distribution.
    # The hit annotations use the repeat-level summary used in Table 1.
    ax = axes[2]
    pred_pool = load_proposed_histogram_predictions()
    trans = conductivity_summary.loc[
        conductivity_summary["model"].eq("TransCVAE")
        & conductivity_summary["condition"].isin(["LOW", "HIGH"])
    ].copy()
    if len(trans) != 2:
        raise ValueError(
            "Expected exactly LOW and HIGH TransCVAE rows in "
            f"{CONDUCTIVITY_SUMMARY_CSV}, found {len(trans)}."
        )
    order = ["LOW", "HIGH"]
    label_map = {"LOW": "lower target", "HIGH": "top-tail target"}
    color_map = {"LOW": "#9eb6d3", "HIGH": "#ef9a9a"}
    trans["condition"] = pd.Categorical(trans["condition"], categories=order, ordered=True)
    trans = trans.sort_values("condition")
    means = trans["hit_1e4_mean"].astype(float).to_numpy()

    bins = np.linspace(
        float(np.nanpercentile(pred_pool["pred_log10_sigma"], 0.3)) - 0.05,
        float(np.nanpercentile(pred_pool["pred_log10_sigma"], 99.7)) + 0.05,
        42,
    )
    density_max = 0.0
    for condition in order:
        subset = pred_pool.loc[pred_pool["condition"].eq(condition), "pred_log10_sigma"]
        density, _, _ = ax.hist(
            subset,
            bins=bins,
            density=True,
            alpha=0.72,
            color=color_map[condition],
            edgecolor="white",
            linewidth=0.25,
            label=label_map[condition],
        )
        density_max = max(density_max, float(np.nanmax(density)))

    ax.axvline(-4.0, color="0.25", lw=1.1, ls="--", zorder=2)
    ax.text(
        -4.02,
        0.70,
        "10$^{-4}$ S cm$^{-1}$ surrogate hit threshold",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=4.8,
        color="0.20",
    )
    ax.text(
        0.04,
        0.88,
        f"lower hit = {means[0]:.3f}\n"
        f"top-tail hit = {means[1]:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        bbox=dict(facecolor="white", edgecolor="0.78", boxstyle="round,pad=0.20"),
        zorder=8,
    )
    ax.set_ylim(0, max(3.80, density_max * 1.34))
    ax.set_xlabel("PolyBERT-Ridge predicted\nlog$_{10}(\\sigma$ / S cm$^{-1})$", fontsize=9.0)
    ax.set_ylabel("Density", fontsize=9.0)
    ax.legend(
        frameon=False,
        fontsize=4.8,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.91),
        borderaxespad=0.20,
        handlelength=0.95,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    ax.tick_params(axis="y", labelsize=8.0, width=0.9, length=4)
    ax.tick_params(axis="x", labelsize=8.0, width=0.9, length=4)
    add_panel_label(ax, "C")
    despine(ax)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.96, wspace=0.34)

    svg_path = FIGURES_DIR / "Fig7.svg"
    tif_path = FIGURES_DIR / "Fig7.tif"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    save_tif_600dpi(fig, tif_path)
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {tif_path}")
    print(
        "Panel C source:",
        CONDUCTIVITY_SUMMARY_CSV,
        f"lower={means[0]:.6f}",
        f"top-tail={means[1]:.6f}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
