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

POLYBERT_DIR = ROOT / "MY_PAPER_RELATED" / "polybert_con"
OOF_CSV = POLYBERT_DIR / "oof_predictions.csv"
MODEL_SELECTION_CSV = POLYBERT_DIR / "weighted_model_selection_canonical_group.csv"

BASELINE_ID = "baseline_unweighted__ridge_alpha_1"
WEIGHTED_ID = "smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100"


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
            "svg.hashsalt": "SPE-DESIGN-PIPELINE-Fig7",
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
    fig.savefig(tmp_png, dpi=600, facecolor="white")
    with Image.open(tmp_png) as image:
        image.convert("RGB").save(
            tif_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw"
        )
    tmp_png.unlink(missing_ok=True)


def strip_svg_trailing_whitespace(svg_path: Path) -> None:
    lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main(output_dir: Path | None = None, output_stem: str = "Fig7") -> int:
    setup_matplotlib()
    default_output = output_dir is None

    for path in (OOF_CSV, MODEL_SELECTION_CSV):
        require(path)

    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(OOF_CSV)
    model_selection = pd.read_csv(MODEL_SELECTION_CSV)

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
        2,
        figsize=(6.20, 3.05),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.34},
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
        fontsize=7.0,
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
                fontsize=7.0,
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Metric value", fontsize=9.0)
    ax.legend(frameon=False, fontsize=7.0, loc="upper right")
    ax.tick_params(axis="y", labelsize=8.0, width=0.9, length=4)
    ax.tick_params(axis="x", width=0.9, length=4)
    add_panel_label(ax, "B")
    despine(ax)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.23, top=0.93, wspace=0.32)

    svg_path = output_dir / f"{output_stem}.svg"
    pdf_path = output_dir / f"{output_stem}.pdf"
    tif_path = output_dir / f"{output_stem}{'.tif' if default_output else '.tiff'}"
    fig.savefig(
        svg_path,
        facecolor="white",
        metadata={"Date": None},
    )
    fig.savefig(pdf_path, facecolor="white", metadata={"CreationDate": None})
    strip_svg_trailing_whitespace(svg_path)
    save_tif_600dpi(fig, tif_path)
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {tif_path}")
    print(f"Inputs: {OOF_CSV} rows={len(oof)}; {MODEL_SELECTION_CSV} rows={len(model_selection)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
