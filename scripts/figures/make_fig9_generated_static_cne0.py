#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
SOURCE_CSV = (
    ROOT
    / "MY_PAPER_RELATED"
    / "gromacs_eval_pred_conductivity"
    / "reproducibility"
    / "static_cne0_release"
    / "data"
    / "generated"
    / "generated_md_results_60.csv"
)

GROUP_ORDER = [
    "HIGH_top",
    "HIGH_middle_stratified",
    "HIGH_bottom",
    "LOW_top",
    "LOW_middle_stratified",
    "LOW_bottom",
]
GROUP_LABELS = {
    "HIGH_top": "TT-top",
    "HIGH_middle_stratified": "TT-middle",
    "HIGH_bottom": "TT-bottom",
    "LOW_top": "LT-top",
    "LOW_middle_stratified": "LT-middle",
    "LOW_bottom": "LT-bottom",
}
GROUP_COLORS = {
    "HIGH_top": "#1f5b95",
    "HIGH_middle_stratified": "#4c78a8",
    "HIGH_bottom": "#8fb7dd",
    "LOW_top": "#c64616",
    "LOW_middle_stratified": "#e67e22",
    "LOW_bottom": "#f2a51a",
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.9,
            "svg.fonttype": "path",
        }
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def save_tif_600dpi(fig, tif_path: Path) -> None:
    temp = tif_path.with_suffix(".tmp.png")
    fig.savefig(temp, dpi=600, bbox_inches="tight", facecolor="white")
    with Image.open(temp) as image:
        image.convert("RGB").save(
            tif_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw"
        )
    temp.unlink(missing_ok=True)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_CSV)
    required = {
        "trajectory_id",
        "sample_group",
        "production_replicas",
        "temperature_K",
        "CONDUCTIVITY",
        "sigma_static_cNE0_mean_S_cm",
        "sigma_static_cNE0_std_S_cm",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing generated Fig. 9 columns: {sorted(missing)}")
    if len(data) != 60 or data["trajectory_id"].nunique() != 60:
        raise ValueError("Fig. 9 requires 60 unique generated candidates")
    counts = data.groupby("sample_group")["trajectory_id"].nunique()
    if set(counts.index) != set(GROUP_ORDER) or not counts.eq(10).all():
        raise ValueError(f"Fig. 9 requires 10 candidates in each group: {counts.to_dict()}")
    if not data["production_replicas"].eq(3).all():
        raise ValueError("Fig. 9 requires three production replicas per candidate")
    if not data["temperature_K"].eq(353.0).all():
        raise ValueError("Fig. 9 source contains non-353 K rows")
    numeric = ["CONDUCTIVITY", "sigma_static_cNE0_mean_S_cm"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric].isna().any().any() or not data[numeric].gt(0).all().all():
        raise ValueError("Fig. 9 conductivity values must be finite and positive")
    data["log10_polybert"] = np.log10(data["CONDUCTIVITY"])
    data["log10_static_cNE0"] = np.log10(data["sigma_static_cNE0_mean_S_cm"])
    return data


def make_figure(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=(7.8, 3.25), gridspec_kw={"wspace": 0.24}
    )
    fig.patch.set_facecolor("white")
    y_label = r"GROMACS static cNE0 log$_{10}$($\sigma$ / S cm$^{-1}$)"

    ax = axes[0]
    for position, group in enumerate(GROUP_ORDER):
        subset = data[data["sample_group"].eq(group)].sort_values(
            "log10_static_cNE0"
        )
        offsets = np.linspace(-0.16, 0.16, len(subset))
        ax.scatter(
            np.full(len(subset), position) + offsets,
            subset["log10_static_cNE0"],
            s=18,
            facecolor=GROUP_COLORS[group],
            edgecolor="black",
            linewidth=0.30,
            alpha=0.95,
            zorder=3,
        )
        mean_value = subset["log10_static_cNE0"].mean()
        ax.plot(
            [position - 0.23, position + 0.23],
            [mean_value, mean_value],
            color="black",
            lw=1.05,
            zorder=4,
        )
    ax.axhline(-4.0, color="0.50", lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(-0.45, len(GROUP_ORDER) - 0.55)
    ax.set_ylim(-6.5, -2.3)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(
        [GROUP_LABELS[group] for group in GROUP_ORDER], rotation=28, ha="right"
    )
    ax.set_yticks(np.arange(-6.0, -2.0, 0.5))
    ax.set_ylabel(y_label)
    panel_label(ax, "A")
    despine(ax)

    ax = axes[1]
    for group in GROUP_ORDER:
        subset = data[data["sample_group"].eq(group)]
        ax.scatter(
            subset["log10_polybert"],
            subset["log10_static_cNE0"],
            s=20,
            facecolor=GROUP_COLORS[group],
            edgecolor="black",
            linewidth=0.30,
            alpha=0.95,
            label=GROUP_LABELS[group],
            zorder=3,
        )
    low = min(data["log10_polybert"].min(), data["log10_static_cNE0"].min()) - 0.12
    high = max(data["log10_polybert"].max(), data["log10_static_cNE0"].max()) + 0.12
    ax.plot([low, high], [low, high], color="0.55", lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(-6.5, -2.25)
    ax.set_ylim(-6.5, -2.3)
    ax.set_xticks([-6, -5, -4, -3])
    ax.set_yticks(np.arange(-6.0, -2.0, 0.5))
    ax.set_xlabel(r"PolyBERT-Ridge predicted log$_{10}$($\sigma$ / S cm$^{-1}$)")
    ax.set_ylabel(y_label)
    panel_label(ax, "B")
    ax.legend(
        loc="lower right",
        frameon=False,
        handletextpad=0.30,
        borderaxespad=0.20,
        ncol=2,
        columnspacing=0.80,
    )
    despine(ax)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.92, wspace=0.28)
    return fig


def main() -> int:
    setup_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    fig = make_figure(data)
    svg_path = FIGURES_DIR / "Fig9.svg"
    tif_path = FIGURES_DIR / "Fig9.tif"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    save_tif_600dpi(fig, tif_path)
    plt.close(fig)
    print(f"Saved: {svg_path}")
    print(f"Saved: {tif_path}")
    print(f"Source: {SOURCE_CSV}")
    print("Validated: candidates=60 replicas=180 temperature_K=353")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
