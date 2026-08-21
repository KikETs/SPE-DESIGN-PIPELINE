#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kendalltau, mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"

RUN_RESULTS_CSV = (
    ROOT / "MY_PAPER_RELATED" / "machine_readable" / "reference_run_status_120.csv"
)
CNE0_SOURCE_CSV = (
    ROOT
    / "MY_PAPER_RELATED"
    / "machine_readable"
    / "figure_data"
    / "fig8_reference_stratified_static_cne0.csv"
)

GROUP_ORDER = ["bottom", "middle_stratified", "top"]
GROUP_LABELS = {
    "bottom": "Bottom",
    "middle_stratified": "Middle-\nstratified",
    "top": "Top",
}
GROUP_COLORS = {
    "bottom": "#4c78a8",
    "middle_stratified": "#f58518",
    "top": "#54a24b",
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.9,
            "svg.fonttype": "path",
        }
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.17,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
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


def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    yv = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    if len(xv) == 0 or len(yv) == 0:
        return float("nan")
    signs = [np.sign(a - b) for a in xv for b in yv]
    return float(np.mean(signs))


def load_data() -> tuple[pd.DataFrame, int]:
    require(RUN_RESULTS_CSV)
    require(CNE0_SOURCE_CSV)

    run = pd.read_csv(RUN_RESULTS_CSV)
    total_selected = int(run["sample_group"].isin(GROUP_ORDER).sum())

    data = pd.read_csv(CNE0_SOURCE_CSV)
    expected_filter = (
        data["sample_group"].isin(GROUP_ORDER)
        & data["protocol"].eq("manuscript_static_cNE0")
        & data["cutoff_source"].eq("Li_TFSI_O_RDF_first_minimum")
        & data["association_filter"].eq("Li_TFSI_O_contacts_ge_2")
    )
    if not expected_filter.all():
        raise ValueError("Published Fig. 8 source contains rows outside the documented filter")

    numeric_cols = [
        "sigma_static_cNE0_S_cm",
        "selection_reference_conductivity_S_cm",
        "sigma_NE_S_cm",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(
        subset=["sigma_static_cNE0_S_cm", "selection_reference_conductivity_S_cm"]
    )
    data = data[
        data["sigma_static_cNE0_S_cm"].gt(0)
        & data["selection_reference_conductivity_S_cm"].gt(0)
    ].copy()

    data["log10_reference"] = np.log10(
        data["selection_reference_conductivity_S_cm"]
    )
    data["log10_static_cne0"] = np.log10(data["sigma_static_cNE0_S_cm"])
    data["log10_ratio"] = data["log10_static_cne0"] - data["log10_reference"]
    data["ratio"] = (
        data["sigma_static_cNE0_S_cm"]
        / data["selection_reference_conductivity_S_cm"]
    )
    data["abs_log_error"] = data["log10_ratio"].abs()
    data["group"] = pd.Categorical(
        data["sample_group"], categories=GROUP_ORDER, ordered=True
    )
    return data.sort_values(["group", "trajectory_id"]), total_selected


def draw_group_box(ax, data: pd.DataFrame, value_col: str, ylabel: str, rng: np.random.Generator) -> None:
    positions = np.arange(len(GROUP_ORDER))
    values = [data.loc[data["group"].eq(group), value_col].dropna().to_numpy() for group in GROUP_ORDER]
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="0.25", linewidth=0.8),
        capprops=dict(color="0.25", linewidth=0.8),
        boxprops=dict(color="0.25", linewidth=0.8),
    )
    for patch, group in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(0.32)

    for idx, group in enumerate(GROUP_ORDER):
        subset = data.loc[data["group"].eq(group), value_col].dropna().to_numpy()
        jitter = rng.normal(0.0, 0.035, size=len(subset))
        ax.scatter(
            np.full(len(subset), idx) + jitter,
            subset,
            s=18,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER])
    ax.set_ylabel(ylabel)
    ax.tick_params(width=0.85, length=3.5)
    despine(ax)


def make_figure(data: pd.DataFrame, total_selected: int) -> tuple[plt.Figure, dict[str, float]]:
    rng = np.random.default_rng(7)
    completed = len(data)
    maloge = float(data["abs_log_error"].mean())
    median_ratio = float(data["ratio"].median())
    spearman = float(spearmanr(data["log10_reference"], data["log10_static_cne0"]).statistic)
    kendall = float(kendalltau(data["log10_reference"], data["log10_static_cne0"]).statistic)

    bottom = data.loc[data["group"].eq("bottom"), "log10_static_cne0"]
    top = data.loc[data["group"].eq("top"), "log10_static_cne0"]
    median_shift = float(top.median() - bottom.median())
    mw = mannwhitneyu(top, bottom, alternative="two-sided")
    delta = cliffs_delta(top, bottom)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.45, 2.70),
        gridspec_kw={"width_ratios": [1.12, 0.82, 0.86], "wspace": 0.34},
    )
    fig.patch.set_facecolor("white")

    # A. Reference-label conductivity versus corrected static cNE0 reassessment.
    ax = axes[0]
    for group in GROUP_ORDER:
        subset = data[data["group"].eq(group)]
        ax.scatter(
            subset["log10_reference"],
            subset["log10_static_cne0"],
            s=20,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            label=GROUP_LABELS[group].replace("\n", " "),
        )
    lo = float(np.floor(min(data["log10_reference"].min(), data["log10_static_cne0"].min()) * 2) / 2)
    hi = float(np.ceil(max(data["log10_reference"].max(), data["log10_static_cne0"].max()) * 2) / 2)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color="0.38", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("HTP-MD reference\nlog$_{10}$($\\sigma$ / S cm$^{-1}$)")
    ax.set_ylabel("GROMACS static cNE0\nlog$_{10}$($\\sigma$ / S cm$^{-1}$)")
    ax.legend(frameon=False, loc="lower right", handlelength=0.9, borderaxespad=0.25)
    ax.text(
        0.04,
        0.96,
        f"Completed = {completed}/{total_selected}\n"
        f"MALogE = {maloge:.3f}\n"
        f"Median ratio = {median_ratio:.2f}\n"
        f"Spearman $\\rho$ = {spearman:.3f}\n"
        f"Kendall $\\tau$ = {kendall:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        bbox=dict(facecolor="white", edgecolor="0.78", boxstyle="round,pad=0.22"),
    )
    add_panel_label(ax, "A")
    ax.tick_params(width=0.85, length=3.5)
    despine(ax)

    # B. Corrected static cNE0 distribution by reference stratum.
    ax = axes[1]
    draw_group_box(
        ax,
        data,
        "log10_static_cne0",
        "GROMACS static cNE0\nlog$_{10}$($\\sigma$ / S cm$^{-1}$)",
        rng,
    )
    ax.text(
        0.05,
        0.97,
        f"Top-bottom median shift = {median_shift:.3f} on log$_{{10}}$ scale\n"
        f"Mann-Whitney U = {mw.statistic:.1f}\n"
        f"p = {mw.pvalue:.2e}\n"
        f"Cliff's $\\delta$ = {delta:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox=dict(facecolor="white", edgecolor="0.78", boxstyle="round,pad=0.22"),
    )
    add_panel_label(ax, "B")

    # C. Static cNE0/reference ratio by reference stratum.
    ax = axes[2]
    draw_group_box(
        ax,
        data,
        "log10_ratio",
        "log$_{10}$(static cNE0 / reference)",
        rng,
    )
    ax.axhline(0, ls="--", lw=0.9, color="0.35", zorder=1)
    ratio_lines = [
        f"{GROUP_LABELS[group].replace(chr(10), ' ')} = {data.loc[data['group'].eq(group), 'ratio'].median():.3g}"
        for group in GROUP_ORDER
    ]
    ax.text(
        0.05,
        0.97,
        "Median ratio\n" + "\n".join(ratio_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox=dict(facecolor="white", edgecolor="0.78", boxstyle="round,pad=0.22"),
    )
    add_panel_label(ax, "C")

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.94, wspace=0.34)
    stats = {
        "completed": completed,
        "total_selected": total_selected,
        "maloge": maloge,
        "median_ratio": median_ratio,
        "spearman": spearman,
        "kendall": kendall,
        "median_shift": median_shift,
        "mann_whitney_u": float(mw.statistic),
        "mann_whitney_p": float(mw.pvalue),
        "cliffs_delta": delta,
    }
    return fig, stats


def main() -> int:
    setup_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data, total_selected = load_data()
    fig, stats = make_figure(data, total_selected)

    svg_path = FIGURES_DIR / "Fig8.svg"
    tif_path = FIGURES_DIR / "Fig8.tif"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    save_tif_600dpi(fig, tif_path)
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {tif_path}")
    print(
        "Source:",
        CNE0_SOURCE_CSV,
        "validated_filter=cutoff_source:Li_TFSI_O_RDF_first_minimum,association_filter:Li_TFSI_O_contacts_ge_2",
    )
    print(
        "Stats:",
        f"completed={stats['completed']}/{stats['total_selected']}",
        f"MALogE={stats['maloge']:.6f}",
        f"median_shift={stats['median_shift']:.6f}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
