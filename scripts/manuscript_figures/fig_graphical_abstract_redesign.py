from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

from scripts.manuscript_figures.common import (
    FIGURES_DIR,
    FONT_FAMILY,
    MATH_FONT_FAMILY,
    ROOT,
    font_available,
    math_font_family,
    save_tif_600dpi,
    setup_liberation_sans,
)


CANVAS_W = 2048
CANVAS_H = 1068
OUTPUT_NAME = "GraphicalAbstract_redesign"
MD_BOX_SOURCE = ROOT / "assets" / "manuscript_figures" / "md_box_source.png"
LEGACY_GA_PATH = FIGURES_DIR / "GraphicalAbstract.tif"
MD_BOX_CROP = (5580, 1320, 7000, 2540)

PURPLE = "#5A1CFF"
TEAL = "#008D99"
BLACK = "#111111"
O_RED = "#FF0000"
N_BLUE = "#001CFF"
MAJOR_LABEL_SIZE = 16
TRAINING_GRAY = "#777777"


def _math_family() -> str:
    return MATH_FONT_FAMILY if font_available(MATH_FONT_FAMILY) else math_font_family()


def _text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 13,
    weight: str = "normal",
    color: str = BLACK,
    ha: str = "center",
    va: str = "center",
    family: str | None = None,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        fontfamily=family or FONT_FAMILY,
    )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = BLACK, lw: float = 1.8) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": lw,
            "color": color,
            "mutation_scale": 18,
            "joinstyle": "miter",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def _line_arrow_path(ax, points: list[tuple[float, float]], *, color: str, lw: float = 2.0) -> None:
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", solid_joinstyle="miter")
    _arrow(ax, points[-2], points[-1], color=color, lw=lw)


def _dashed_arrow_path(ax, points: list[tuple[float, float]], *, color: str, lw: float = 1.35) -> None:
    arrow_len = 13.0
    end_x, end_y = points[-1]
    prev_x, prev_y = points[-2]
    dx = end_x - prev_x
    dy = end_y - prev_y
    length = (dx**2 + dy**2) ** 0.5
    if length > arrow_len:
        head_start = (end_x - arrow_len * dx / length, end_y - arrow_len * dy / length)
    else:
        head_start = points[-2]

    line_points = [*points[:-1], head_start]
    xs, ys = zip(*line_points)
    ax.plot(
        xs,
        ys,
        color=color,
        lw=lw,
        linestyle=(0, (4, 4)),
        solid_capstyle="butt",
        solid_joinstyle="miter",
    )
    ax.add_patch(
        patches.FancyArrowPatch(
            head_start,
            points[-1],
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=0,
            facecolor=color,
            edgecolor=color,
        )
    )


def _round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str = BLACK,
    face: str = "white",
    lw: float = 1.7,
    radius: float = 14,
    linestyle: str = "solid",
) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            linestyle=linestyle,
        )
    )


def _card_stack(ax) -> None:
    x, y, w, h = 72, 404, 245, 224
    for dx, dy in [(22, -36), (11, -22), (0, 0)]:
        _round_box(ax, x + dx, y + dy, w, h, edge=BLACK, lw=1.6, radius=14)

    _text(ax, x + w / 2, y + 54, "[*] O C C O [*]", size=13.2, weight="bold")
    _text(ax, x + w / 2, y + 112, "[*] C(=O) O C [*]", size=13.2, weight="bold")
    _text(ax, x + w / 2, y + 174, "...", size=17, weight="bold")
    _text(ax, x + w / 2, y + h + 43, "endpoints preserved", size=14)
    _text(ax, x + w / 2, y + h + 80, "SELFIES-PSMILES", size=MAJOR_LABEL_SIZE, weight="bold")
    _text(ax, x + w / 2, y + h + 116, "corpus", size=MAJOR_LABEL_SIZE, weight="bold")
    _dashed_arrow_path(ax, [(330, 356), (438, 356), (438, 448), (495, 448)], color=TRAINING_GRAY, lw=1.25)
    _text(ax, 422, 314, "trained on", size=8.6, color=TRAINING_GRAY, ha="right")
    _text(ax, 422, 338, "SELFIES-PSMILES strings", size=8.6, color=TRAINING_GRAY, ha="right")


def _target_axis(ax) -> None:
    ax.plot([455, 455], [880, 170], color=BLACK, lw=1.6)
    _arrow(ax, (455, 215), (455, 165), color=BLACK, lw=1.6)
    ax.scatter([455], [248], s=120, color=PURPLE, zorder=5)
    ax.scatter([455], [768], s=120, color=TEAL, zorder=5)
    _text(ax, 458, 86, "continuous", size=15)
    _text(ax, 458, 116, "target", size=15)
    _text(ax, 458, 150, "log₁₀ σ", size=14, family=_math_family())

    _text(ax, 576, 192, "representative", size=10.4, weight="bold", color=PURPLE)
    _text(ax, 576, 224, "top-tail", size=12.2, weight="bold", color=PURPLE)
    _text(ax, 576, 256, "target", size=12.2, weight="bold", color=PURPLE)
    _arrow(ax, (576, 302), (576, 415), color=PURPLE, lw=1.8)

    _arrow(ax, (576, 690), (576, 599), color=TEAL, lw=1.8)
    _text(ax, 576, 724, "representative", size=10.4, weight="bold", color=TEAL)
    _text(ax, 576, 756, "lower", size=12.2, weight="bold", color=TEAL)
    _text(ax, 576, 788, "target", size=12.2, weight="bold", color=TEAL)


def _transcvae(ax) -> None:
    _round_box(ax, 495, 415, 162, 184, edge=BLACK, lw=1.8, radius=12)
    _text(ax, 576, 468, "trained", size=10.2)
    _text(ax, 576, 506, "TransCVAE", size=11.0)
    _text(ax, 576, 544, "generator", size=10.8)
    _line_arrow_path(ax, [(657, 470), (690, 470), (690, 357), (725, 357)], color=PURPLE, lw=1.9)
    _line_arrow_path(ax, [(657, 545), (690, 545), (690, 764), (725, 764)], color=TEAL, lw=1.9)


def _bracket(ax, x: float, y: float, h: float, *, side: str, scale: float = 1.0, lw: float = 1.4) -> None:
    tick = 12 * scale
    if side == "left":
        ax.plot([x, x], [y, y + h], color=BLACK, lw=lw)
        ax.plot([x, x + tick], [y, y], color=BLACK, lw=lw)
        ax.plot([x, x + tick], [y + h, y + h], color=BLACK, lw=lw)
    else:
        ax.plot([x, x], [y, y + h], color=BLACK, lw=lw)
        ax.plot([x - tick, x], [y, y], color=BLACK, lw=lw)
        ax.plot([x - tick, x], [y + h, y + h], color=BLACK, lw=lw)


def _atom(ax, x: float, y: float, label: str, *, size: float = 13) -> None:
    color = O_RED if label == "O" else N_BLUE if label == "NH" else BLACK
    ax.text(
        x,
        y,
        label,
        fontsize=size,
        fontweight="bold" if label in {"O", "NH"} else "normal",
        color=color,
        ha="center",
        va="center",
        fontfamily=FONT_FAMILY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
        zorder=6,
    )


def _bond(ax, points: list[tuple[float, float]], *, lw: float = 1.8) -> None:
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=BLACK, lw=lw, solid_capstyle="round", solid_joinstyle="round")


def _carbonyl(ax, x: float, y: float, *, scale: float = 1.0) -> None:
    ax.plot([x - 4 * scale, x - 4 * scale], [y, y - 58 * scale], color=BLACK, lw=1.5 * scale)
    ax.plot([x + 4 * scale, x + 4 * scale], [y, y - 58 * scale], color=BLACK, lw=1.5 * scale)
    _atom(ax, x, y - 74 * scale, "O", size=13 * scale)


def _polymer(ax, x: float, y: float, *, scale: float = 1.0, variant: str = "top1") -> None:
    s = scale
    left_bracket = x + 50 * s
    right_bracket = x + 438 * s
    _bracket(ax, left_bracket, y - 68 * s, 132 * s, side="left", scale=s)
    _bracket(ax, right_bracket, y - 68 * s, 132 * s, side="right", scale=s)
    _text(ax, x + 16 * s, y, "[*]", size=12 * s, weight="bold")
    _text(ax, x + 474 * s, y, "[*]", size=12 * s, weight="bold")
    _text(ax, right_bracket + 22 * s, y + 72 * s, "n", size=11 * s)

    chains = {
        "top1": {
            "points": [(54, 0), (94, 24), (134, 0), (176, 24), (218, 0), (272, -30), (326, 0), (378, -30), (426, 0)],
            "atoms": [(176, 24, "O"), (382, -30, "NH")],
            "carbonyl": (134, 0),
        },
        "top2": {
            "points": [(54, 0), (94, 24), (134, 0), (176, 24), (224, 0), (280, -30), (336, 0), (386, 26), (426, 4)],
            "atoms": [(176, 24, "NH"), (386, 26, "O")],
            "carbonyl": (134, 0),
        },
        "low1": {
            "points": [(54, 0), (96, 24), (140, 0), (190, 28), (246, -4), (304, 28), (360, 0), (410, 28), (426, 12)],
            "atoms": [(140, 0, "NH"), (304, 28, "O"), (410, 28, "NH")],
            "carbonyl": None,
        },
        "low2": {
            "points": [(54, 0), (94, 24), (134, 0), (184, 34), (248, 0), (310, 34), (372, 0), (410, 24), (426, 10)],
            "atoms": [(410, 24, "O")],
            "carbonyl": (134, 0),
        },
        "sel1": {
            "points": [(54, 0), (94, 24), (134, 0), (176, 24), (218, 0), (272, -30), (326, 0), (378, -30), (426, 0)],
            "atoms": [(176, 24, "O"), (382, -30, "NH")],
            "carbonyl": (134, 0),
        },
        "sel2": {
            "points": [(54, 0), (96, 24), (140, 0), (190, 28), (246, -4), (304, 28), (360, 0), (410, 28), (426, 12)],
            "atoms": [(140, 0, "NH"), (340, 0, "O")],
            "carbonyl": None,
        },
    }
    chain = chains[variant]
    _bond(ax, [(x + px * s, y + py * s) for px, py in chain["points"]], lw=1.8 * s)
    carbonyl = chain["carbonyl"]
    if carbonyl is not None:
        _carbonyl(ax, x + carbonyl[0] * s, y + carbonyl[1] * s, scale=s)
    for px, py, label in chain["atoms"]:
        _atom(ax, x + px * s, y + py * s, label, size=(12 if label == "NH" else 13) * s)


def _pool_boxes(ax) -> None:
    _text(ax, 1008, 50, "valid-endpoint generated pool", size=MAJOR_LABEL_SIZE, weight="bold")

    _text(ax, 1008, 110, "top-tail target generated examples", size=15, color=PURPLE)
    _round_box(ax, 725, 132, 565, 384, edge=PURPLE, face="white", lw=1.5, radius=24, linestyle=(0, (3, 3)))
    _polymer(ax, 762, 236, scale=0.98, variant="top1")
    _polymer(ax, 762, 420, scale=0.98, variant="top2")

    _text(ax, 1008, 585, "lower target generated examples", size=15, color=TEAL)
    _round_box(ax, 725, 615, 565, 332, edge=TEAL, face="white", lw=1.5, radius=24, linestyle=(0, (3, 3)))
    _polymer(ax, 762, 700, scale=0.98, variant="low1")
    _polymer(ax, 762, 858, scale=0.98, variant="low2")


def _ranking_and_selected(ax) -> None:
    _round_box(ax, 1318, 440, 282, 205, edge=BLACK, lw=1.7, radius=13)
    _text(ax, 1459, 520, "PolyBERT-Ridge", size=12.2)
    _text(ax, 1459, 563, "surrogate ranking", size=12.2)

    _line_arrow_path(ax, [(1290, 357), (1459, 357), (1459, 440)], color=PURPLE, lw=2.2)
    _line_arrow_path(ax, [(1290, 764), (1459, 764), (1459, 645)], color=TEAL, lw=2.2)
    _line_arrow_path(ax, [(1600, 542), (1650, 542), (1650, 260), (1696, 260)], color=BLACK, lw=2.0)

    _text(ax, 1860, 110, "selected subset", size=MAJOR_LABEL_SIZE, weight="bold")
    _round_box(ax, 1705, 136, 310, 270, edge=BLACK, lw=1.6, radius=12)
    _polymer(ax, 1714, 226, scale=0.56, variant="sel1")
    _polymer(ax, 1714, 347, scale=0.56, variant="sel2")
    _arrow(ax, (1860, 406), (1860, 490), color=BLACK, lw=1.6)


def _load_md_box() -> Image.Image:
    if MD_BOX_SOURCE.exists():
        with Image.open(MD_BOX_SOURCE) as image:
            return image.convert("RGB")

    if not LEGACY_GA_PATH.exists():
        raise FileNotFoundError(f"Missing MD box source asset: {MD_BOX_SOURCE}")
    with Image.open(LEGACY_GA_PATH) as image:
        return image.convert("RGB").crop(MD_BOX_CROP)


def _md_panel(ax) -> None:
    md_box = _load_md_box()
    ax.imshow(md_box, extent=(1676, 2024, 505, 814), interpolation="lanczos", origin="lower", zorder=1)
    _text(ax, 1860, 858, "static cNE0 MD", size=MAJOR_LABEL_SIZE, weight="bold")
    _text(ax, 1860, 896, "reassessment", size=MAJOR_LABEL_SIZE, weight="bold")


def make_figure() -> plt.Figure:
    setup_liberation_sans()
    fig = plt.figure(figsize=(7026 / 600, 3660 / 600), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _card_stack(ax)
    _target_axis(ax)
    _transcvae(ax)
    _pool_boxes(ax)
    _ranking_and_selected(ax)
    _md_panel(ax)
    return fig


def main(display_inline: bool = False, output_name: str = OUTPUT_NAME) -> dict[str, str]:
    output_path = FIGURES_DIR / f"{output_name}.tif"
    fig = make_figure()
    save_tif_600dpi(fig, output_path, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    row = {"figure": output_name, "tif": str(output_path.relative_to(ROOT))}
    print(f"{row['figure']}: {row['tif']}")
    if display_inline:
        from IPython.display import Image as IPythonImage
        from IPython.display import display

        preview = Image.open(output_path).convert("RGB")
        preview.thumbnail((1600, 900), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        display(IPythonImage(data=buffer.getvalue()))
    return row


if __name__ == "__main__":
    main(display_inline=False)
