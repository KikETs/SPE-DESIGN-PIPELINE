from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.transforms import Bbox
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
MANUSCRIPT_FIGURE_DIR = ROOT / "manuscript_figure"
SPEC_DIR = ROOT / "assets" / "manuscript_figures" / "specs"
FONT_ASSET_DIR = ROOT / "assets" / "manuscript_figures" / "fonts"

SVG_CANVAS_DPI = 100
POINTS_PER_SVG_UNIT = 72 / SVG_CANVAS_DPI
FONT_FAMILY = "Liberation Sans"
MATH_FONT_FAMILY = "Cambria Math"
MATH_FALLBACK_FONT_FAMILY = "DejaVu Math TeX Gyre"
MATH_FALLBACK_FONT_SCALE = 0.86
CAMBRIA_MATH_EXTRACTED = FONT_ASSET_DIR / "CambriaMath.ttf"


def manuscript_svg_path(slide_number: int) -> Path:
    return MANUSCRIPT_FIGURE_DIR / f"\uc2ac\ub77c\uc774\ub4dc{slide_number}.SVG"


def manuscript_spec_path(spec_name: str) -> Path:
    return SPEC_DIR / f"{spec_name}.json"


def setup_liberation_sans() -> None:
    for font_dir in (
        ROOT,
        ROOT.parent,
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        FONT_ASSET_DIR,
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ):
        if not font_dir.exists():
            continue
        for font_path in candidate_font_files(font_dir):
            if font_path.suffix.lower() == ".ttc" and "cambria" in font_path.name.lower():
                extracted = extract_cambria_math(font_path)
                if extracted is not None:
                    add_font(extracted)
            add_font(font_path)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": SVG_CANVAS_DPI,
            "savefig.dpi": 600,
        }
    )


def font_available(name: str) -> bool:
    return any(font.name == name for font in font_manager.fontManager.ttflist)


def candidate_font_files(font_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in font_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
            continue
        lower_name = path.name.lower()
        if "cambria" in lower_name or "liberation" in lower_name:
            paths.append(path)
    return sorted(paths)


def add_font(path: Path) -> None:
    try:
        font_manager.fontManager.addfont(str(path))
    except Exception:
        return


def extract_cambria_math(ttc_path: Path) -> Path | None:
    if CAMBRIA_MATH_EXTRACTED.exists():
        return CAMBRIA_MATH_EXTRACTED
    try:
        from fontTools.ttLib import TTCollection

        collection = TTCollection(str(ttc_path))
        for font in collection.fonts:
            family_names = []
            for name in font["name"].names:
                if name.nameID == 1:
                    try:
                        family_names.append(name.toUnicode())
                    except Exception:
                        continue
            if "Cambria Math" in family_names:
                FONT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
                font.save(str(CAMBRIA_MATH_EXTRACTED))
                return CAMBRIA_MATH_EXTRACTED
    except Exception:
        return None
    return None


def math_font_family() -> str:
    if font_available(MATH_FONT_FAMILY):
        return MATH_FONT_FAMILY
    return MATH_FALLBACK_FONT_FAMILY


def using_math_fallback() -> bool:
    return not font_available(MATH_FONT_FAMILY)


def content_bbox_inches(fig: plt.Figure, padding_px: int = 4, white_threshold: int = 250) -> Bbox | None:
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    image = np.asarray(fig.canvas.buffer_rgba())
    non_white = (image[:, :, 3] > 0) & np.any(image[:, :, :3] < white_threshold, axis=2)
    if not np.any(non_white):
        return None

    ys, xs = np.where(non_white)
    x0 = max(int(xs.min()) - padding_px, 0)
    x1 = min(int(xs.max()) + 1 + padding_px, width_px)
    y0 = max(int(ys.min()) - padding_px, 0)
    y1 = min(int(ys.max()) + 1 + padding_px, height_px)
    dpi = fig.dpi
    return Bbox.from_extents(x0 / dpi, (height_px - y1) / dpi, x1 / dpi, (height_px - y0) / dpi)


def save_tif_600dpi(fig: plt.Figure, tif_path: Path, *, bbox_inches=None, pad_inches: float = 0.0) -> None:
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = tif_path.with_suffix(".tmp.png")
    fig.savefig(tmp_png, dpi=600, facecolor="white", bbox_inches=bbox_inches, pad_inches=pad_inches)
    with Image.open(tmp_png) as image:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        image.save(tif_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")
    tmp_png.unlink(missing_ok=True)


def figure_row(name: str, svg_path: Path, tif_path: Path) -> dict[str, str]:
    return {
        "figure": name,
        "svg": str(svg_path.relative_to(ROOT)),
        "tif": str(tif_path.relative_to(ROOT)),
    }


def display_or_print(row: dict[str, str], display_inline: bool = False) -> dict[str, str]:
    print(f"{row['figure']}: {row['svg']} | {row['tif']}")
    if display_inline:
        from IPython.display import SVG, display

        display(SVG(filename=ROOT / row["svg"]))
    return row


def require_outputs(name: str) -> dict[str, str]:
    svg_path = FIGURES_DIR / f"{name}.svg"
    tif_path = FIGURES_DIR / f"{name}.tif"
    if not svg_path.exists() or not tif_path.exists():
        raise FileNotFoundError(f"Missing generated outputs for {name}: {svg_path}, {tif_path}")
    return figure_row(name, svg_path, tif_path)
