#!/usr/bin/env python
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"


@dataclass(frozen=True)
class FigureSpec:
    final_stem: str
    label: str
    source_stem: Path
    fallback_stems: tuple[Path, ...] = ()


FINAL_FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        final_stem="Fig1A",
        label="Fig1A",
        source_stem=Path("figures/fig1_ab_clean/panel_a_clean_based_final"),
        fallback_stems=(Path("figures/fig1_ab_clean/panel_a_clean_data_based_final"),),
    ),
    FigureSpec(
        final_stem="Fig1B",
        label="Fig1B",
        source_stem=Path("figures/fig1_ab_clean/panel_b_endpoint_valid_9x15_final"),
    ),
    FigureSpec(
        final_stem="Fig2",
        label="Fig2",
        source_stem=Path("figures/Figure_raw_training_label_distribution_ecdf"),
    ),
    FigureSpec(
        final_stem="Fig6",
        label="Fig6",
        source_stem=Path("figures/Figure3_generated_pool_enrichment"),
    ),
    FigureSpec(
        final_stem="Fig7",
        label="Fig7",
        source_stem=Path("figures/Fig7"),
    ),
    FigureSpec(
        final_stem="Fig8",
        label="Fig8",
        source_stem=Path("figures/Fig8"),
    ),
    FigureSpec(
        final_stem="Fig9",
        label="Fig9",
        source_stem=Path("figures/Figure_MD_reassessment_surrogate_vs_md"),
    ),
)


VECTOR_EXTENSIONS = (".svg",)
RASTER_EXTENSIONS = (".tif", ".tiff", ".png", ".jpeg", ".jpg")
STALE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".pdf")


def _existing_source(spec: FigureSpec, extensions: tuple[str, ...]) -> Path:
    stems = (spec.source_stem, *spec.fallback_stems, Path(f"figures/{spec.final_stem}"))
    for stem in stems:
        for ext in extensions:
            path = ROOT / f"{stem}{ext}"
            if path.exists():
                return path
    tried = ", ".join(str(ROOT / f"{stem}{ext}") for stem in stems for ext in extensions)
    raise FileNotFoundError(f"Missing source for {spec.final_stem}. Tried: {tried}")


def _save_tif_600dpi(source: Path, dest: Path) -> None:
    with Image.open(source) as image:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        image.save(dest, format="TIFF", dpi=(600, 600), compression="tiff_lzw")


def export_figure(spec: FigureSpec, figures_dir: Path = FIGURES_DIR) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in STALE_EXTENSIONS:
        stale = figures_dir / f"{spec.final_stem}{ext}"
        if stale.exists():
            stale.unlink()

    svg_source = _existing_source(spec, VECTOR_EXTENSIONS)
    tif_source = _existing_source(spec, RASTER_EXTENSIONS)

    svg_dest = figures_dir / f"{spec.final_stem}.svg"
    tif_dest = figures_dir / f"{spec.final_stem}.tif"

    if svg_source.resolve() != svg_dest.resolve():
        shutil.copy2(svg_source, svg_dest)
    if tif_source.resolve() != tif_dest.resolve():
        _save_tif_600dpi(tif_source, tif_dest)

    return {
        "figure": spec.final_stem,
        "label": spec.label,
        "svg": str(svg_dest.relative_to(ROOT)),
        "tif": str(tif_dest.relative_to(ROOT)),
        "svg_source": str(svg_source.relative_to(ROOT)),
        "tif_source": str(tif_source.relative_to(ROOT)),
    }


def export_all_figures(figures_dir: Path = FIGURES_DIR) -> list[dict[str, str]]:
    return [export_figure(spec, figures_dir=figures_dir) for spec in FINAL_FIGURES]


def figure_names() -> list[str]:
    return [spec.final_stem for spec in FINAL_FIGURES]


def figure_spec(name: str) -> FigureSpec:
    normalized = name.strip().lower()
    for spec in FINAL_FIGURES:
        if spec.final_stem.lower() == normalized or spec.label.lower() == normalized:
            return spec
    valid = ", ".join(figure_names())
    raise ValueError(f"Unknown figure {name!r}. Expected one of: {valid}")


def export_figure_by_name(name: str, figures_dir: Path = FIGURES_DIR) -> dict[str, str]:
    return export_figure(figure_spec(name), figures_dir=figures_dir)


def final_svg_paths(figures_dir: Path = FIGURES_DIR) -> list[Path]:
    return [figures_dir / f"{spec.final_stem}.svg" for spec in FINAL_FIGURES]


def final_tif_paths(figures_dir: Path = FIGURES_DIR) -> list[Path]:
    return [figures_dir / f"{spec.final_stem}.tif" for spec in FINAL_FIGURES]


def main() -> int:
    rows = export_all_figures()
    for row in rows:
        print(f"{row['figure']}: {row['svg']} | {row['tif']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
