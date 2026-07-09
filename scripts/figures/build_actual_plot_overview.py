#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[2]
SOURCE_NOTEBOOK = ROOT / "plot.ipynb"
OUT_NOTEBOOK = ROOT / "final_figures_overview.ipynb"


SETUP_CELLS = (0, 1, 2, 3)
FIGURE_CELLS = {
    "Fig1A": 8,
    "Fig1B": 9,
    "Fig2": 7,
    "Fig6": 4,
    "Fig9": 6,
}
FIGURE_SCRIPT_CELLS = {
    "Fig7": (
        "from scripts.figures.make_fig7_surrogate_reliability import main as make_fig7\n"
        "make_fig7()\n"
        "display_generated_svg(\"Fig7\")"
    ),
    "Fig8": (
        "from scripts.figures.make_fig8_reference_stratified_static_cne0 import main as make_fig8\n"
        "make_fig8()\n"
        "display_generated_svg(\"Fig8\")"
    ),
}
FIGURE_ORDER = ("Fig1A", "Fig1B", "Fig2", "Fig6", "Fig7", "Fig8", "Fig9")


PREVIEW_HELPER = r'''from pathlib import Path
from IPython.display import SVG, display

ROOT = Path.cwd().resolve()

from scripts.figures.final_figures import export_figure_by_name


SOURCE_OUTPUTS = {
    "Fig1A": [
        "figures/fig1_ab_clean/panel_a_clean_data_based_final.png",
        "figures/fig1_ab_clean/panel_a_clean_data_based_final.svg",
    ],
    "Fig1B": [
        "figures/fig1_ab_clean/panel_b_endpoint_valid_9x15_final.png",
        "figures/fig1_ab_clean/panel_b_endpoint_valid_9x15_final.svg",
    ],
    "Fig2": [
        "figures/Figure_raw_training_label_distribution_ecdf.jpeg",
        "figures/Figure_raw_training_label_distribution_ecdf.pdf",
        "figures/Figure_raw_training_label_distribution_ecdf.svg",
        "figures/Figure_raw_training_label_distribution_ecdf.tif",
    ],
    "Fig6": [
        "figures/Figure3_generated_pool_enrichment.jpeg",
        "figures/Figure3_generated_pool_enrichment.pdf",
        "figures/Figure3_generated_pool_enrichment.svg",
        "figures/Figure3_generated_pool_enrichment.tif",
    ],
    "Fig9": [
        "figures/Figure_MD_reassessment_surrogate_vs_md.jpeg",
        "figures/Figure_MD_reassessment_surrogate_vs_md.pdf",
        "figures/Figure_MD_reassessment_surrogate_vs_md.svg",
        "figures/Figure_MD_reassessment_surrogate_vs_md.tif",
    ],
}


def cleanup_source_outputs(name: str) -> None:
    for rel in SOURCE_OUTPUTS.get(name, []):
        path = ROOT / rel
        if path.exists():
            path.unlink()
    fig1_dir = ROOT / "figures" / "fig1_ab_clean"
    if fig1_dir.exists() and not any(fig1_dir.iterdir()):
        fig1_dir.rmdir()


def export_final_outputs_and_display_svg(name: str) -> None:
    row = export_figure_by_name(name)
    cleanup_source_outputs(name)
    print(f"{row['figure']}: {row['svg']} | {row['tif']}")
    display(SVG(filename=ROOT / row["svg"]))


def display_generated_svg(name: str) -> None:
    svg = ROOT / "figures" / f"{name}.svg"
    tif = ROOT / "figures" / f"{name}.tif"
    if not svg.exists() or not tif.exists():
        raise FileNotFoundError(f"Missing generated figure outputs for {name}: {svg}, {tif}")
    print(f"{name}: {svg.relative_to(ROOT)} | {tif.relative_to(ROOT)}")
    display(SVG(filename=svg))
'''


def source_cell(nb: nbformat.NotebookNode, idx: int) -> str:
    cell = nb.cells[idx]
    if cell.cell_type != "code":
        raise TypeError(f"plot.ipynb cell {idx} is {cell.cell_type}, expected code.")
    return cell.source


def copied_code_cell(nb: nbformat.NotebookNode, idx: int) -> nbformat.NotebookNode:
    cell = new_code_cell(source=source_cell(nb, idx))
    cell.metadata["source_notebook"] = "plot.ipynb"
    cell.metadata["source_cell"] = idx
    return cell


def build_notebook() -> nbformat.NotebookNode:
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)

    cells: list[nbformat.NotebookNode] = [
        new_markdown_cell(
            "# Final Figures Actual Plotting Notebook\n\n"
            "This notebook executes the actual plotting code copied from `plot.ipynb`. "
            "Display cells only export the newly generated source files to final `Fig*.svg`/`Fig*.tif` names "
            "and render the generated SVG in the notebook."
        ),
        new_code_cell(PREVIEW_HELPER),
        new_markdown_cell("## Shared Setup From `plot.ipynb`"),
    ]

    for idx in SETUP_CELLS:
        cells.append(new_markdown_cell(f"### `plot.ipynb` Cell {idx}"))
        cells.append(copied_code_cell(source, idx))

    for name in FIGURE_ORDER:
        cells.append(new_markdown_cell(f"## {name}"))
        if name in FIGURE_CELLS:
            idx = FIGURE_CELLS[name]
            cells.append(new_markdown_cell(f"### Actual plotting code from `plot.ipynb` Cell {idx}"))
            cells.append(copied_code_cell(source, idx))
            cells.append(new_code_cell(f'export_final_outputs_and_display_svg("{name}")'))
        elif name in FIGURE_SCRIPT_CELLS:
            cells.append(new_markdown_cell("### Actual plotting code from `scripts/figures`"))
            cells.append(new_code_cell(FIGURE_SCRIPT_CELLS[name]))
        else:
            cells.append(
                new_markdown_cell(
                    "No corresponding plotting cell for this figure exists in the current `plot.ipynb` "
                    "or script tree. This notebook does not reconstruct it; it displays the generated "
                    "Matplotlib SVG/TIF output only."
                )
            )
            cells.append(new_code_cell(f'display_generated_svg("{name}")'))

    out = new_notebook(cells=cells)
    out.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    return out


def main() -> int:
    nb = build_notebook()
    nbformat.write(nb, OUT_NOTEBOOK)
    print(f"Wrote {OUT_NOTEBOOK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
