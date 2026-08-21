from __future__ import annotations

import contextlib
import os
from pathlib import Path

import nbformat
from IPython.display import display

from scripts.figures.final_figures import export_figure_by_name
from scripts.manuscript_figures.common import ROOT, display_or_print


PLOT_NOTEBOOK = ROOT / "plot.ipynb"
SETUP_CELLS = (0, 1, 2, 3)
FIGURE_CELLS = {
    "Fig2": 7,
    "Fig6": 4,
    "Fig9": 6,
}
SOURCE_OUTPUTS = {
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

_NOTEBOOK: nbformat.NotebookNode | None = None
_PLOT_NAMESPACE: dict[str, object] | None = None


@contextlib.contextmanager
def _repo_cwd():
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _notebook() -> nbformat.NotebookNode:
    global _NOTEBOOK
    if _NOTEBOOK is None:
        _NOTEBOOK = nbformat.read(PLOT_NOTEBOOK, as_version=4)
    return _NOTEBOOK


def _source_cell(idx: int) -> str:
    cell = _notebook().cells[idx]
    if cell.cell_type != "code":
        raise TypeError(f"{PLOT_NOTEBOOK.name} cell {idx} is {cell.cell_type}, expected code")
    return _sanitize_code(cell.source)


def _sanitize_code(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            lines.append("# Notebook-only command removed: " + line)
        else:
            lines.append(line)
    return "\n".join(lines).rstrip()


def _exec_cell(idx: int, namespace: dict[str, object]) -> None:
    code = compile(_source_cell(idx), f"{PLOT_NOTEBOOK.name}:cell-{idx}", "exec")
    exec(code, namespace)


def _setup_namespace() -> dict[str, object]:
    global _PLOT_NAMESPACE
    if _PLOT_NAMESPACE is None:
        namespace: dict[str, object] = {"__name__": "__plot_ipynb_cells__", "display": display}
        with _repo_cwd():
            for idx in SETUP_CELLS:
                _exec_cell(idx, namespace)
        _PLOT_NAMESPACE = namespace
    return _PLOT_NAMESPACE


def cleanup_source_outputs(name: str) -> None:
    for rel in SOURCE_OUTPUTS.get(name, []):
        path = ROOT / rel
        if path.exists():
            path.unlink()


def run_plot_ipynb_figure(name: str, display_inline: bool = False) -> dict[str, str]:
    if name not in FIGURE_CELLS:
        valid = ", ".join(sorted(FIGURE_CELLS))
        raise ValueError(f"Unsupported plot.ipynb figure {name!r}. Expected one of: {valid}")

    namespace = _setup_namespace()
    with _repo_cwd():
        _exec_cell(FIGURE_CELLS[name], namespace)
        row = export_figure_by_name(name)
        cleanup_source_outputs(name)
    return display_or_print(row, display_inline=display_inline)
