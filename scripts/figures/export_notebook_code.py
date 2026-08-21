#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "scripts/figures/notebook_exports"

NOTEBOOKS = {
    "plot.ipynb": OUTDIR / "plot_pipeline_from_notebook.py",
}


def _sanitize_code(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            lines.append("# Notebook-only command removed: " + line)
        else:
            lines.append(line)
    return "\n".join(lines).rstrip()


def export_notebook(notebook_path: Path, out_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    chunks = [
        "#!/usr/bin/env python",
        "# Generated from " + notebook_path.name + ".",
        "# Keep figure logic in scripts; use final_figures_overview.ipynb for visual review.",
        "from __future__ import annotations",
        "",
    ]
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code = _sanitize_code(cell.get("source", ""))
        if not code.strip():
            continue
        chunks.append(f"# %% [cell {idx}]")
        chunks.append(code)
        chunks.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    for notebook, out_path in NOTEBOOKS.items():
        notebook_path = ROOT / notebook
        if not notebook_path.exists():
            raise FileNotFoundError(notebook_path)
        export_notebook(notebook_path, out_path)
        print(f"{notebook_path.relative_to(ROOT)} -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
