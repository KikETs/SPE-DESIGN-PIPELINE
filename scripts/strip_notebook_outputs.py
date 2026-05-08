#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip outputs from tracked notebooks.")
    parser.add_argument("root", nargs="?", default=".", help="Repository root or subdirectory.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    changed = 0
    for path in sorted(root.rglob("*.ipynb")):
        if ".git" in path.parts:
            continue
        changed += int(strip_notebook(path))
    print(f"notebooks_changed={changed}")


if __name__ == "__main__":
    main()
