#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def require_path(root: Path, rel: str) -> None:
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {rel}")


def assert_no_blocked_paths(root: Path) -> None:
    blocked = ("grom" + "acs", "lam" + "mps")
    hits = []
    for path in root.rglob("*"):
        if should_skip(path):
            continue
        rel = path.relative_to(root).as_posix().lower()
        if any(term in rel for term in blocked):
            hits.append(path.relative_to(root).as_posix())
    if hits:
        preview = "\n".join(hits[:20])
        raise RuntimeError(f"Blocked path names found:\n{preview}")


def assert_file_size_limit(root: Path, limit_mb: int = 100) -> None:
    limit = limit_mb * 1024 * 1024
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        if path.stat().st_size >= limit:
            offenders.append((path.stat().st_size, path.relative_to(root).as_posix()))
    if offenders:
        preview = "\n".join(f"{size} {rel}" for size, rel in offenders[:20])
        raise RuntimeError(f"Files at or above {limit_mb} MB found:\n{preview}")


def assert_notebooks_stripped(root: Path) -> None:
    offenders = []
    for path in root.rglob("*.ipynb"):
        if should_skip(path):
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs"):
                offenders.append(f"{path.relative_to(root).as_posix()} cell {idx}: outputs")
            if cell.get("execution_count") is not None:
                offenders.append(f"{path.relative_to(root).as_posix()} cell {idx}: execution_count")
    if offenders:
        preview = "\n".join(offenders[:20])
        raise RuntimeError(f"Notebook outputs were not stripped:\n{preview}")


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def check_required_files(root: Path) -> None:
    required = [
        "LICENSE",
        "CITATION.cff",
        "CHECKPOINTS.md",
        "MY_PAPER_RELATED/MODELS/README.md",
        "MY_PAPER_RELATED/MODELS/data/simulation-trajectory-aggregate_aligned.csv",
        "MY_PAPER_RELATED/MODELS/FCD_runs/final_summary_all_models_repeated.csv",
        "MY_PAPER_RELATED/MODELS/notebooks/calculate_FCD_unified.ipynb",
        "MY_PAPER_RELATED/polybert_con/train_polybert_conductivity_4fold.py",
        "MY_PAPER_RELATED/revised/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py",
        "MY_PAPER_RELATED/selfies-psmiles/pyproject.toml",
        "vendor/psmiles_local/pyproject.toml",
        "vendor/canonicalize_psmiles-0.1.2-py3-none-any.whl",
    ]
    for rel in required:
        require_path(root, rel)

    data_rows = count_csv_rows(root / "MY_PAPER_RELATED/MODELS/data/simulation-trajectory-aggregate_aligned.csv")
    if data_rows <= 0:
        raise RuntimeError("Training data CSV has no rows")


def check_imports(root: Path) -> None:
    sys.path.insert(0, str(root / "MY_PAPER_RELATED/selfies-psmiles"))
    sys.path.insert(0, str(root / "MY_PAPER_RELATED/MODELS"))

    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import rdkit  # noqa: F401
    import selfies  # noqa: F401
    import selfies_psmiles  # noqa: F401
    import torch  # noqa: F401
    from psmiles import PolymerSmiles  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the curated reproducibility release.")
    parser.add_argument("--check-imports", action="store_true", help="Also import runtime dependencies.")
    args = parser.parse_args()

    root = repo_root()
    check_required_files(root)
    assert_no_blocked_paths(root)
    assert_file_size_limit(root)
    assert_notebooks_stripped(root)
    if args.check_imports:
        check_imports(root)

    print(f"release_root={root}")
    print("structure_ok=1")
    print("blocked_paths_ok=1")
    print("file_size_ok=1")
    print("notebooks_stripped=1")
    print(f"imports_checked={int(args.check_imports)}")


if __name__ == "__main__":
    main()
