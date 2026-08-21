#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
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


def tracked_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / value.decode() for value in result.stdout.split(b"\0") if value]


def assert_no_blocked_artifacts(root: Path, paths: list[Path]) -> None:
    blocked_suffixes = {".xtc", ".trr", ".edr", ".cpt"}
    hits = [
        path.relative_to(root).as_posix()
        for path in paths
        if path.suffix.lower() in blocked_suffixes
    ]
    if hits:
        preview = "\n".join(hits[:20])
        raise RuntimeError(f"Large/raw MD artifact types are tracked:\n{preview}")


def assert_file_size_limit(root: Path, paths: list[Path], limit_mb: int = 100) -> None:
    limit = limit_mb * 1024 * 1024
    offenders = []
    for path in paths:
        if not path.is_file():
            continue
        if path.stat().st_size >= limit:
            offenders.append((path.stat().st_size, path.relative_to(root).as_posix()))
    if offenders:
        preview = "\n".join(f"{size} {rel}" for size, rel in offenders[:20])
        raise RuntimeError(f"Files at or above {limit_mb} MB found:\n{preview}")


def assert_notebooks_valid(root: Path, paths: list[Path]) -> None:
    offenders = []
    for path in paths:
        if path.suffix.lower() != ".ipynb" or not path.is_file():
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(notebook.get("cells"), list):
                offenders.append(f"{path.relative_to(root).as_posix()}: missing cells list")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            offenders.append(f"{path.relative_to(root).as_posix()}: {error}")
    if offenders:
        preview = "\n".join(offenders[:20])
        raise RuntimeError(f"Invalid tracked notebooks:\n{preview}")


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
        "MY_PAPER_RELATED/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py",
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
    paths = tracked_paths(root)
    check_required_files(root)
    assert_no_blocked_artifacts(root, paths)
    assert_file_size_limit(root, paths)
    assert_notebooks_valid(root, paths)
    if args.check_imports:
        check_imports(root)

    print(f"release_root={root}")
    print("structure_ok=1")
    print("blocked_artifacts_ok=1")
    print("file_size_ok=1")
    print("notebooks_valid=1")
    print(f"imports_checked={int(args.check_imports)}")


if __name__ == "__main__":
    main()
