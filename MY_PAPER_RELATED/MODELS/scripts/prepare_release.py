#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def strip_notebook_outputs(nb_dir: Path) -> int:
    count = 0
    for p in sorted(nb_dir.glob('*.ipynb')):
        nb = json.loads(p.read_text(encoding='utf-8'))
        changed = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            if cell.get('outputs'):
                cell['outputs'] = []
                changed = True
            if cell.get('execution_count') is not None:
                cell['execution_count'] = None
                changed = True
        if changed:
            p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
            count += 1
    return count


def remove_pycache(root: Path) -> int:
    removed = 0
    for d in root.rglob('__pycache__'):
        if d.is_dir():
            shutil.rmtree(d)
            removed += 1
    return removed


def purge_dir_contents(path: Path, keep_dotfiles: bool = True) -> int:
    removed = 0
    if not path.exists():
        return removed
    for item in path.iterdir():
        if keep_dotfiles and item.name.startswith('.'):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)
        removed += 1
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare MODELS directory for GitHub release.')
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--purge-artifacts', action='store_true', help='Also clear checkpoints/ and FCD_runs/ contents.')
    args = ap.parse_args()

    root = args.root.resolve()
    notebooks = root / 'notebooks'
    checkpoints = root / 'checkpoints'
    fcd_runs = root / 'FCD_runs'

    stripped = strip_notebook_outputs(notebooks)
    pyc = remove_pycache(root)

    purged_ckpt = 0
    purged_fcd = 0
    if args.purge_artifacts:
        purged_ckpt = purge_dir_contents(checkpoints, keep_dotfiles=True)
        purged_fcd = purge_dir_contents(fcd_runs, keep_dotfiles=True)

    print(f'root={root}')
    print(f'notebooks_stripped={stripped}')
    print(f'pycache_removed={pyc}')
    print(f'checkpoints_purged={purged_ckpt}')
    print(f'fcd_runs_purged={purged_fcd}')


if __name__ == '__main__':
    main()
