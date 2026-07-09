#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figures.final_figures import export_all_figures, export_figure_by_name, figure_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/export one final journal figure.")
    parser.add_argument(
        "figure",
        nargs="?",
        help="Figure name to generate, for example Fig1A. Use --all for every final figure.",
    )
    parser.add_argument("--all", action="store_true", help="Generate every final figure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        rows = export_all_figures()
    else:
        if not args.figure:
            valid = ", ".join(figure_names())
            raise SystemExit(f"Missing figure name. Expected one of: {valid}")
        rows = [export_figure_by_name(args.figure)]

    for row in rows:
        print(f"{row['figure']}: {row['svg']} | {row['tif']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
