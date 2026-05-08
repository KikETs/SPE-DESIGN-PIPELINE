from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common.io_utils import read_jsonl, write_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize common failure modes from predictions.jsonl")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows = read_jsonl(args.predictions)

    counter = Counter()
    for r in rows:
        if not bool(r.get("pair_accuracy", False)):
            counter["invalid_endpoint_localization"] += 1
        if bool(r.get("pair_accuracy", False)) and bool(r.get("syntax_valid", False)) and not bool(r.get("canonical_match", False)):
            counter["reconstruction_valid_but_non_canonical"] += 1
        if not bool(r.get("syntax_valid", False)):
            counter["syntax_invalid_after_reconstruction"] += 1
        if bool(r.get("reconstruction_repaired", False)):
            counter["repair_applied"] += 1

        fr = r.get("failure_reason")
        if fr:
            counter[str(fr)] += 1

    out = {
        "num_samples": len(rows),
        "failure_mode_counts": dict(sorted(counter.items())),
        "top_failure_modes": counter.most_common(20),
    }
    write_json(args.output, out)
    print(out)


if __name__ == "__main__":
    main()
