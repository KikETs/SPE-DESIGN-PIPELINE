from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import pandas as pd

from common.io_utils import ensure_dir, write_json, write_jsonl
from data.canonical_labeler import generate_canonical_label
from data.filters import apply_scope_filters
from data.split_dataset import deterministic_split, summarize_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical endpoint prediction dataset from PI1M"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("../../PI1M/PI1M_v2.csv"),
        help="PI1M CSV path (PI1M.csv or PI1M_v2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pi1m_endpoint_dataset"),
    )
    parser.add_argument("--smiles-col", type=str, default="SMILES")
    parser.add_argument("--sa-col", type=str, default="SA Score")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-token-length", type=int, default=512)
    parser.add_argument(
        "--allow-multi-endpoint-resolve",
        action="store_true",
        help="Resolve >2 endpoint candidates deterministically instead of exclusion.",
    )
    parser.add_argument(
        "--skip-selfies-validity",
        action="store_true",
        help="Disable SELFIES-PSMILES round-trip validity check.",
    )
    parser.add_argument(
        "--save-excluded",
        action="store_true",
        help="Save excluded sample metadata to excluded_samples.jsonl.",
    )
    parser.add_argument(
        "--max-excluded-log",
        type=int,
        default=200_000,
        help="Maximum number of excluded rows to store when --save-excluded is set.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _to_record(label_dict: dict, source_idx: int, sa_score: float | None) -> dict:
    pair = label_dict["endpoint_pair"]
    record = {
        "sample_id": label_dict["sample_id"],
        "source_index": int(source_idx),
        "source_dataset": "PI1M",
        "raw_psmiles": label_dict["raw_psmiles"],
        "canonical_psmiles": label_dict["canonical_psmiles"],
        "canonical_target_psmiles": label_dict["canonical_target_psmiles"],
        "base_psmiles": label_dict["base_psmiles"],
        "base_tokens": label_dict["base_tokens"],
        "base_token_len": int(len(label_dict["base_tokens"])),
        "endpoint_pair": [int(pair[0]), int(pair[1])],
        "endpoint_candidates": [int(x) for x in label_dict["endpoint_candidates"]],
        "endpoint_distance": int(abs(pair[1] - pair[0])),
        "canonical_backend": label_dict["canonical_backend"],
        "ambiguity_note": label_dict.get("ambiguity_note"),
        "sa_score": sa_score,
    }
    return record


def main() -> None:
    args = _parse_args()
    if args.debug and args.max_rows is None:
        args.max_rows = 20_000

    output_dir = ensure_dir(args.output_dir)

    header = pd.read_csv(args.input_csv, nrows=0)
    columns = set(header.columns)
    if args.smiles_col not in columns:
        raise ValueError(f"missing smiles column '{args.smiles_col}' in {args.input_csv}")

    has_sa = args.sa_col in columns
    usecols = [args.smiles_col] + ([args.sa_col] if has_sa else [])

    total_raw = 0
    usable = 0
    excluded = 0
    excluded_reason = Counter()
    backend_counter = Counter()
    duplicate_removed = 0

    records: list[dict] = []
    excluded_rows: list[dict] = []

    seen_keys: set[str] = set()

    reader = pd.read_csv(args.input_csv, usecols=usecols, chunksize=args.chunk_size)
    for chunk in reader:
        for idx, row in chunk.iterrows():
            if args.max_rows is not None and total_raw >= int(args.max_rows):
                break

            total_raw += 1
            raw_psmiles = str(row[args.smiles_col]).strip()
            sample_id = f"pi1m_{idx}"

            label, reason = generate_canonical_label(
                sample_id=sample_id,
                raw_psmiles=raw_psmiles,
                allow_multi_endpoint_resolve=bool(args.allow_multi_endpoint_resolve),
                prefer_psmiles_backend=True,
            )

            if label is None:
                excluded += 1
                excluded_reason[reason or "unknown"] += 1
                if args.save_excluded and len(excluded_rows) < args.max_excluded_log:
                    excluded_rows.append(
                        {
                            "sample_id": sample_id,
                            "source_index": int(idx),
                            "raw_psmiles": raw_psmiles,
                            "reason": reason or "unknown",
                        }
                    )
                continue

            f = apply_scope_filters(
                label,
                max_token_length=int(args.max_token_length),
                require_selfies_validity=not bool(args.skip_selfies_validity),
            )
            if not f.keep:
                excluded += 1
                excluded_reason[f.reason or "unknown"] += 1
                if args.save_excluded and len(excluded_rows) < args.max_excluded_log:
                    excluded_rows.append(
                        {
                            "sample_id": sample_id,
                            "source_index": int(idx),
                            "raw_psmiles": raw_psmiles,
                            "canonical_psmiles": label.canonical_psmiles,
                            "canonical_target_psmiles": label.canonical_target_psmiles,
                            "reason": f.reason or "unknown",
                        }
                    )
                continue

            label_dict = label.to_dict()
            dedup_key = label_dict["canonical_target_psmiles"]
            if dedup_key in seen_keys:
                excluded += 1
                duplicate_removed += 1
                excluded_reason["duplicate_canonical_target"] += 1
                if args.save_excluded and len(excluded_rows) < args.max_excluded_log:
                    excluded_rows.append(
                        {
                            "sample_id": sample_id,
                            "source_index": int(idx),
                            "raw_psmiles": raw_psmiles,
                            "reason": "duplicate_canonical_target",
                        }
                    )
                continue
            seen_keys.add(dedup_key)

            sa_score = None
            if has_sa:
                try:
                    sa_score = float(row[args.sa_col])
                except Exception:
                    sa_score = None

            rec = _to_record(label_dict, source_idx=int(idx), sa_score=sa_score)
            records.append(rec)
            usable += 1
            backend_counter[rec["canonical_backend"]] += 1

        if args.max_rows is not None and total_raw >= int(args.max_rows):
            break

    train_rows, valid_rows, test_rows = deterministic_split(
        records,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        valid_ratio=float(args.valid_ratio),
    )

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    if args.save_excluded:
        write_jsonl(output_dir / "excluded_samples.jsonl", excluded_rows)

    sa_values = [r["sa_score"] for r in records if r.get("sa_score") is not None]

    summary = {
        "input_csv": str(args.input_csv.resolve()),
        "config": {
            "smiles_col": args.smiles_col,
            "sa_col": args.sa_col,
            "chunk_size": int(args.chunk_size),
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "valid_ratio": float(args.valid_ratio),
            "max_token_length": int(args.max_token_length),
            "allow_multi_endpoint_resolve": bool(args.allow_multi_endpoint_resolve),
            "skip_selfies_validity": bool(args.skip_selfies_validity),
            "save_excluded": bool(args.save_excluded),
        },
        "scope": {
            "target": "linear_homopolymer_repeat_units_two_endpoints_v1",
            "filtering_strategy": "heuristic_rule_based_with_audit_logs",
        },
        "counts": {
            "total_raw_samples": int(total_raw),
            "usable_samples": int(usable),
            "excluded_samples": int(excluded),
            "duplicate_removed": int(duplicate_removed),
            "split": {
                "train": int(len(train_rows)),
                "valid": int(len(valid_rows)),
                "test": int(len(test_rows)),
            },
        },
        "excluded_by_reason": dict(sorted(excluded_reason.items())),
        "canonical_backend_distribution": dict(sorted(backend_counter.items())),
        "dataset_stats": summarize_rows(records),
        "split_stats": {
            "train": summarize_rows(train_rows),
            "valid": summarize_rows(valid_rows),
            "test": summarize_rows(test_rows),
        },
        "label_distribution": {
            "mean_endpoint_distance": float(mean([r["endpoint_distance"] for r in records])) if records else 0.0,
            "endpoint_distance_histogram_top": dict(
                Counter(int(r["endpoint_distance"]) for r in records).most_common(20)
            ),
        },
        "sa_score_stats": {
            "available": bool(has_sa),
            "num_with_sa": int(len(sa_values)),
            "num_without_sa": int(len(records) - len(sa_values)),
            "mean": float(mean(sa_values)) if sa_values else None,
            "min": float(min(sa_values)) if sa_values else None,
            "max": float(max(sa_values)) if sa_values else None,
        },
        "excluded_log_saved": bool(args.save_excluded),
        "excluded_log_count": int(len(excluded_rows)),
        "excluded_log_truncated": bool(args.save_excluded and excluded > len(excluded_rows)),
    }

    write_json(output_dir / "dataset_summary.json", summary)
    print(f"[done] wrote dataset artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
