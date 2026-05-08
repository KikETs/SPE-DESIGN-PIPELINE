from __future__ import annotations

from collections import Counter
from statistics import mean, median

from common.repr_utils import stable_hash_int


def deterministic_split(
    rows: list[dict],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= valid_ratio < 1.0):
        raise ValueError("valid_ratio must be in [0,1)")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")

    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []

    for row in rows:
        key = row.get("canonical_target_psmiles") or row.get("canonical_psmiles") or row.get("sample_id")
        h = stable_hash_int(str(key), seed=seed)
        u = (h % 1_000_000) / 1_000_000.0
        if u < train_ratio:
            train.append(row)
        elif u < train_ratio + valid_ratio:
            valid.append(row)
        else:
            test.append(row)
    return train, valid, test


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "length": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
            "endpoint_distance": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
            "canonical_backend": {},
        }

    lens = [int(r.get("base_token_len", len(r.get("base_tokens", [])))) for r in rows]
    dists = [abs(int(r["endpoint_pair"][1]) - int(r["endpoint_pair"][0])) for r in rows]
    backend = Counter(str(r.get("canonical_backend", "unknown")) for r in rows)

    return {
        "count": len(rows),
        "length": {
            "min": min(lens),
            "max": max(lens),
            "mean": float(mean(lens)),
            "median": float(median(lens)),
        },
        "endpoint_distance": {
            "min": min(dists),
            "max": max(dists),
            "mean": float(mean(dists)),
            "median": float(median(dists)),
        },
        "canonical_backend": dict(sorted(backend.items())),
    }
