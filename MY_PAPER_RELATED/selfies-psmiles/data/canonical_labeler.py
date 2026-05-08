from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from common.repr_utils import (
    all_pairs,
    canonicalize_psmiles,
    choose_best_pair_by_reconstruction,
    detokenize,
    endpoint_insertion_positions,
    insert_two_endpoints,
    normalize_pair,
    remove_endpoint_tokens,
    tokenize_psmiles,
)


@dataclass(frozen=True)
class CanonicalLabel:
    sample_id: str
    raw_psmiles: str
    canonical_psmiles: str
    canonical_backend: str
    base_tokens: list[str]
    base_psmiles: str
    endpoint_pair: tuple[int, int]
    endpoint_candidates: list[int]
    canonical_target_psmiles: str
    ambiguity_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["endpoint_pair"] = [int(self.endpoint_pair[0]), int(self.endpoint_pair[1])]
        return out


def _validate_pair(base_tokens: list[str], pair: tuple[int, int]) -> tuple[bool, str | None]:
    a, b = normalize_pair(pair)
    n = len(base_tokens)
    if a < 0 or b < 0 or a > n or b > n:
        return False, "endpoint_pair_out_of_bounds"
    if a == b:
        return False, "endpoint_pair_duplicate"
    return True, None


def generate_canonical_label(
    sample_id: str,
    raw_psmiles: str,
    *,
    allow_multi_endpoint_resolve: bool = False,
    prefer_psmiles_backend: bool = True,
) -> tuple[CanonicalLabel | None, str | None]:
    """
    Deterministic canonical endpoint label generation.

    Returns:
      (label, None) on success
      (None, reason) on exclusion
    """
    raw = str(raw_psmiles or "").strip()
    if not raw:
        return None, "empty_input"

    cano = canonicalize_psmiles(raw, prefer_psmiles_backend=prefer_psmiles_backend)
    canonical = cano.canonical
    if not canonical:
        return None, "empty_after_canonicalization"

    tokens = tokenize_psmiles(canonical)
    if not tokens:
        return None, "tokenization_failed"

    endpoint_candidates = endpoint_insertion_positions(tokens)
    endpoint_count = len(endpoint_candidates)

    if endpoint_count < 2:
        return None, "endpoint_count_lt_2"

    base_tokens = remove_endpoint_tokens(tokens)
    if not base_tokens:
        return None, "empty_base_after_endpoint_removal"

    ambiguity_note: str | None = None

    if endpoint_count == 2:
        pair = normalize_pair((endpoint_candidates[0], endpoint_candidates[1]))
    else:
        if not allow_multi_endpoint_resolve:
            return None, "multi_endpoint_topology"
        pair_candidates = all_pairs(endpoint_candidates)
        if not pair_candidates:
            return None, "no_candidate_endpoint_pair"
        pair = choose_best_pair_by_reconstruction(base_tokens, pair_candidates)
        ambiguity_note = (
            f"resolved_from_{endpoint_count}_candidates_with_deterministic_ranking"
        )

    ok, reason = _validate_pair(base_tokens, pair)
    if not ok:
        return None, reason

    try:
        canonical_target_tokens = insert_two_endpoints(base_tokens, pair, endpoint_token="[*]")
        canonical_target_psmiles = detokenize(canonical_target_tokens)
    except Exception:
        return None, "reconstruction_failed"

    label = CanonicalLabel(
        sample_id=str(sample_id),
        raw_psmiles=raw,
        canonical_psmiles=canonical,
        canonical_backend=cano.backend,
        base_tokens=base_tokens,
        base_psmiles=detokenize(base_tokens),
        endpoint_pair=pair,
        endpoint_candidates=sorted(endpoint_candidates),
        canonical_target_psmiles=canonical_target_psmiles,
        ambiguity_note=ambiguity_note,
    )
    return label, None
