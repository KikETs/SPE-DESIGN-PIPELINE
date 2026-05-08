from __future__ import annotations

from dataclasses import dataclass

from common.repr_utils import selfies_roundtrip_ok
from data.canonical_labeler import CanonicalLabel


@dataclass(frozen=True)
class FilterResult:
    keep: bool
    reason: str | None


# NOTE:
# Linear homopolymer filtering is implemented as auditable heuristics in v1.
# If chemistry backends are available, this can be replaced with stricter graph checks.
def apply_scope_filters(
    label: CanonicalLabel,
    *,
    max_token_length: int = 512,
    require_selfies_validity: bool = True,
) -> FilterResult:
    if "." in label.canonical_target_psmiles:
        return FilterResult(False, "disconnected_components")

    if len(label.base_tokens) < 2:
        return FilterResult(False, "base_too_short")

    if len(label.base_tokens) > int(max_token_length):
        return FilterResult(False, "sequence_too_long")

    a, b = label.endpoint_pair
    if a == b:
        return FilterResult(False, "endpoint_pair_duplicate")

    if require_selfies_validity:
        ok, _, _ = selfies_roundtrip_ok(label.canonical_target_psmiles)
        if not ok:
            return FilterResult(False, "selfies_roundtrip_invalid")

    return FilterResult(True, None)
