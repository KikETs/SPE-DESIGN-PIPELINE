from __future__ import annotations

from dataclasses import dataclass

from common.repr_utils import detokenize, insert_two_endpoints, normalize_pair


@dataclass(frozen=True)
class ReconstructionResult:
    pair_input: tuple[int, int]
    pair_used: tuple[int, int]
    repaired: bool
    repair_reason: str | None
    reconstructed_psmiles: str


def _repair_pair(pair: tuple[int, int], n: int) -> tuple[tuple[int, int], bool, str | None]:
    a, b = int(pair[0]), int(pair[1])
    repaired = False
    reasons: list[str] = []

    if a < 0 or a > n:
        a = min(max(a, 0), n)
        repaired = True
        reasons.append("clamp_a")
    if b < 0 or b > n:
        b = min(max(b, 0), n)
        repaired = True
        reasons.append("clamp_b")

    a, b = normalize_pair((a, b))

    if a == b:
        repaired = True
        if b < n:
            b += 1
            reasons.append("shift_right")
        elif a > 0:
            a -= 1
            reasons.append("shift_left")
        else:
            # n == 0 edge case
            b = min(1, n)
            reasons.append("degenerate_fix")
        a, b = normalize_pair((a, b))

    return (a, b), repaired, ";".join(reasons) if reasons else None


def reconstruct_with_constraints(
    base_tokens: list[str],
    pair: tuple[int, int],
    *,
    constrained: bool = True,
    endpoint_token: str = "[*]",
) -> ReconstructionResult:
    n = len(base_tokens)
    pair_in = normalize_pair(pair)

    if constrained:
        pair_used, repaired, reason = _repair_pair(pair_in, n)
    else:
        pair_used = pair_in
        repaired = False
        reason = None

    # If unconstrained pair is invalid, this can still fail and be surfaced by caller.
    tokens = insert_two_endpoints(base_tokens, pair_used, endpoint_token=endpoint_token)
    reconstructed = detokenize(tokens)

    return ReconstructionResult(
        pair_input=pair_in,
        pair_used=pair_used,
        repaired=repaired,
        repair_reason=reason,
        reconstructed_psmiles=reconstructed,
    )
