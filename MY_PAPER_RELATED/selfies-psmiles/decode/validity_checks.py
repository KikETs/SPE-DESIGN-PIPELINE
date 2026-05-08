from __future__ import annotations

import re
from dataclasses import dataclass

from common.repr_utils import (
    canonicalize_psmiles,
    denormalize_stars,
    normalize_stars,
    selfies_roundtrip_ok,
)

try:
    import selfies_psmiles as sfp
except Exception:
    sfp = None

try:
    from rdkit import Chem
except Exception:
    Chem = None


STAR_TOKEN_RE = re.compile(r"\[\*\]")


@dataclass(frozen=True)
class ValidityResult:
    exact_two_star: bool
    syntax_valid: bool
    roundtrip_valid: bool
    canonical_match: bool | None
    graph_equivalent: bool | None
    failure_reason: str | None


def count_star_tokens(psmiles: str) -> int:
    return len(STAR_TOKEN_RE.findall(normalize_stars(psmiles)))


def _graph_equivalent_if_possible(a: str, b: str) -> bool | None:
    if Chem is None:
        return None
    try:
        ma = Chem.MolFromSmiles(denormalize_stars(a))
        mb = Chem.MolFromSmiles(denormalize_stars(b))
        if ma is None or mb is None:
            return False
        ca = Chem.MolToSmiles(ma, canonical=True)
        cb = Chem.MolToSmiles(mb, canonical=True)
        return ca == cb
    except Exception:
        return False


def evaluate_reconstruction(
    reconstructed_psmiles: str,
    *,
    target_canonical_psmiles: str | None = None,
) -> ValidityResult:
    norm_pred = normalize_stars(reconstructed_psmiles)
    two_star = count_star_tokens(norm_pred) == 2
    if not two_star:
        return ValidityResult(
            exact_two_star=False,
            syntax_valid=False,
            roundtrip_valid=False,
            canonical_match=False if target_canonical_psmiles is not None else None,
            graph_equivalent=False if target_canonical_psmiles is not None else None,
            failure_reason="not_exactly_two_[*]",
        )

    syntax_valid = False
    roundtrip_valid = False
    decoded_norm = None

    if sfp is not None:
        try:
            sf = sfp.encoder_psmiles(denormalize_stars(norm_pred), strict=False)
            back = sfp.decoder_psmiles(sf, psmiles=True)
            decoded_norm = normalize_stars(back)
            syntax_valid = True
            roundtrip_valid = True
        except Exception:
            syntax_valid = False
            roundtrip_valid = False

    canonical_match: bool | None = None
    graph_eq: bool | None = None
    if target_canonical_psmiles is not None:
        target_norm = normalize_stars(target_canonical_psmiles)
        can_pred = canonicalize_psmiles(norm_pred, prefer_psmiles_backend=True).canonical
        can_tgt = canonicalize_psmiles(target_norm, prefer_psmiles_backend=True).canonical
        canonical_match = can_pred == can_tgt
        graph_eq = _graph_equivalent_if_possible(can_pred, can_tgt)

    failure_reason = None
    if not syntax_valid:
        failure_reason = "selfies_psmiles_validation_failed"

    return ValidityResult(
        exact_two_star=two_star,
        syntax_valid=syntax_valid,
        roundtrip_valid=roundtrip_valid,
        canonical_match=canonical_match,
        graph_equivalent=graph_eq,
        failure_reason=failure_reason,
    )
