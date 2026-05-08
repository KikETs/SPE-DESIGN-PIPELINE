from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from typing import Iterable

try:
    from psmiles import PolymerSmiles as PS
except Exception:
    PS = None

try:
    import selfies_psmiles as sfp
except Exception:
    sfp = None


BARE_STAR_RE = re.compile(r"(?<!\[)\*(?!\])")
WRAP_BRACES_RE = re.compile(r"^\{\s*(.+?)\s*\}\s*(\d+(?:-\d+)?)?$")
WRAP_PARENS_RE = re.compile(r"^\(\s*(.+?)\s*\)\s*(\d+(?:-\d+)?)?$")

# Keep bracketed symbols and common two-char atoms intact.
PSMILES_TOKEN_RE = re.compile(
    r"\[[^\[\]]+\]|Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|Fe|Zn|Cu|Mn|Hg|Ag|Sn|Pb|As|%[0-9]{2}|."
)


@dataclass(frozen=True)
class CanonicalizationResult:
    canonical: str
    backend: str


def normalize_stars(text: str | None) -> str:
    if text is None:
        return ""
    return BARE_STAR_RE.sub("[*]", str(text).strip())


def denormalize_stars(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).replace("[*]", "*")


def strip_repeat_wrapper(text: str | None) -> str:
    s = str(text or "").strip()
    changed = True
    while changed and s:
        changed = False
        m1 = WRAP_BRACES_RE.match(s)
        if m1:
            s = m1.group(1).strip()
            changed = True
            continue
        m2 = WRAP_PARENS_RE.match(s)
        if m2:
            s = m2.group(1).strip()
            changed = True
            continue
    return s


def tokenize_psmiles(psmiles: str | None) -> list[str]:
    s = normalize_stars(strip_repeat_wrapper(psmiles))
    if not s:
        return []
    return PSMILES_TOKEN_RE.findall(s)


def detokenize(tokens: Iterable[str]) -> str:
    return "".join(tokens)


def is_endpoint_token(tok: str) -> bool:
    return tok in {"*", "[*]"}


def count_endpoint_tokens(tokens: Iterable[str]) -> int:
    return sum(1 for t in tokens if is_endpoint_token(t))


def endpoint_insertion_positions(tokens: list[str]) -> list[int]:
    """Map endpoint-token indices to insertion positions in endpoint-removed tokens."""
    positions: list[int] = []
    non_endpoint_prefix = 0
    for tok in tokens:
        if is_endpoint_token(tok):
            positions.append(non_endpoint_prefix)
        else:
            non_endpoint_prefix += 1
    return positions


def remove_endpoint_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if not is_endpoint_token(t)]


def canonicalize_psmiles(raw: str, prefer_psmiles_backend: bool = True) -> CanonicalizationResult:
    base = normalize_stars(strip_repeat_wrapper(raw))
    if not base:
        return CanonicalizationResult(canonical="", backend="empty")

    # 1) Prefer psmiles canonicalizer if available.
    if prefer_psmiles_backend and PS is not None:
        try:
            canonical = PS(denormalize_stars(base)).canonicalize.psmiles
            return CanonicalizationResult(
                canonical=normalize_stars(strip_repeat_wrapper(canonical)),
                backend="psmiles",
            )
        except Exception:
            pass

    # 2) Fallback to SELFIES-PSMILES round-trip canonicalization.
    if sfp is not None:
        try:
            sf = sfp.encoder_psmiles(denormalize_stars(base), strict=False)
            canonical = sfp.decoder_psmiles(sf, psmiles=True)
            return CanonicalizationResult(
                canonical=normalize_stars(strip_repeat_wrapper(canonical)),
                backend="selfies_psmiles",
            )
        except Exception:
            pass

    return CanonicalizationResult(canonical=base, backend="raw")


def normalize_pair(pair: tuple[int, int] | list[int]) -> tuple[int, int]:
    a, b = int(pair[0]), int(pair[1])
    return (a, b) if a <= b else (b, a)


def insert_two_endpoints(base_tokens: list[str], pair: tuple[int, int], endpoint_token: str = "[*]") -> list[str]:
    a, b = normalize_pair(pair)
    n = len(base_tokens)
    if a < 0 or b < 0 or a > n or b > n:
        raise ValueError(f"invalid insertion pair {pair} for length {n}")
    if a == b:
        raise ValueError(f"duplicate insertion pair {pair}")

    out: list[str] = []
    for idx in range(n + 1):
        if idx == a:
            out.append(endpoint_token)
        if idx < n:
            out.append(base_tokens[idx])
        if idx == b:
            out.append(endpoint_token)
    return out


def stable_hash_int(text: str, seed: int = 0) -> int:
    blob = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest(), 16)


def build_variant_strings(canonical_psmiles: str) -> list[str]:
    """
    Deterministic equivalent-string variants used for consistency checks.
    """
    s = normalize_stars(strip_repeat_wrapper(canonical_psmiles))
    variants = {
        s,
        denormalize_stars(s),
        "{" + s + "}1",
        "(" + s + ")1",
        "{" + denormalize_stars(s) + "}1",
        "(" + denormalize_stars(s) + ")1",
    }
    return sorted(v for v in variants if v)


def choose_best_pair_by_reconstruction(
    base_tokens: list[str],
    candidate_pairs: Iterable[tuple[int, int]],
) -> tuple[int, int]:
    """
    Deterministic ranking:
    1) shorter reconstructed length
    2) lexicographically smaller reconstructed string
    3) smaller insertion pair
    """
    best_pair: tuple[int, int] | None = None
    best_score: tuple[int, str, tuple[int, int]] | None = None

    for pair in candidate_pairs:
        pair = normalize_pair(pair)
        if pair[0] == pair[1]:
            continue
        try:
            recon = detokenize(insert_two_endpoints(base_tokens, pair, endpoint_token="[*]"))
        except Exception:
            continue
        score = (len(recon), recon, pair)
        if best_score is None or score < best_score:
            best_score = score
            best_pair = pair

    if best_pair is None:
        raise ValueError("no valid candidate pair")
    return best_pair


def all_pairs(values: list[int]) -> list[tuple[int, int]]:
    return [normalize_pair(p) for p in itertools.combinations(values, 2)]


def selfies_roundtrip_ok(psmiles: str) -> tuple[bool, str | None, str | None]:
    """
    Returns (ok, selfies_str, decoded_psmiles_norm)
    """
    if sfp is None:
        return False, None, None
    try:
        sf = sfp.encoder_psmiles(denormalize_stars(psmiles), strict=False)
        decoded = sfp.decoder_psmiles(sf, psmiles=True)
        return True, sf, normalize_stars(decoded)
    except Exception:
        return False, None, None
