#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
selfies-psmiles (Unofficial)
- Upstream-compatible API (encoder/decoder/constraints/utils...)
- Plus: PSMILES '*' endpoint helpers & opt-in wrappers.

This package is intended to CO-EXIST with upstream 'selfies':
    import selfies as sf          # upstream
    import selfies_psmiles as sp  # this package
"""

from __future__ import annotations
import re

# -------- Version -------------------------------------------------------------
try:
    # 배포 메타데이터에서 버전 로드(권장)
    try:
        from importlib.metadata import version as _pkg_version  # Py>=3.8
    except Exception:
        from importlib_metadata import version as _pkg_version  # backport
    __version__ = _pkg_version("selfies-psmiles")
except Exception:
    # 개발 중엔 하드코딩 버전 사용 가능
    __version__ = "0.1.0"

# -------- Upstream-compatible public API -------------------------------------
__all__ = [
    # upstream public
    "encoder", "decoder",
    "get_preset_constraints", "get_semantic_robust_alphabet",
    "get_semantic_constraints", "set_semantic_constraints",
    "len_selfies", "split_selfies", "get_alphabet_from_selfies",
    "selfies_to_encoding", "batch_selfies_to_flat_hot",
    "encoding_to_selfies", "batch_flat_hot_to_selfies",
    "EncoderError", "DecoderError",
    # extras (PSMILES helpers/wrappers)
    "normalize_psmiles_stars", "denormalize_psmiles_stars",
    "encoder_psmiles", "decoder_psmiles",
]

# --- keep relative imports to avoid clashing with upstream package name ------
from .bond_constraints import (
    get_preset_constraints,
    get_semantic_constraints,
    get_semantic_robust_alphabet,
    set_semantic_constraints,
)
from .decoder import decoder   # your modified decoder is okay here
from .encoder import encoder   # your modified encoder is okay here
from .exceptions import DecoderError, EncoderError
from .utils.encoding_utils import (
    batch_flat_hot_to_selfies,
    batch_selfies_to_flat_hot,
    encoding_to_selfies,
    selfies_to_encoding,
)
from .utils.selfies_utils import (
    get_alphabet_from_selfies,
    len_selfies,
    split_selfies,
)

# -------- PSMILES helpers (safe no-op if you already handle '*' internally) --
# bare '*' → '[*]' 로 정규화(입력 전처리), 디코드 시 '[*]' → '*' 복원(후처리)
_BARE_STAR = re.compile(r'(?<!\[)\*(?!\])')

def normalize_psmiles_stars(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    return _BARE_STAR.sub("[*]", smiles)

def denormalize_psmiles_stars(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    return smiles.replace("[*]", "*")

# -------- Opt-in wrappers -----------------------------------------------------
def encoder_psmiles(smiles: str, *, psmiles: bool = True, **kwargs) -> str:
    """
    Wrapper: SMILES/PSMILES -> SELFIES
    - psmiles=True: '*'를 '[*]'로 정규화 후 encoder 호출
    - kwargs: upstream encoder 인자(strict 등) 그대로 전달
    """
    inp = normalize_psmiles_stars(smiles) if psmiles else smiles
    return encoder(inp, **kwargs)

def decoder_psmiles(selfies: str, *, psmiles: bool = True, **kwargs) -> str:
    """
    Wrapper: SELFIES -> (PS)SMILES
    - psmiles=True: 디코드 후 '[*]'를 '*'로 복원
    - kwargs: upstream decoder 인자(strict 등) 그대로 전달
    """
    smi = decoder(selfies, **kwargs)
    return denormalize_psmiles_stars(smi) if psmiles else smi
