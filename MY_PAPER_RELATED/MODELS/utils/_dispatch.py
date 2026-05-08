from __future__ import annotations

import os
from importlib import import_module

_ALIAS = {}

_SUPPORTED = {
    "Encoder_Only",
    "Encoder_Only_PSMILES",
    "LSTM_CVAE",
    "LSTM_CVAE_PSMILES",
    "TransCVAE",
    "TransCVAE_PSMILES",
}

_DEFAULT = "LSTM_CVAE"


def current_variant() -> str:
    raw = os.environ.get("MODELS_VARIANT", _DEFAULT).strip() or _DEFAULT
    variant = _ALIAS.get(raw, raw)
    if variant not in _SUPPORTED:
        variant = _DEFAULT
    return variant


def _module_name(base: str) -> str:
    variant = current_variant()

    if base == "pi1m_pretrain" and variant.startswith("TransCVAE"):
        # PI1M pretrain is only used for Decoder/LSTM notebooks.
        variant = "LSTM_CVAE"

    if variant == "Encoder_Only_PSMILES" and base in {"eval", "generate", "Trans_util", "LSTM_util"}:
        # Reuse Encoder_Only logic modules; only tokenization/pretrain differ.
        variant = "Encoder_Only"

    if base == "LSTM_util" and variant.startswith("TransCVAE"):
        # No Trans-specific LSTM_util module exists.
        variant = "LSTM_CVAE"

    return f"{base}_{variant}"


def reexport(namespace: dict, base: str) -> str:
    mod = import_module(f"utils.{_module_name(base)}")
    public = getattr(mod, "__all__", None)
    if public is None:
        public = [k for k in mod.__dict__.keys() if not k.startswith("_")]

    for name in public:
        namespace[name] = getattr(mod, name)

    namespace["__all__"] = public
    namespace["_TARGET_MODULE"] = mod.__name__
    return mod.__name__
