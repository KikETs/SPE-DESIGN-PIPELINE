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

_MODEL_MAP = {
    "Trans": {
        "Encoder_Only": "Trans_Encoder_Only",
        "Encoder_Only_PSMILES": "Trans_Encoder_Only",
        "LSTM_CVAE": "Trans_LSTM_CVAE",
        "LSTM_CVAE_PSMILES": "Trans_LSTM_CVAE_PSMILES",
        "TransCVAE": "Trans_TransCVAE",
        "TransCVAE_PSMILES": "Trans_TransCVAE_PSMILES",
    },
    "LSTM": {
        "Encoder_Only": "LSTM_LSTM_CVAE",
        "Encoder_Only_PSMILES": "LSTM_LSTM_CVAE_PSMILES",
        "LSTM_CVAE": "LSTM_LSTM_CVAE",
        "LSTM_CVAE_PSMILES": "LSTM_LSTM_CVAE_PSMILES",
        "TransCVAE": "LSTM_LSTM_CVAE",
        "TransCVAE_PSMILES": "LSTM_LSTM_CVAE_PSMILES",
    },
    # Keep old evaluate notebooks importable.
    "Trans_MHA": {
        "Encoder_Only": "Trans_Encoder_Only",
        "Encoder_Only_PSMILES": "Trans_Encoder_Only",
        "LSTM_CVAE": "Trans_LSTM_CVAE",
        "LSTM_CVAE_PSMILES": "Trans_LSTM_CVAE_PSMILES",
        "TransCVAE": "Trans_TransCVAE",
        "TransCVAE_PSMILES": "Trans_TransCVAE_PSMILES",
    },
    "LSTM_MHA": {
        "Encoder_Only": "LSTM_LSTM_CVAE",
        "Encoder_Only_PSMILES": "LSTM_LSTM_CVAE_PSMILES",
        "LSTM_CVAE": "LSTM_LSTM_CVAE",
        "LSTM_CVAE_PSMILES": "LSTM_LSTM_CVAE_PSMILES",
        "TransCVAE": "LSTM_LSTM_CVAE",
        "TransCVAE_PSMILES": "LSTM_LSTM_CVAE_PSMILES",
    },
}


def current_variant() -> str:
    raw = os.environ.get("MODELS_VARIANT", _DEFAULT).strip() or _DEFAULT
    variant = _ALIAS.get(raw, raw)
    if variant not in _SUPPORTED:
        variant = _DEFAULT
    return variant


def reexport(namespace: dict, family: str) -> str:
    variant = current_variant()
    target = _MODEL_MAP[family][variant]
    mod = import_module(f"models.{target}")

    public = getattr(mod, "__all__", None)
    if public is None:
        public = [k for k in mod.__dict__.keys() if not k.startswith("_")]

    for name in public:
        namespace[name] = getattr(mod, name)

    namespace["__all__"] = public
    namespace["_TARGET_MODULE"] = mod.__name__
    return mod.__name__
