import math
import os
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import HTML, clear_output, display
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdmolops
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.notebook import tqdm

try:
    import ipywidgets as widgets
except Exception:
    widgets = None

try:
    import umap.umap_ as umap
except Exception:
    umap = None

try:
    from psmiles import PolymerSmiles as PS
except Exception:
    PS = None

try:
    import selfies_psmiles as sfp
except Exception:
    sfp = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_transformer = FunctionTransformer(np.log1p, validate=True)
log_minmax_pipeline = Pipeline(
    steps=[
        ("log", log_transformer),
        ("minmax", MinMaxScaler()),
    ]
)

# PSMILES tokenizer:
# - keeps bracket expressions (e.g. [*], [SiH2]) intact
# - keeps common two-letter atoms outside brackets
# - falls back to single-char tokens for operators/unknown chars
PSMILES_TOKEN_RE = re.compile(
    r"\[[^\[\]]+\]|Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|Fe|Zn|Cu|Mn|Hg|Ag|Sn|Pb|As|%[0-9]{2}|."
)


def split_psmiles_tokens(psmiles: str) -> list[str]:
    psmiles = str(psmiles).strip()
    if not psmiles:
        return []
    return PSMILES_TOKEN_RE.findall(psmiles)


def decode_keep_star(token_str: str, sanitize: bool = False, verbose: bool = True):
    """
    Token string(SELFIES/PSMILES) -> RDKit Mol while preserving '*' dummy atoms.
    - If selfies_psmiles is available, first try SELFIES-style decode.
    - Otherwise parse the raw string as SMILES/PSMILES directly.
    """
    smiles = None
    if sfp is not None:
        try:
            smiles = sfp.decoder_psmiles(token_str)
        except Exception:
            smiles = None

    if smiles is None:
        smiles = token_str

    if verbose:
        print(f"[decode_keep_star] INPUT : {token_str}")
        print(f"[decode_keep_star] SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    if sanitize:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)

    return mol


def _canonicalize_polymer_smiles(smiles: str) -> str:
    if PS is None:
        return smiles
    try:
        cand = PS(smiles).canonicalize.psmiles
        if cand.count("*") == 2 and Chem.MolFromSmiles(cand):
            return cand
    except Exception:
        pass
    return smiles


def tok_ids_to_smiles(tok_ids, id2tok):
    """
    Token-id sequence -> canonical polymer SMILES/PSMILES.
    - uses tokens before [EOS]
    - drops [SOS]/[PAD]
    """
    tokens = [id2tok[i] for i in tok_ids]
    if "[EOS]" in tokens:
        tokens = tokens[: tokens.index("[EOS]")]
    tokens = [t for t in tokens if t not in {"[SOS]", "[PAD]"}]
    if not tokens:
        return None

    joined = "".join(tokens)
    try:
        mol = decode_keep_star(joined, sanitize=False, verbose=False)
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None

    return _canonicalize_polymer_smiles(smiles)


def compute_ess(log_w: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Effective Sample Size (ESS)
    """
    log_w = log_w.double()
    log_sum_w = torch.logsumexp(log_w, dim=dim)
    log_sum_w2 = torch.logsumexp(2.0 * log_w, dim=dim)
    ess = torch.exp(2.0 * log_sum_w - log_sum_w2)
    return ess


def make_src_key_padding_mask(lengths: torch.Tensor, max_len: int | None = None):
    """
    lengths: (batch,) sequence lengths
    return : (batch, max_len) True=PAD, False=valid
    """
    if max_len is None:
        max_len = lengths.max().item()
    range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = range_row >= lengths.unsqueeze(1)
    return mask


def _unwrap_state_dict_for_loading(checkpoint_or_state_dict):
    if isinstance(checkpoint_or_state_dict, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint_or_state_dict and isinstance(checkpoint_or_state_dict[key], dict):
                return checkpoint_or_state_dict[key]
    return checkpoint_or_state_dict


def _infer_legacy_encoder_only_vocab(current_vocab: dict, removed_tokens=()):
    removed = set(removed_tokens)
    toks = sorted([tok for tok in current_vocab.keys() if tok not in removed])
    return {tok: i for i, tok in enumerate(toks)}


def load_encoder_only_checkpoint_compat(
    module: nn.Module,
    checkpoint_or_state_dict,
    current_vocab: dict | None = None,
    *,
    removed_vocab_tokens=(),
    verbose: bool = True,
):
    """
    Backward-compatible checkpoint loader for Encoder_Only models.
    Token-dependent layers are remapped by token string if vocab changed.
    """
    state_dict = _unwrap_state_dict_for_loading(checkpoint_or_state_dict)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict-like dict, got {type(state_dict).__name__}")

    current = module.state_dict()
    token_keys = ("smi_emb.weight", "predict.weight", "predict.bias")

    matched = {}
    skipped = []
    for k, v in state_dict.items():
        if k in token_keys:
            continue
        if k in current and tuple(current[k].shape) == tuple(v.shape):
            matched[k] = v
        else:
            skipped.append(k)
    module.load_state_dict(matched, strict=False)

    src_vocab = None
    src_vocab_source = None
    if isinstance(checkpoint_or_state_dict, dict) and isinstance(checkpoint_or_state_dict.get("vocab"), dict):
        src_vocab = checkpoint_or_state_dict["vocab"]
        src_vocab_source = "checkpoint.vocab"

    token_remap = {"mode": "none", "copied": {}, "overlap": None}
    current_vocab_local = current_vocab
    if current_vocab_local is not None:
        current_vocab_local = {str(k): int(v) for k, v in current_vocab_local.items()}

    if src_vocab is None and current_vocab_local is not None:
        src_emb_rows = None
        if "smi_emb.weight" in state_dict and torch.is_tensor(state_dict["smi_emb.weight"]):
            src_emb_rows = int(state_dict["smi_emb.weight"].shape[0])
        dst_emb_rows = int(current["smi_emb.weight"].shape[0]) if "smi_emb.weight" in current else None
        n_cond = sum(1 for t in removed_vocab_tokens if t in current_vocab_local)
        if src_emb_rows is not None and dst_emb_rows is not None and src_emb_rows == dst_emb_rows - n_cond:
            src_vocab = _infer_legacy_encoder_only_vocab(current_vocab_local, removed_tokens=removed_vocab_tokens)
            src_vocab_source = "inferred_legacy_vocab"

    if src_vocab is not None and current_vocab_local is not None:
        copied = {"smi_emb.weight": 0, "predict.weight": 0, "predict.bias": 0}
        with torch.no_grad():
            if "smi_emb.weight" in state_dict and "smi_emb.weight" in current:
                src = state_dict["smi_emb.weight"]
                dst = module.smi_emb.weight
                for tok, dst_idx in current_vocab_local.items():
                    src_idx = src_vocab.get(tok)
                    if src_idx is None or src_idx >= src.size(0) or dst_idx >= dst.size(0):
                        continue
                    dst[dst_idx].copy_(src[src_idx].to(dst.device))
                    copied["smi_emb.weight"] += 1
            if "predict.weight" in state_dict and "predict.weight" in current:
                src = state_dict["predict.weight"]
                dst = module.predict.weight
                for tok, dst_idx in current_vocab_local.items():
                    src_idx = src_vocab.get(tok)
                    if src_idx is None or src_idx >= src.size(0) or dst_idx >= dst.size(0):
                        continue
                    dst[dst_idx].copy_(src[src_idx].to(dst.device))
                    copied["predict.weight"] += 1
            if "predict.bias" in state_dict and "predict.bias" in current:
                src = state_dict["predict.bias"]
                dst = module.predict.bias
                for tok, dst_idx in current_vocab_local.items():
                    src_idx = src_vocab.get(tok)
                    if src_idx is None or src_idx >= src.size(0) or dst_idx >= dst.size(0):
                        continue
                    dst[dst_idx].copy_(src[src_idx].to(dst.device))
                    copied["predict.bias"] += 1
        token_remap = {
            "mode": "token_string_remap",
            "copied": copied,
            "overlap": int(sum(1 for tok in current_vocab_local if tok in src_vocab)),
            "src_vocab_source": src_vocab_source,
        }
    else:
        direct_copied = {}
        direct_state = {}
        for k in token_keys:
            if k in state_dict and k in current and tuple(state_dict[k].shape) == tuple(current[k].shape):
                direct_state[k] = state_dict[k]
                direct_copied[k] = int(current[k].shape[0]) if current[k].ndim > 0 else 1
        if direct_state:
            module.load_state_dict(direct_state, strict=False)
        if direct_copied:
            token_remap = {"mode": "direct_copy_same_shape", "copied": direct_copied, "overlap": None}

    summary = {
        "matched": len(matched),
        "skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "token_remap": token_remap,
    }
    if verbose:
        print("[load_encoder_only_checkpoint_compat]", summary)
    return summary
