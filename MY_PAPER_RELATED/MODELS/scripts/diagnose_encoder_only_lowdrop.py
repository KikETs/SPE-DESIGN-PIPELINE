#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


def is_models_root(p: Path) -> bool:
    return (p / "utils").is_dir() and (p / "models").is_dir() and (p / "notebooks").is_dir() and (p / "data").is_dir()


def resolve_models_root() -> Path:
    cwd = Path.cwd().resolve()
    for base in [cwd] + list(cwd.parents):
        if is_models_root(base):
            return base
        cand = base / "MY_PAPER_RELATED" / "MODELS"
        if is_models_root(cand):
            return cand
    raise FileNotFoundError("Could not locate MODELS root.")


def purge_project_modules():
    to_del = []
    for k, mod in list(sys.modules.items()):
        if k == "__main__":
            continue
        if mod is None:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        if "/MY_PAPER_RELATED/MODELS/" in str(f).replace("\\", "/"):
            to_del.append(k)
    for k in to_del:
        del sys.modules[k]


def normalize_ref_smiles(smiles: str, canonicalize_polymer) -> str | None:
    if not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is None:
            return None
        can = Chem.MolToSmiles(mol)
        return canonicalize_polymer(can)
    except Exception:
        return None


def is_degenerate(smiles: str | None) -> bool:
    if smiles is None:
        return True
    s = str(smiles).strip()
    if s in {"", "[*]", "[*][*]"}:
        return True
    core = s.replace("[*]", "").replace("*", "")
    core = re.sub(r"[^A-Za-z0-9]", "", core)
    return core == ""


def checkpoint_path(checkpoint_root: Path, model_variant: str, ckpt_variant: str) -> Path:
    stem = f"encoder_only_{model_variant}"
    if ckpt_variant == "baseline":
        return checkpoint_root / f"{stem}.pth"
    if ckpt_variant == "pi1m_finetuned":
        return checkpoint_root / f"{stem}_pi1m_finetuned.pth"
    if ckpt_variant == "pi1m_finetuned_bundle":
        return checkpoint_root / f"{stem}_pi1m_finetuned_bundle.pth"
    raise ValueError(f"Unknown ckpt variant: {ckpt_variant}")


def summarize_rows(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    out = (
        df.groupby(list(group_cols), as_index=False)
        .agg(
            runs=("run", "nunique"),
            valid_mean=("valid_count", "mean"),
            valid_std=("valid_count", "std"),
            valid_unique_mean=("valid_unique_count", "mean"),
            valid_unique_std=("valid_unique_count", "std"),
            novel_unique_mean=("valid_unique_novel_count", "mean"),
            novel_unique_std=("valid_unique_novel_count", "std"),
            invalid_mean=("invalid_count", "mean"),
            invalid_std=("invalid_count", "std"),
            degenerate_mean=("degenerate_count", "mean"),
            degenerate_std=("degenerate_count", "std"),
        )
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )
    for col in out.columns:
        if col.endswith("_std"):
            out[col] = out[col].fillna(0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rounds", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="0 means None")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    models_root = resolve_models_root()
    checkpoints = models_root / "checkpoints"
    out_dir = models_root / "FCD_runs" / "_encoder_only_ckpt_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_variants = ["Encoder_Only", "Encoder_Only_PSMILES"]
    ckpt_variants = ["baseline", "pi1m_finetuned", "pi1m_finetuned_bundle"]
    cond_values = {"LOW": -0.54, "HIGH": 12.0}

    run_rows = []
    remap_rows = []
    top_rows = []

    for model_variant in model_variants:
        os.environ["MODELS_VARIANT"] = model_variant
        purge_project_modules()

        from utils import dataloader as dloader
        from utils import generate as gmod
        from utils import utils as u
        from models.Trans import Encoder_Only

        dataset = dloader.dataset
        vocab = dataset.vocab
        id2tok = {idx: tok for tok, idx in vocab.items()}
        device = u.device

        ref_df = pd.read_csv(models_root / "data" / "simulation-trajectory-aggregate_aligned.csv")
        ref_raw = ref_df.iloc[:, 1].astype(str).tolist()
        ref_norm = [normalize_ref_smiles(s, u._canonicalize_polymer_smiles) for s in ref_raw]
        ref_set = set([s for s in ref_norm if s is not None])

        sos = vocab["[SOS]"]
        eos = vocab["[EOS]"]
        pad = vocab["[PAD]"]
        star_id = vocab.get("[*]", None)

        for ckpt_variant in ckpt_variants:
            ckpt = checkpoint_path(checkpoints, model_variant, ckpt_variant)
            if not ckpt.exists():
                continue
            print(f"\n[Model={model_variant}] [Ckpt={ckpt_variant}] loading: {ckpt.name}")

            model = Encoder_Only(vocab_size=dataset.vocab_size).to(device).eval()
            blob = torch.load(ckpt, map_location=device)
            remap_summary = u.load_encoder_only_checkpoint_compat(
                model, blob, current_vocab=vocab, verbose=False
            )
            remap_rows.append(
                {
                    "model_variant": model_variant,
                    "ckpt_variant": ckpt_variant,
                    "checkpoint": str(ckpt),
                    "remap_summary_json": json.dumps(remap_summary, ensure_ascii=False),
                }
            )

            for cond_key, cond_val in cond_values.items():
                for decode_mode in ["raw", "constrained"]:
                    if decode_mode == "constrained" and cond_key != "LOW":
                        continue
                    print(
                        f"  - cond={cond_key:>4} mode={decode_mode:<11} "
                        f"rounds={args.num_rounds} batch={args.batch_size}"
                    )

                    for run_idx in range(1, args.num_rounds + 1):
                        cond_tensor = torch.full(
                            (args.batch_size, 1, 1),
                            float(cond_val),
                            dtype=torch.float32,
                            device=device,
                        )

                        gen_kwargs = dict(
                            max_length=dataset.max_len + 2,
                            start_token=sos,
                            end_token=eos,
                            pad_token=pad,
                            temperature=float(args.temperature),
                            top_k=None if args.top_k <= 0 else int(args.top_k),
                            top_p=float(args.top_p),
                            device=str(device),
                        )

                        if decode_mode == "constrained":
                            gen_kwargs["forbidden_token_ids"] = [pad, sos]
                            gen_kwargs["eos_min_generated_tokens"] = 4
                            gen_kwargs["star_token_id"] = star_id
                            gen_kwargs["min_non_star_before_eos"] = 1

                        tokens = gmod.generate_batch_sequence(model, cond_tensor, **gen_kwargs)
                        smiles = [u.tok_ids_to_smiles(row, id2tok) for row in tokens]

                        valid = [s for s in smiles if s is not None]
                        valid_unique = list(dict.fromkeys(valid))
                        novel_unique = [s for s in valid_unique if s not in ref_set]
                        degenerate = [s for s in valid if is_degenerate(s)]

                        run_rows.append(
                            {
                                "model_variant": model_variant,
                                "ckpt_variant": ckpt_variant,
                                "condition": cond_key,
                                "decode_mode": decode_mode,
                                "run": run_idx,
                                "batch_size": args.batch_size,
                                "valid_count": int(len(valid)),
                                "valid_unique_count": int(len(valid_unique)),
                                "valid_unique_novel_count": int(len(novel_unique)),
                                "invalid_count": int(args.batch_size - len(valid)),
                                "degenerate_count": int(len(degenerate)),
                            }
                        )
                        if run_idx == 1 or run_idx == args.num_rounds:
                            print(
                                f"    run {run_idx:02d}: valid={len(valid)} "
                                f"uniq={len(valid_unique)} novel={len(novel_unique)} "
                                f"deg={len(degenerate)}"
                            )

                        c = Counter(valid)
                        for sm, cnt in c.most_common(15):
                            top_rows.append(
                                {
                                    "model_variant": model_variant,
                                    "ckpt_variant": ckpt_variant,
                                    "condition": cond_key,
                                    "decode_mode": decode_mode,
                                    "run": run_idx,
                                    "smiles": sm,
                                    "count": int(cnt),
                                    "is_degenerate": bool(is_degenerate(sm)),
                                }
                            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run_df = pd.DataFrame(run_rows)
    remap_df = pd.DataFrame(remap_rows)
    top_df = pd.DataFrame(top_rows)

    summary_df = summarize_rows(
        run_df,
        group_cols=["model_variant", "ckpt_variant", "condition", "decode_mode"],
    )

    run_csv = out_dir / "diagnosis_runs.csv"
    summary_csv = out_dir / "diagnosis_summary.csv"
    remap_csv = out_dir / "checkpoint_remap_logs.csv"
    top_csv = out_dir / "diagnosis_top_smiles.csv"

    run_df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    remap_df.to_csv(remap_csv, index=False)
    top_df.to_csv(top_csv, index=False)

    print("saved:", run_csv)
    print("saved:", summary_csv)
    print("saved:", remap_csv)
    print("saved:", top_csv)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
