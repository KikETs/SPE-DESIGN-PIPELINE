from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from common.io_utils import ensure_dir, utc_now_tag, write_json, write_jsonl
from common.repr_utils import build_variant_strings, canonicalize_psmiles
from decode.constrained_reconstruction import reconstruct_with_constraints
from decode.validity_checks import evaluate_reconstruction
from data.canonical_labeler import generate_canonical_label
from models.endpoint_pointer import EndpointPointerModel, decode_two_positions
from train.datasets import EndpointDataset, Vocab, collate_endpoint_batch
from train.metrics import pair_match


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate endpoint prediction model")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--constrained-decoding", action="store_true")
    p.add_argument("--unconstrained-decoding", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--augmentation-eval-samples", type=int, default=200)
    p.add_argument("--max-variants-per-sample", type=int, default=4)
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _length_bucket(n: int) -> str:
    if n <= 32:
        return "<=32"
    if n <= 64:
        return "33-64"
    if n <= 128:
        return "65-128"
    return ">128"


def _augmentation_consistency(
    model: EndpointPointerModel,
    vocab: Vocab,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    constrained: bool,
    max_samples: int,
    max_variants_per_sample: int,
) -> dict[str, Any]:
    if max_samples <= 0:
        return {"evaluated_samples": 0, "consistency_rate": None}

    model.eval()
    subset = rows[: min(max_samples, len(rows))]

    consistent = 0
    evaluated = 0

    with torch.no_grad():
        for row in subset:
            variants = build_variant_strings(row["canonical_target_psmiles"])[:max_variants_per_sample]
            canonical_preds: list[str] = []

            for vi, variant in enumerate(variants):
                lbl, reason = generate_canonical_label(
                    sample_id=f"{row['sample_id']}_aug{vi}",
                    raw_psmiles=variant,
                    allow_multi_endpoint_resolve=False,
                    prefer_psmiles_backend=True,
                )
                if lbl is None:
                    continue

                ids = [int(vocab.token_to_id.get(tok, vocab.unk_id)) for tok in lbl.base_tokens]
                if not ids:
                    continue

                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                mask = torch.ones_like(input_ids, dtype=torch.bool)
                insertion_mask = torch.ones((1, input_ids.size(1) + 1), dtype=torch.bool, device=device)

                logits_a, logits_b = model(input_ids, mask)
                pred_pair = decode_two_positions(
                    logits_a,
                    logits_b,
                    insertion_mask,
                    constrained=constrained,
                )[0]

                recon = reconstruct_with_constraints(
                    lbl.base_tokens,
                    pred_pair,
                    constrained=constrained,
                )
                can = canonicalize_psmiles(recon.reconstructed_psmiles, prefer_psmiles_backend=True).canonical
                canonical_preds.append(can)

            if len(canonical_preds) < 2:
                continue

            evaluated += 1
            if len(set(canonical_preds)) == 1:
                consistent += 1

    return {
        "evaluated_samples": int(evaluated),
        "consistency_rate": (consistent / evaluated) if evaluated > 0 else None,
    }


def main() -> None:
    args = _parse_args()
    cfg = _load_yaml(args.config)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = Vocab(token_to_id={str(k): int(v) for k, v in ckpt["vocab"].items()})

    model_cfg = ckpt.get("model_config", {})
    model = EndpointPointerModel(
        vocab_size=len(vocab.token_to_id),
        pad_id=vocab.pad_id,
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)

    device_name = cfg.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model.to(device)
    model.eval()

    dataset_dir = args.dataset_dir or Path(cfg.get("data", {}).get("dataset_dir", "outputs/pi1m_endpoint_dataset"))
    split_path = dataset_dir / f"{args.split}.jsonl"

    ds = EndpointDataset.from_jsonl(split_path, vocab=vocab, with_labels=True, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("training", {}).get("batch_size", 128)),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_endpoint_batch(b, pad_id=vocab.pad_id),
    )

    constrained = True
    if args.unconstrained_decoding:
        constrained = False
    elif args.constrained_decoding:
        constrained = True

    mode_tag = "constrained" if constrained else "unconstrained"
    if args.output_dir is not None:
        run_out = args.output_dir
    else:
        base_out = Path("outputs/eval") / f"{args.checkpoint.stem}_{mode_tag}_{utc_now_tag()}"
        run_out = base_out
        suffix = 1
        while run_out.exists():
            run_out = Path(f"{base_out}_{suffix}")
            suffix += 1
    run_out = ensure_dir(run_out)

    predictions: list[dict[str, Any]] = []
    pair_correct = 0
    ordered_correct = 0
    two_star_ok = 0
    syntax_ok = 0
    canonical_ok = 0
    roundtrip_ok = 0
    graph_eq_known = 0
    graph_eq_true = 0

    by_len = defaultdict(lambda: {"n": 0, "pair_acc": 0, "canonical_acc": 0})
    failure_counter = Counter()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            insertion_mask = batch["insertion_mask"].to(device)
            labels_a = batch["labels_a"].to(device)
            labels_b = batch["labels_b"].to(device)

            logits_a, logits_b = model(input_ids, attention_mask)
            pred_pairs = decode_two_positions(logits_a, logits_b, insertion_mask, constrained=constrained)

            for i, pred_pair in enumerate(pred_pairs):
                target_pair = (int(labels_a[i].item()), int(labels_b[i].item()))
                pair_ok = pair_match(pred_pair, target_pair)
                ordered_ok = (int(pred_pair[0]) == target_pair[0]) and (int(pred_pair[1]) == target_pair[1])

                recon = reconstruct_with_constraints(
                    batch["base_tokens"][i],
                    pred_pair,
                    constrained=constrained,
                )
                validity = evaluate_reconstruction(
                    recon.reconstructed_psmiles,
                    target_canonical_psmiles=batch["canonical_target_psmiles"][i],
                )

                pair_correct += int(pair_ok)
                ordered_correct += int(ordered_ok)
                two_star_ok += int(validity.exact_two_star)
                syntax_ok += int(validity.syntax_valid)
                roundtrip_ok += int(validity.roundtrip_valid)
                canonical_ok += int(bool(validity.canonical_match))

                if validity.graph_equivalent is not None:
                    graph_eq_known += 1
                    graph_eq_true += int(validity.graph_equivalent)

                L = len(batch["base_tokens"][i])
                bucket = _length_bucket(L)
                by_len[bucket]["n"] += 1
                by_len[bucket]["pair_acc"] += int(pair_ok)
                by_len[bucket]["canonical_acc"] += int(bool(validity.canonical_match))

                if not pair_ok:
                    failure_counter["invalid_endpoint_localization"] += 1
                elif validity.syntax_valid and not bool(validity.canonical_match):
                    failure_counter["reconstruction_valid_but_non_canonical"] += 1
                elif not validity.syntax_valid:
                    failure_counter["syntax_invalid_after_reconstruction"] += 1
                if recon.repaired:
                    failure_counter["repair_applied"] += 1
                if validity.failure_reason:
                    failure_counter[validity.failure_reason] += 1

                predictions.append(
                    {
                        "sample_id": batch["sample_ids"][i],
                        "raw_psmiles": batch["raw_psmiles"][i],
                        "canonical_psmiles": batch["canonical_psmiles"][i],
                        "base_psmiles": batch["base_psmiles"][i],
                        "target_endpoint_pair": [int(target_pair[0]), int(target_pair[1])],
                        "pred_endpoint_pair": [int(pred_pair[0]), int(pred_pair[1])],
                        "pair_accuracy": bool(pair_ok),
                        "ordered_accuracy": bool(ordered_ok),
                        "reconstructed_psmiles": recon.reconstructed_psmiles,
                        "reconstruction_repaired": bool(recon.repaired),
                        "reconstruction_repair_reason": recon.repair_reason,
                        "exact_two_star": bool(validity.exact_two_star),
                        "syntax_valid": bool(validity.syntax_valid),
                        "roundtrip_valid": bool(validity.roundtrip_valid),
                        "canonical_match": bool(validity.canonical_match),
                        "graph_equivalent": validity.graph_equivalent,
                        "failure_reason": validity.failure_reason,
                    }
                )

    n = len(predictions)
    metrics = {
        "num_samples": int(n),
        "endpoint_exact_match_accuracy": ordered_correct / n if n else 0.0,
        "endpoint_pair_accuracy": pair_correct / n if n else 0.0,
        "exact_two_star_reconstruction_rate": two_star_ok / n if n else 0.0,
        "syntax_validity_rate": syntax_ok / n if n else 0.0,
        "canonical_reconstruction_exact_match_rate": canonical_ok / n if n else 0.0,
        "roundtrip_success_rate": roundtrip_ok / n if n else 0.0,
        "graph_equivalence_rate": (graph_eq_true / graph_eq_known) if graph_eq_known > 0 else None,
        "graph_equivalence_known_fraction": (graph_eq_known / n) if n else 0.0,
        "decoding_mode": "constrained" if constrained else "unconstrained",
        "per_length_breakdown": {
            k: {
                "n": int(v["n"]),
                "endpoint_pair_accuracy": v["pair_acc"] / v["n"] if v["n"] else 0.0,
                "canonical_match_rate": v["canonical_acc"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in sorted(by_len.items())
        },
    }

    aug = _augmentation_consistency(
        model,
        vocab,
        ds.rows,
        device,
        constrained=constrained,
        max_samples=int(args.augmentation_eval_samples),
        max_variants_per_sample=int(args.max_variants_per_sample),
    )
    metrics["augmentation_consistency"] = aug

    write_json(run_out / "metrics.json", metrics)
    write_jsonl(run_out / "predictions.jsonl", predictions)

    df = pd.DataFrame(predictions)
    if not df.empty:
        bad = df[df["pair_accuracy"] == False].head(200)
        good = df[df["pair_accuracy"] == True].head(50)
        qual = pd.concat([bad, good], ignore_index=True)
    else:
        qual = df
    qual.to_csv(run_out / "qualitative_samples.csv", index=False)

    error_analysis = {
        "total_predictions": int(n),
        "failure_mode_counts": dict(sorted(failure_counter.items())),
        "top_failure_modes": failure_counter.most_common(20),
    }
    write_json(run_out / "error_analysis.json", error_analysis)

    print(json.dumps({"output_dir": str(run_out.resolve()), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
