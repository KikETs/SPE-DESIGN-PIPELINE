from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from common.io_utils import ensure_dir, utc_now_tag, write_json
from common.seed import set_seed
from models.endpoint_pointer import EndpointPointerModel, decode_two_positions, masked_pointer_loss
from train.datasets import EndpointDataset, build_vocab, collate_endpoint_batch, save_vocab
from train.metrics import endpoint_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train endpoint prediction baseline")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for k in ["input_ids", "attention_mask", "insertion_mask", "labels_a", "labels_b"]:
        if k in out:
            out[k] = out[k].to(device)
    return out


def _evaluate(model: EndpointPointerModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    preds: list[tuple[int, int]] = []
    tgts: list[tuple[int, int]] = []

    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            logits_a, logits_b = model(b["input_ids"], b["attention_mask"])
            loss = masked_pointer_loss(
                logits_a,
                logits_b,
                b["labels_a"],
                b["labels_b"],
                b["insertion_mask"],
            )
            losses.append(float(loss.item()))

            pred_pairs = decode_two_positions(
                logits_a,
                logits_b,
                b["insertion_mask"],
                constrained=True,
            )
            target_pairs = [
                (int(a), int(bb))
                for a, bb in zip(b["labels_a"].detach().cpu().tolist(), b["labels_b"].detach().cpu().tolist())
            ]
            preds.extend(pred_pairs)
            tgts.extend(target_pairs)

    m = endpoint_metrics(preds, tgts)
    m["loss"] = float(sum(losses) / max(1, len(losses)))
    return m


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=True)

    exp_name = str(cfg.get("experiment", {}).get("name", "pi1m_endpoint_baseline"))
    output_root = Path(cfg.get("experiment", {}).get("output_root", "outputs/experiments"))
    run_dir = ensure_dir(output_root / f"{exp_name}_{utc_now_tag()}")

    data_cfg = cfg.get("data", {})
    dataset_dir = Path(data_cfg.get("dataset_dir", "outputs/pi1m_endpoint_dataset"))
    train_file = dataset_dir / str(data_cfg.get("train_file", "train.jsonl"))
    valid_file = dataset_dir / str(data_cfg.get("valid_file", "valid.jsonl"))

    train_rows = EndpointDataset.from_jsonl(train_file, vocab=build_vocab([]), with_labels=True).rows
    valid_rows = EndpointDataset.from_jsonl(valid_file, vocab=build_vocab([]), with_labels=True).rows

    if args.debug:
        train_rows = train_rows[: min(2048, len(train_rows))]
        valid_rows = valid_rows[: min(512, len(valid_rows))]
    else:
        max_train = data_cfg.get("max_train_samples")
        max_valid = data_cfg.get("max_valid_samples")
        if max_train is not None:
            train_rows = train_rows[: int(max_train)]
        if max_valid is not None:
            valid_rows = valid_rows[: int(max_valid)]

    vocab = build_vocab(train_rows, min_freq=int(data_cfg.get("min_token_freq", 1)))

    train_ds = EndpointDataset(train_rows, vocab=vocab, with_labels=True)
    valid_ds = EndpointDataset(valid_rows, vocab=vocab, with_labels=True)

    tr_cfg = cfg.get("training", {})
    batch_size = int(tr_cfg.get("batch_size", 128))
    num_workers = int(tr_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_endpoint_batch(b, pad_id=vocab.pad_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_endpoint_batch(b, pad_id=vocab.pad_id),
    )

    model_cfg = cfg.get("model", {})
    model = EndpointPointerModel(
        vocab_size=len(vocab.token_to_id),
        pad_id=vocab.pad_id,
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )

    device_name = str(tr_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-2)),
    )

    epochs = int(tr_cfg.get("epochs", 30))
    patience = int(tr_cfg.get("early_stopping_patience", 6))
    grad_clip = float(tr_cfg.get("grad_clip", 1.0))

    best_metric = -1.0
    best_epoch = -1
    wait = 0
    history: list[dict[str, Any]] = []

    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    save_vocab(vocab, run_dir / "vocab.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_steps = 0

        for batch in train_loader:
            b = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            logits_a, logits_b = model(b["input_ids"], b["attention_mask"])
            loss = masked_pointer_loss(
                logits_a,
                logits_b,
                b["labels_a"],
                b["labels_b"],
                b["insertion_mask"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running += float(loss.item())
            n_steps += 1

        train_loss = running / max(1, n_steps)
        valid_m = _evaluate(model, valid_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_m["loss"],
            "valid_endpoint_exact_match_accuracy": valid_m["endpoint_exact_match_accuracy"],
            "valid_endpoint_pair_accuracy": valid_m["endpoint_pair_accuracy"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        score = float(valid_m["endpoint_pair_accuracy"])
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "model_config": model_cfg,
                    "training_config": tr_cfg,
                    "vocab": vocab.token_to_id,
                    "best_valid_endpoint_pair_accuracy": best_metric,
                },
                ckpt_dir / "best.pt",
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"[early_stop] patience reached at epoch {epoch}")
                break

    write_json(run_dir / "train_history.json", history)
    write_json(
        run_dir / "train_summary.json",
        {
            "config_path": str(args.config.resolve()),
            "run_dir": str(run_dir.resolve()),
            "best_epoch": int(best_epoch),
            "best_valid_endpoint_pair_accuracy": float(best_metric),
            "num_train_samples": len(train_ds),
            "num_valid_samples": len(valid_ds),
            "vocab_size": len(vocab.token_to_id),
            "device": str(device),
        },
    )
    write_json(run_dir / "resolved_config.json", cfg)

    print(f"[done] run_dir={run_dir.resolve()}")


if __name__ == "__main__":
    main()
