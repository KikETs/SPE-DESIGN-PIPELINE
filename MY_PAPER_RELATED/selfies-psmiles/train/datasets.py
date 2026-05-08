from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from common.io_utils import read_jsonl


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]

    @property
    def id_to_token(self) -> dict[int, str]:
        return {v: k for k, v in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[PAD_TOKEN])

    @property
    def unk_id(self) -> int:
        return int(self.token_to_id[UNK_TOKEN])


def build_vocab(rows: list[dict[str, Any]], min_freq: int = 1) -> Vocab:
    cnt = Counter()
    for row in rows:
        for tok in row.get("base_tokens", []):
            cnt[str(tok)] += 1

    tokens = [PAD_TOKEN, UNK_TOKEN]
    for tok, c in sorted(cnt.items()):
        if int(c) >= int(min_freq):
            tokens.append(tok)

    return Vocab(token_to_id={tok: i for i, tok in enumerate(tokens)})


def save_vocab(vocab: Vocab, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"token_to_id": vocab.token_to_id}, path)


def load_vocab(path: str | Path) -> Vocab:
    blob = torch.load(Path(path), map_location="cpu")
    return Vocab(token_to_id={str(k): int(v) for k, v in blob["token_to_id"].items()})


class EndpointDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        vocab: Vocab,
        *,
        with_labels: bool = True,
    ):
        self.rows = rows
        self.vocab = vocab
        self.with_labels = bool(with_labels)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        vocab: Vocab,
        *,
        with_labels: bool = True,
        max_samples: int | None = None,
    ) -> "EndpointDataset":
        rows = read_jsonl(path)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        return cls(rows, vocab=vocab, with_labels=with_labels)

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_tokens(self, tokens: list[str]) -> list[int]:
        ids: list[int] = []
        for tok in tokens:
            ids.append(int(self.vocab.token_to_id.get(tok, self.vocab.unk_id)))
        return ids

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        tokens = [str(x) for x in row["base_tokens"]]
        token_ids = self._encode_tokens(tokens)

        out = {
            "sample_id": row["sample_id"],
            "token_ids": token_ids,
            "base_tokens": tokens,
            "base_psmiles": row["base_psmiles"],
            "canonical_target_psmiles": row["canonical_target_psmiles"],
            "raw_psmiles": row.get("raw_psmiles", ""),
            "canonical_psmiles": row.get("canonical_psmiles", ""),
        }

        if self.with_labels:
            pair = row["endpoint_pair"]
            a, b = int(pair[0]), int(pair[1])
            if a > b:
                a, b = b, a
            out["label_a"] = a
            out["label_b"] = b
        return out


def collate_endpoint_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    bsz = len(batch)
    max_len = max(len(x["token_ids"]) for x in batch)

    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    insertion_mask = torch.zeros((bsz, max_len + 1), dtype=torch.bool)

    sample_ids: list[str] = []
    base_tokens: list[list[str]] = []
    base_psmiles: list[str] = []
    canonical_target: list[str] = []
    raw_psmiles: list[str] = []
    canonical_psmiles: list[str] = []

    labels_a: list[int] = []
    labels_b: list[int] = []
    has_labels = "label_a" in batch[0]

    for i, item in enumerate(batch):
        ids = item["token_ids"]
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :L] = True
        insertion_mask[i, : L + 1] = True

        sample_ids.append(item["sample_id"])
        base_tokens.append(item["base_tokens"])
        base_psmiles.append(item["base_psmiles"])
        canonical_target.append(item["canonical_target_psmiles"])
        raw_psmiles.append(item["raw_psmiles"])
        canonical_psmiles.append(item["canonical_psmiles"])

        if has_labels:
            labels_a.append(int(item["label_a"]))
            labels_b.append(int(item["label_b"]))

    out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "insertion_mask": insertion_mask,
        "sample_ids": sample_ids,
        "base_tokens": base_tokens,
        "base_psmiles": base_psmiles,
        "canonical_target_psmiles": canonical_target,
        "raw_psmiles": raw_psmiles,
        "canonical_psmiles": canonical_psmiles,
    }
    if has_labels:
        out["labels_a"] = torch.tensor(labels_a, dtype=torch.long)
        out["labels_b"] = torch.tensor(labels_b, dtype=torch.long)
    return out
