from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EndpointPointerModel(nn.Module):
    """
    Boundary pointer baseline:
    - Encodes base token sequence
    - Predicts two insertion indices among [0..L]
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        pad_id: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_id)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        enc_dim = hidden_dim * 2

        self.boundary_start = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.boundary_end = nn.Parameter(torch.zeros(1, 1, enc_dim))

        self.norm = nn.LayerNorm(enc_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.head_a = nn.Linear(enc_dim * 2, 1)
        self.head_b = nn.Linear(enc_dim * 2, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input_ids: [B, L], attention_mask: [B, L] bool
        x = self.embed(input_ids)

        lengths = attention_mask.sum(dim=1).clamp_min(1).to(torch.long).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        enc_packed, _ = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(
            enc_packed,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        B = enc.size(0)
        left = torch.cat([self.boundary_start.expand(B, -1, -1), enc], dim=1)
        right = torch.cat([enc, self.boundary_end.expand(B, -1, -1)], dim=1)
        boundary = torch.cat([left, right], dim=-1)
        boundary = self.drop(self.norm(boundary))

        logits_a = self.head_a(boundary).squeeze(-1)  # [B, L+1]
        logits_b = self.head_b(boundary).squeeze(-1)
        return logits_a, logits_b


def masked_pointer_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    insertion_mask: torch.Tensor,
) -> torch.Tensor:
    neg_inf = torch.finfo(logits_a.dtype).min
    la = logits_a.masked_fill(~insertion_mask, neg_inf)
    lb = logits_b.masked_fill(~insertion_mask, neg_inf)
    loss_a = F.cross_entropy(la, labels_a)
    loss_b = F.cross_entropy(lb, labels_b)
    return 0.5 * (loss_a + loss_b)


def decode_two_positions(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    insertion_mask: torch.Tensor,
    *,
    constrained: bool = True,
) -> list[tuple[int, int]]:
    B, P = logits_a.shape
    neg_inf = torch.finfo(logits_a.dtype).min
    la = logits_a.masked_fill(~insertion_mask, neg_inf)
    lb = logits_b.masked_fill(~insertion_mask, neg_inf)

    out: list[tuple[int, int]] = []
    for i in range(B):
        a_i = la[i]
        b_i = lb[i]

        if constrained:
            score = a_i[:, None] + b_i[None, :]
            valid = insertion_mask[i]
            pair_mask = valid[:, None] & valid[None, :]
            pair_mask = pair_mask & torch.triu(torch.ones((P, P), dtype=torch.bool, device=score.device), diagonal=1)
            score = score.masked_fill(~pair_mask, neg_inf)
            flat = int(torch.argmax(score).item())
            pa = flat // P
            pb = flat % P
            out.append((int(pa), int(pb)))
            continue

        pa = int(torch.argmax(a_i).item())
        pb = int(torch.argmax(b_i).item())

        if pa == pb:
            # deterministic tie-repair
            candidates = torch.argsort(b_i, descending=True).tolist()
            fixed = None
            for c in candidates:
                if int(c) != pa and bool(insertion_mask[i, int(c)]):
                    fixed = int(c)
                    break
            if fixed is None:
                fixed = min(pa + 1, P - 1)
                if fixed == pa:
                    fixed = max(pa - 1, 0)
            pb = fixed

        if pa > pb:
            pa, pb = pb, pa
        out.append((pa, pb))

    return out
