import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.utils import *
from utils.dataloader import dataset


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=4,
        d_ff=64,
        enc_seq_len=5000,
        dropout=0.2,
        vocab_size=None,
        pad_idx=None,
        n_layers=2,
    ):
        super().__init__()
        vocab_size = dataset.vocab_size if vocab_size is None else int(vocab_size)
        if pad_idx is None:
            pad_idx = dataset.vocab["[PAD]"]
        self.smi_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, smi, prop=None, smi_mask=None):
        smi_emb = self.smi_embed(smi)
        if smi_mask is not None:
            smi_mask = smi_mask.to(torch.bool)
            smi_lengths = (~smi_mask).sum(dim=1).clamp_min(1).to(torch.long).cpu()
            packed = pack_padded_sequence(
                smi_emb,
                lengths=smi_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.encoder(packed)
        else:
            _, (h_n, _) = self.encoder(smi_emb)
        return h_n[-1]


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=4,
        d_ff=64,
        enc_seq_len=5000,
        dropout=0.2,
        latent_dim=128,
        n_layers=2,
        vocab_size=None,
        pad_idx=None,
    ):
        super().__init__()
        vocab_size = dataset.vocab_size if vocab_size is None else int(vocab_size)
        if pad_idx is None:
            pad_idx = dataset.vocab["[PAD]"]
        self.d_model = d_model
        self.n_layers = n_layers
        self.smi_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.to_h0 = nn.Linear(latent_dim, d_model)
        self.to_c0 = nn.Linear(latent_dim, d_model)
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, dec_input, latent, enc_smi_mask=None, dec_smi_mask=None, latent_mask=None):
        dec_emb = self.smi_embed(dec_input)

        if latent.dim() == 3:
            if latent_mask is not None:
                if latent_mask.dim() == 2:
                    latent_mask = latent_mask.unsqueeze(-1)
                latent_mask = latent_mask.to(device=latent.device, dtype=latent.dtype)
                z_ctx = (latent * latent_mask).sum(dim=1) / latent_mask.sum(dim=1).clamp_min(1.0)
            else:
                z_ctx = latent.mean(dim=1)
        else:
            z_ctx = latent
        z_ctx = torch.tanh(z_ctx)

        B = dec_input.size(0)
        h0 = dec_emb.new_zeros(self.n_layers, B, self.d_model)
        c0 = dec_emb.new_zeros(self.n_layers, B, self.d_model)
        h0[0] = torch.tanh(self.to_h0(z_ctx))
        c0[0] = torch.tanh(self.to_c0(z_ctx))

        if dec_smi_mask is not None:
            dec_smi_mask = dec_smi_mask.to(torch.bool)
            dec_lengths = (~dec_smi_mask).sum(dim=1).clamp_min(1).to(torch.long).cpu()
            packed = pack_padded_sequence(
                dec_emb,
                lengths=dec_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.decoder(packed, (h0, c0))
            decoded, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=dec_emb.size(1),
            )
        else:
            decoded, _ = self.decoder(dec_emb, (h0, c0))
        return decoded
