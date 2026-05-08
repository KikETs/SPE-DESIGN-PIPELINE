from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.Trans_util import *
from utils.dataloader import dataset
from torch.nn.utils.parametrizations import weight_norm


def _normalize_scalar_condition(cond_values, batch_size: int, device):
    if torch.is_tensor(cond_values):
        x = cond_values.to(device=device, dtype=torch.float32)
    else:
        x = torch.tensor(cond_values, device=device, dtype=torch.float32)

    if x.dim() == 0:
        x = x.view(1, 1, 1)
    elif x.dim() == 1:
        x = x.view(-1, 1, 1)
    elif x.dim() == 2:
        if x.size(-1) != 1:
            raise ValueError(f"scalar condition 2D tensor must have last dim=1, got {tuple(x.shape)}")
        x = x.unsqueeze(1)
    elif x.dim() == 3:
        if x.size(-1) != 1:
            raise ValueError(f"scalar condition 3D tensor must have last dim=1, got {tuple(x.shape)}")
    else:
        raise ValueError(f"unsupported scalar condition shape: {tuple(x.shape)}")

    if x.size(0) == 1 and batch_size > 1:
        x = x.repeat(batch_size, 1, 1)
    if x.size(0) != batch_size:
        raise ValueError(f"scalar condition batch={x.size(0)} != batch_size={batch_size}")
    return x


def _condition_embedding(cond_input, smi_emb: nn.Embedding, mlp: nn.Module, batch_size: int, device):
    del smi_emb
    cond_scalar = _normalize_scalar_condition(cond_input, batch_size, device)
    return mlp(cond_scalar)


class Encoder_Only(nn.Module):
    def __init__(self, d_model=256, latent_dim=64, vocab_size=None):
        super().__init__()
        vocab_size = dataset.vocab_size if vocab_size is None else int(vocab_size)

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=512,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=2,
        )
        self.smi_emb = nn.Embedding(vocab_size, d_model)
        self.predict = nn.Linear(d_model, vocab_size)
        self.mlp = nn.Sequential(
            weight_norm(nn.Linear(1, d_model)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.pos = PositionalEncoding(d_model)

    def forward(self, smiles_dec_input, cond_values, attn_mask=None):
        cond_emb = _condition_embedding(
            cond_values, self.smi_emb, self.mlp, smiles_dec_input.size(0), smiles_dec_input.device
        )
        dec_in = self.smi_emb(smiles_dec_input).clone()
        dec_in[:, :1] = dec_in[:, :1] + cond_emb
        dec_in = self.pos(dec_in)

        T = dec_in.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(dec_in.device)
        if attn_mask is None:
            output = self.decoder(dec_in, causal_mask)
        else:
            output = self.decoder(dec_in, causal_mask, src_key_padding_mask=attn_mask)
        output = self.predict(output)
        return output
