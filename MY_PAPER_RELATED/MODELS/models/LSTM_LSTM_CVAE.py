from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.LSTM_util import *
from utils.dataloader import dataset
from torch.nn.utils.parametrizations import weight_norm


class LSTMCVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        mid = (d_model + latent_dim) // 2
        self.to_means = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.Dropout(0.1),
            nn.Linear(mid, latent_dim),
        )
        self.to_var = nn.Linear(d_model, latent_dim)
        # Wider posterior sigma band to prevent collapse around a narrow scale.
        self.q_sigma_floor = 0.07
        self.q_sigma_range = 0.26
        self.q_sigma_temp = 1.0

        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder(latent_dim=latent_dim)
        self.to_prop = nn.Linear(latent_dim, 1)
        self.to_prop_z = nn.Linear(latent_dim, 1)
        self.prior = LSTMPriorNet(y_dim=1, latent_dim=latent_dim)

        self.predict = nn.Linear(d_model, dataset.vocab_size)

        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        nn.init.constant_(self.to_var.bias, 0.0)

    def reparameterize(self, mu, lv):
        q = Normal(mu, (0.5 * lv).exp())
        return q.rsample()

    def forward(self, smiles_enc, smiles_dec_input, properties, enc_smi_mask=None, dec_smi_mask=None):
        properties_e = self.input_embedding(properties)
        if properties_e.dim() == 3:
            prop_ctx = properties_e.squeeze(1)
        else:
            prop_ctx = properties_e

        encoded_last = self.encoder(smiles_enc, smi_mask=enc_smi_mask)
        post_ctx = encoded_last + prop_ctx

        means = self.to_means(post_ctx)
        q_sigma = self.q_sigma_floor + self.q_sigma_range * torch.sigmoid(self.to_var(post_ctx) / self.q_sigma_temp)
        log_var = (q_sigma * q_sigma).log()

        z = self.reparameterize(means, log_var)

        tgt = self.to_prop(means)
        tgt_z = self.to_prop_z(z)
        output = self.decoder(
            smiles_dec_input,
            z,
            enc_smi_mask,
            dec_smi_mask,
        )

        output = self.predict(output)

        return output, tgt, means.unsqueeze(1), log_var.unsqueeze(1), tgt_z


class LSTMPriorNet(nn.Module):
    """
    Simple Prior Network that maps condition y to prior distribution parameters (mu_p, logvar_p).
    """

    def __init__(self, y_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            weight_norm(nn.Linear(y_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.y_skip = nn.Linear(y_dim, hidden_dim, bias=False)
        self.y_sigma_skip = nn.Linear(y_dim, hidden_dim, bias=False)
        self.y_sigma_direct = nn.Linear(y_dim, latent_dim, bias=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)
        nn.init.normal_(self.fc_sigma.weight, mean=0.0, std=0.04)
        nn.init.constant_(self.fc_sigma.bias, -0.20)
        nn.init.normal_(self.y_sigma_direct.weight, mean=0.0, std=0.08)
        nn.init.constant_(self.y_sigma_direct.bias, 0.0)
        # Wider prior sigma band to reduce boundary pile-up.
        self.sigma_floor = 0.06
        self.sigma_range = 0.24
        self.sigma_temp = 1.20
        self.y_sigma_scale = 0.35

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.dim() == 3:
            y = y.squeeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        h_base = self.mlp(y)
        h_mu = h_base + self.y_skip(y)
        # Compress skewed condition values to avoid prior-sigma pile-up at the upper edge.
        y_sigma = torch.tanh(y * self.y_sigma_scale)
        h_sigma = h_base + self.y_sigma_skip(y_sigma)
        mu = self.fc_mu(h_mu).unsqueeze(1)
        sigma_logits = self.fc_sigma(h_sigma) + self.y_sigma_direct(y_sigma)
        sigma = self.sigma_floor + self.sigma_range * torch.sigmoid(sigma_logits / self.sigma_temp)
        return mu, (sigma * sigma).log().unsqueeze(1)


# Backward-compatible aliases
CVAE = LSTMCVAE
PriorNet = LSTMPriorNet
