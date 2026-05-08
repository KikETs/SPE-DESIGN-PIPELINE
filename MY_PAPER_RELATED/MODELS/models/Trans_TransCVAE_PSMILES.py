from pathlib import Path
import sys
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.Trans_util_TransCVAE_PSMILES import *
from utils.dataloader import dataset
from torch.nn.utils.parametrizations import weight_norm
class CVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim = 64, vocab_size=None, max_len=None, pad_idx=None):
        super().__init__()
        vocab_size = dataset.vocab_size if vocab_size is None else int(vocab_size)
        max_len = dataset.max_len if max_len is None else int(max_len)
        self.len = max_len + 1  # sequence + conductivity token
        self.latent_dim = latent_dim
        mid=(d_model+latent_dim)//2
        self.to_means = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.Dropout(0.1),
            nn.Linear(mid, latent_dim)
        )
        self.to_var = nn.Linear(d_model, latent_dim)

        self.encoder = TFEncoder(vocab_size=vocab_size, pad_idx=pad_idx)
        self.decoder = TFDecoder(latent_dim=latent_dim, vocab_size=vocab_size, pad_idx=pad_idx)
        self.to_prop = nn.Linear(self.len*latent_dim, 1)
        self.to_prop_z = nn.Linear(self.len*latent_dim, 1)
        self.prior = PriorNet(y_dim=1, latent_dim=latent_dim)

        self.predict = nn.Linear(d_model, vocab_size)

        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        nn.init.constant_(self.to_var.bias, -3.0)

    def reparameterize(self, mu, lv):
        q = Normal(mu, (0.5 * lv).exp())
        return q.rsample()

    def forward(self, smiles_enc, smiles_dec_input, properties, enc_smi_mask = None, dec_smi_mask = None):
        B = smiles_enc.size(0)
        properties_e = self.input_embedding(properties)

        encoded = self.encoder(smiles_enc, properties_e, enc_smi_mask)

        means = self.to_means(encoded)
        log_var = self.to_var(encoded).clamp_(min=-4)
        latent_mask = None
        if enc_smi_mask is not None:
            # latent tokens for PAD positions are masked out; property token stays valid.
            enc_valid = (~enc_smi_mask).to(device=means.device, dtype=means.dtype)
            prop_valid = torch.ones(B, 1, device=means.device, dtype=means.dtype)
            latent_mask = torch.cat((enc_valid, prop_valid), dim=1)  # [B, L+1]
            means = means * latent_mask.unsqueeze(-1)
            pad_lv = torch.full_like(log_var, -20.0)
            log_var = torch.where(latent_mask.unsqueeze(-1) > 0, log_var, pad_lv)

        z = self.reparameterize(means, log_var)
        if latent_mask is not None:
            z = z * latent_mask.unsqueeze(-1)

        tgt = self.to_prop(means.view(-1, self.len*self.latent_dim))
        tgt_z = self.to_prop_z(z.view(-1, self.len*self.latent_dim))
        output = self.decoder(smiles_dec_input, z, enc_smi_mask, dec_smi_mask)


        output = self.predict(output)



        return output, tgt, means, log_var, tgt_z
class PriorNet(nn.Module):
    """
    Simple Prior Network that maps condition y to prior distribution parameters (mu_p, logvar_p).

    Args:
        y_dim (int): Dimensionality of condition vector y.
        latent_dim (int): Dimensionality of latent space.
        hidden_dim (int): Hidden size for MLP.
    """
    def __init__(self, y_dim: int, latent_dim: int, hidden_dim: int = 256, max_len=None):
        super().__init__()
        max_len = dataset.max_len if max_len is None else int(max_len)
        self.len = max_len + 1  # sequence + conductivity token
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            weight_norm(nn.Linear(y_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(hidden_dim, hidden_dim*self.len)),
            nn.GELU()
        )
        self.y_skip = nn.Linear(y_dim, hidden_dim, bias=False)
        self.y_sigma_skip = nn.Linear(y_dim, hidden_dim, bias=False)
        self.y_sigma_direct = nn.Linear(y_dim, latent_dim, bias=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

        nn.init.normal_(self.fc_sigma.weight, mean=0.0, std=0.04)
        nn.init.constant_(self.fc_sigma.bias, 0.0)
        nn.init.normal_(self.y_sigma_direct.weight, mean=0.0, std=0.07)
        nn.init.constant_(self.y_sigma_direct.bias, 0.0)

        # Explicit sigma band avoids over-concentrated prior variance.
        self.sigma_floor = 0.07
        self.sigma_range = 0.26
        self.sigma_temp = 1.10
        self.y_sigma_scale = 0.36

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mu_p and logvar_p given condition y.

        Args:
            y: Tensor of shape [batch_size, y_dim]

        Returns:
            mu_p: Tensor of shape [batch_size, latent_dim]
            logvar_p: Tensor of same shape
        """
        if y.dim() == 3:
            y = y.squeeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        h = self.mlp(y).view(-1, self.len, self.hidden_dim)
        y_seq = y.unsqueeze(1).expand(-1, self.len, -1)

        h_mu = h + self.y_skip(y_seq)
        mu = self.fc_mu(h_mu)

        # Compress skewed scalar condition before sigma mapping.
        y_sigma = torch.tanh(y * self.y_sigma_scale)
        y_sigma_seq = y_sigma.unsqueeze(1).expand(-1, self.len, -1)
        h_sigma = h + self.y_sigma_skip(y_sigma_seq)
        sigma_logits = self.fc_sigma(h_sigma) + self.y_sigma_direct(y_sigma_seq)
        sigma = self.sigma_floor + self.sigma_range * torch.sigmoid(sigma_logits / self.sigma_temp)

        return mu, (sigma * sigma).log()
