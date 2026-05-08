from __future__ import annotations

from utils.dataloader import dataset
from utils.utils import *


class ConditionalVAELoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_beta: float = 1.0,
        cyc_steps: int = 400,
        num_cycles: int = 4,
        anneal_steps: int = 1000,
        free_bits: float = 0.02,
        capacity_max: float = 0.0,
        capacity_inc: float = 0.002,
        gamma: float = 5.0,
        prop_w: float = 1.0,
        nce: float = 0.02,
        sig_pen_q: float = 0.003,
        sig_pen_p: float = 0.003,
        imb: float = 0.05,
        # Optional advanced regularizers (used by current LSTM CVAE notebooks).
        mu_align_w: float = 0.0,
        sigma_p_target: float = 0.12,
        sigma_q_target: float = 0.145,
        sigma_p_target_w: float = 0.0,
        sigma_q_target_w: float = 0.0,
        sigma_align_w: float = 0.0,
        sigma_p_var_floor: float = 0.011,
        sigma_p_var_w: float = 0.0,
        sigma_p_upper_margin: float = 0.010,
        sigma_p_upper_w: float = 0.0,
        sigma_p_hi_ratio_target: float = 0.33,
        sigma_p_hi_ratio_tau: float = 0.008,
        sigma_p_hi_ratio_w: float = 0.0,
        sigma_q_var_floor: float = 0.009,
        sigma_q_var_w: float = 0.0,
        mu_std_floor: float = 0.7,
        mu_std_w: float = 0.0,
        mu_std_global_floor: float = 0.0,
        mu_std_global_w: float = 0.0,
        mu_energy_floor: float = 0.0,
        mu_energy_w: float = 0.0,
        reg_warmup_steps: int = 0,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.V = vocab_size
        self.fb = free_bits
        self.max_beta = max_beta
        self.cyc_steps = cyc_steps
        self.num_cycles = num_cycles
        self.anneal = anneal_steps
        self.C_max = capacity_max
        self.C_inc = capacity_inc
        self.gamma = gamma
        self.prop_w = prop_w
        cond_dim = int(dataset.properties.shape[1]) if hasattr(dataset, "properties") else 1
        self.proj = nn.Linear(cond_dim, self.latent_dim)
        self.nce = nce
        self.sig_pen_q = sig_pen_q
        self.sig_pen_p = sig_pen_p
        self.imb = imb
        self.mu_align_w = mu_align_w
        self.sigma_p_target = sigma_p_target
        self.sigma_q_target = sigma_q_target
        self.sigma_p_target_w = sigma_p_target_w
        self.sigma_q_target_w = sigma_q_target_w
        self.sigma_align_w = sigma_align_w
        self.sigma_p_var_floor = sigma_p_var_floor
        self.sigma_p_var_w = sigma_p_var_w
        self.sigma_p_upper_margin = sigma_p_upper_margin
        self.sigma_p_upper_w = sigma_p_upper_w
        self.sigma_p_hi_ratio_target = sigma_p_hi_ratio_target
        self.sigma_p_hi_ratio_tau = sigma_p_hi_ratio_tau
        self.sigma_p_hi_ratio_w = sigma_p_hi_ratio_w
        self.sigma_q_var_floor = sigma_q_var_floor
        self.sigma_q_var_w = sigma_q_var_w
        self.mu_std_floor = mu_std_floor
        self.mu_std_w = mu_std_w
        self.mu_std_global_floor = mu_std_global_floor
        self.mu_std_global_w = mu_std_global_w
        self.mu_energy_floor = mu_energy_floor
        self.mu_energy_w = mu_energy_w
        self.reg_warmup_steps = max(0, int(reg_warmup_steps))

    def cyclical_beta(self, step: int, max_beta: float, cyc_steps: int, num_cycles: int) -> float:
        cycle_idx = step // cyc_steps
        if cycle_idx >= num_cycles:
            return max_beta
        pos = (step % cyc_steps) / cyc_steps
        return max_beta * pos

    def info_nce(self, z, y, temperature=0.2):
        z = F.normalize(z, dim=-1)
        y = F.normalize(y, dim=-1)
        logits = torch.mm(z, y.t()) / temperature
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(logits, labels)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _ensure_bld(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            return t.unsqueeze(1)
        return t

    def forward(
        self,
        logits,
        target_tokens,
        mu_q,
        lv_q,
        mu_p,
        lv_p,
        prop_pred_mu,
        true_prop,
        prop_pred_z,
        step: int,
        latent_mask=None,
    ):
        B, _, _ = logits.size()
        mu_q = self._ensure_bld(mu_q)
        lv_q = self._ensure_bld(lv_q)
        mu_p = self._ensure_bld(mu_p)
        lv_p = self._ensure_bld(lv_p)

        if latent_mask is None:
            latent_mask = torch.ones(B, mu_q.size(1), dtype=torch.float32, device=mu_q.device)
        latent_mask = latent_mask.to(device=mu_q.device, dtype=torch.float32)
        if latent_mask.size(1) != mu_q.size(1):
            if latent_mask.size(1) > mu_q.size(1):
                latent_mask = latent_mask[:, : mu_q.size(1)]
            else:
                pad = torch.ones(
                    latent_mask.size(0),
                    mu_q.size(1) - latent_mask.size(1),
                    device=latent_mask.device,
                    dtype=latent_mask.dtype,
                )
                latent_mask = torch.cat([latent_mask, pad], dim=1)
        latent_mask_3d = latent_mask.unsqueeze(-1)
        valid_token_count = latent_mask.sum(dim=1).clamp_min(1.0)

        recon = F.cross_entropy(
            logits.reshape(-1, self.V),
            target_tokens.view(-1),
            reduction="sum",
            ignore_index=dataset.vocab["[PAD]"],
        ) / B

        q = Normal(mu_q, torch.exp(0.5 * lv_q))
        p = Normal(mu_p, torch.exp(0.5 * lv_p))
        kld_dim = torch.distributions.kl_divergence(q, p)

        if self.fb > 0.0:
            kld_dim = torch.clamp(kld_dim, min=self.fb)

        kld_token = (kld_dim * latent_mask_3d).sum(-1)
        kld_seq = kld_token.sum(-1)
        raw_kld_seq = kld_seq.mean()
        kld_per_token = (kld_seq / valid_token_count).mean()

        beta = self.cyclical_beta(step, self.max_beta, self.cyc_steps, self.num_cycles)
        kl_term = beta * kld_per_token

        prop_loss_mu = F.mse_loss(prop_pred_mu, true_prop)
        prop_loss_z = F.mse_loss(prop_pred_z, true_prop)

        cond_input = true_prop.squeeze(-1) if true_prop.dim() == 3 else true_prop
        if cond_input.dim() == 1:
            cond_input = cond_input.unsqueeze(-1)
        cond = F.relu(self.proj(cond_input))
        z = q.rsample()
        z_mean = (z * latent_mask_3d).sum(dim=1) / valid_token_count.unsqueeze(-1)
        info_nce = self.info_nce(z_mean, cond)

        kld_valid = kld_token[latent_mask > 0]
        if kld_valid.numel() > 1:
            imb = ((kld_valid - kld_valid.mean()) ** 2).mean()
        else:
            imb = kld_token.new_tensor(0.0)

        sigma_q = torch.exp(0.5 * lv_q)
        sigma_p = torch.exp(0.5 * lv_p)
        valid_latent_count = (latent_mask.sum() * sigma_q.size(-1)).clamp_min(1.0)
        sig_pen_q = (sigma_q * latent_mask_3d).sum() / valid_latent_count
        sig_pen_p = (sigma_p * latent_mask_3d).sum() / valid_latent_count

        mu_align = F.mse_loss(mu_p, mu_q.detach())
        sigma_p_center = (torch.abs(sigma_p - self.sigma_p_target) * latent_mask_3d).sum() / valid_latent_count
        sigma_q_center = (torch.abs(sigma_q - self.sigma_q_target) * latent_mask_3d).sum() / valid_latent_count
        sigma_align = (torch.abs(sigma_p - sigma_q.detach()) * latent_mask_3d).sum() / valid_latent_count

        valid_pos = latent_mask > 0
        if valid_pos.any():
            sigma_p_valid = sigma_p[valid_pos]  # [N_valid, D]
            sigma_q_valid = sigma_q[valid_pos]  # [N_valid, D]
            mu_q_valid = mu_q[valid_pos]        # [N_valid, D]
            if sigma_p_valid.size(0) > 1:
                sigma_p_batch_std = sigma_p_valid.std(dim=0, unbiased=False).mean()
            else:
                sigma_p_batch_std = sigma_p_valid.new_tensor(0.0)
            if sigma_q_valid.size(0) > 1:
                sigma_q_batch_std = sigma_q_valid.std(dim=0, unbiased=False).mean()
            else:
                sigma_q_batch_std = sigma_q_valid.new_tensor(0.0)
            if mu_q_valid.size(0) > 1:
                mu_q_batch_std = mu_q_valid.std(dim=0, unbiased=False).mean()
                mu_q_global_std = mu_q_valid.std(unbiased=False)
                mu_q_energy = (mu_q_valid * mu_q_valid).mean()
            else:
                mu_q_batch_std = mu_q_valid.new_tensor(0.0)
                mu_q_global_std = mu_q_valid.new_tensor(0.0)
                mu_q_energy = mu_q_valid.new_tensor(0.0)
        else:
            sigma_p_batch_std = sigma_p.new_tensor(0.0)
            sigma_q_batch_std = sigma_q.new_tensor(0.0)
            mu_q_batch_std = mu_q.new_tensor(0.0)
            mu_q_global_std = mu_q.new_tensor(0.0)
            mu_q_energy = mu_q.new_tensor(0.0)

        sigma_p_var_pen = F.relu(self.sigma_p_var_floor - sigma_p_batch_std)
        sigma_q_var_pen = F.relu(self.sigma_q_var_floor - sigma_q_batch_std)
        mu_std_pen = F.relu(self.mu_std_floor - mu_q_batch_std)
        mu_std_global_pen = F.relu(self.mu_std_global_floor - mu_q_global_std)
        mu_energy_pen = F.relu(self.mu_energy_floor - mu_q_energy)

        sigma_p_upper = self.sigma_p_target + self.sigma_p_upper_margin
        sigma_p_upper_pen = (F.relu(sigma_p - sigma_p_upper) * latent_mask_3d).sum() / valid_latent_count
        sigma_p_hi_soft = torch.sigmoid((sigma_p - sigma_p_upper) / max(self.sigma_p_hi_ratio_tau, 1e-6))
        sigma_p_hi_ratio = (sigma_p_hi_soft * latent_mask_3d).sum() / valid_latent_count
        sigma_p_hi_ratio_pen = F.relu(sigma_p_hi_ratio - self.sigma_p_hi_ratio_target)
        reg_scale = 1.0
        if self.reg_warmup_steps > 0:
            reg_scale = min(1.0, float(step + 1) / float(self.reg_warmup_steps))

        loss = (
            recon
            + kl_term
            + self.prop_w * (prop_loss_mu + 0.5 * prop_loss_z)
            + self.nce * info_nce
            + self.sig_pen_q * sig_pen_q
            + self.sig_pen_p * sig_pen_p
            + self.imb * imb
            + reg_scale * self.mu_align_w * mu_align
            + reg_scale * self.sigma_p_target_w * sigma_p_center
            + reg_scale * self.sigma_q_target_w * sigma_q_center
            + reg_scale * self.sigma_align_w * sigma_align
            + reg_scale * self.sigma_p_var_w * sigma_p_var_pen
            + reg_scale * self.sigma_p_upper_w * sigma_p_upper_pen
            + reg_scale * self.sigma_p_hi_ratio_w * sigma_p_hi_ratio_pen
            + reg_scale * self.sigma_q_var_w * sigma_q_var_pen
            + reg_scale * self.mu_std_w * mu_std_pen
            + reg_scale * self.mu_std_global_w * mu_std_global_pen
            + reg_scale * self.mu_energy_w * mu_energy_pen
        )

        return (
            loss,
            recon,
            kl_term.detach(),
            raw_kld_seq.detach(),
            kld_per_token.detach(),
            prop_loss_mu.detach(),
        )


# Backward-compatibility alias kept for older notebooks.
ConditionalVAELoss_LSTM = ConditionalVAELoss


class IWAEPropertyLoss(nn.Module):
    def __init__(
        self,
        prop_w=1.0,
        nce=0.02,
        sig_pen_q=0.003,
        sig_pen_p=0.003,
        imb=0.05,
        latent_dim=64,
    ):
        super().__init__()
        self.prop_w = prop_w
        self.nce_w = nce
        self.sig_pen_q = sig_pen_q
        self.sig_pen_p = sig_pen_p
        self.imb_w = imb
        cond_dim = int(dataset.properties.shape[1]) if hasattr(dataset, "properties") else 1
        self.proj = nn.Linear(cond_dim, latent_dim)

    @staticmethod
    def info_nce(z, y, T=0.5):
        z = F.normalize(z, -1)
        y = F.normalize(y, -1)
        logits = z @ y.t() / T
        return F.cross_entropy(logits, torch.arange(z.size(0), device=z.device))

    def forward(self, iw_term, mu_q, lv_q, mu_p, lv_p, prop_pred_mu, true_prop, log_ws: torch.Tensor):
        prop_loss = F.mse_loss(prop_pred_mu, true_prop)

        q = Normal(mu_q, (0.5 * lv_q).exp())
        p = Normal(mu_p, (0.5 * lv_p).exp())
        sig_pen_q = q.scale.mean()
        sig_pen_p = p.scale.mean()
        imb = ((lv_q - lv_q.mean()) ** 2).mean()

        target_std = 0.30
        mu_center = mu_q.mean(dim=(-2, -1), keepdim=True)
        mu_var = ((mu_q - mu_center) ** 2).mean((-2, -1))
        sigma_q = torch.exp(0.5 * lv_q)
        upper_pen = F.relu(sigma_q - 0.7).mean()
        lower_pen = F.relu(0.5 - sigma_q).mean()
        mu_var_pen = F.relu(target_std**2 - mu_var).mean()

        w = (log_ws - log_ws.max(0)[0]).exp()
        w_n = w / w.sum(0, keepdim=True)
        ess_est = 1.0 / (w_n**2).sum(0)
        ess_pen_min = (F.relu(8.0 - ess_est).mean()) ** 2
        ess_pen_max = (F.relu(ess_est - 18.0).mean()) ** 2
        ess_pen = ess_pen_max + ess_pen_min

        loss = (
            -iw_term.mean()
            + self.prop_w * prop_loss
            + self.sig_pen_q * sig_pen_q
            + self.sig_pen_p * sig_pen_p
            + self.imb_w * imb
            + 1e-3 * mu_var_pen
            + (upper_pen + lower_pen)
            + 1e2 * ess_pen
        )

        return loss, prop_loss, imb


__all__ = [
    "ConditionalVAELoss",
    "ConditionalVAELoss_LSTM",
    "IWAEPropertyLoss",
]
