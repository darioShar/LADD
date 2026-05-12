import math
import time
from typing import Any

import lightning.pytorch as L
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from ...utils import instantiate_from_config, instantiate_from_config_hf_pretrained, print_rank_zero
from .masked_process import MaskedDiffusionProcess
from .nonfinite_utils import summarize_nonfinite_tensor
from .noise_sampling import NoiseOutput, NoiseSampling
from .utils import get_anneal_attn_mask


# def loss2bpt(loss, seq_len):
# 	"""Normalize loss to bits per token."""
# 	rescale_to_bpd = 1.0 / (seq_len * math.log(2.0))
# 	bpt = loss * rescale_to_bpd
# 	return bpt


class MaskedDiffusionLoss(nn.Module):
    """MDLM's implementation"""

    def __init__(
        self,
        noise_sampler_config: dict | None = None,
        time_sampler_config: dict | None = None,
        # Make sure to set an id which is different from `pad_token_id`
        mask_token_id: int | None = None,
        attn_mask_annealing_ratio: float | None = None,
        # Proposed in DiffuLLaMA (https://arxiv.org/abs/2410.17891)
        shift_logits: bool = False,
        model_output: str = 'logits',
        conditional: bool = False,
        use_data_loss_elbo: bool = False,
        measure_wall_clock_time: bool = False,
    ):
        super().__init__()
        self.noise_sampler: NoiseSampling = instantiate_from_config(noise_sampler_config) if noise_sampler_config else None
        self.time_sampler = instantiate_from_config(time_sampler_config)
        self.mask_token_id = mask_token_id
        self.attn_mask_annealing_ratio = attn_mask_annealing_ratio
        self.shift_logits = shift_logits
        assert model_output in ['logits', 'logistic_params']
        self.model_output = model_output
        self.masked_process = MaskedDiffusionProcess(mask_token_id)
        self.conditional = conditional
        self.use_data_loss_elbo = use_data_loss_elbo
        self.measure_wall_clock_time = measure_wall_clock_time

    @staticmethod
    def _sync_device_for_timing(device: torch.device) -> None:
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        elif device.type == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def get_annealed_attention_mask(
        self,
        lightning_module: L.LightningModule,
        model: PreTrainedModel,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        # attention mask
        if self.attn_mask_annealing_ratio is not None:
            if lightning_module.training:
                assert lightning_module.trainer.estimated_stepping_batches > 0
                annealing_steps = lightning_module.trainer.estimated_stepping_batches * self.attn_mask_annealing_ratio
                attn_mask_ratio = min(
                    1.0,
                    (lightning_module.trainer.global_step + 1) / annealing_steps,
                )
            else:
                attn_mask_ratio = 1.0
            embed_dtype = model.get_input_embeddings().weight.dtype
            return get_anneal_attn_mask(
                input_ids=x_t,
                attn_mask_ratio=attn_mask_ratio,
                dtype=embed_dtype,
            )
        return None

    def extract_logits_from_output(self, model_output) -> torch.Tensor:
        if self.model_output == 'logits':
            logits = model_output.logits
            # This is logits of E_{p(\hat{x}_0 | x_t)}[\hat{x}_0]
            # mu = F.softmax(logits, dim=-1)  # (bsz, seq_len, vocab_size)
        elif self.model_output == 'logistic_params':
            # `logistic_params` is proposed in D3PM (Austin et al. 2021). See Appendix A.8.
            raise NotImplementedError('Logistic parameters not implemented')
        return logits

    def model_forward(
        self,
        model: PreTrainedModel,
        x_t: torch.Tensor,
        attention_mask: torch.Tensor,
        alphas: NoiseOutput | None,
        t: torch.Tensor | None,
        **model_kwargs: dict[str, Any],
    ):
        model_kwargs['attention_mask'] = attention_mask
        time_type = getattr(model.config, 'time_type', 'none')
        if time_type == 'noise':
            timesteps = alphas.alpha
        elif time_type == 'time':
            timesteps = t
        elif time_type == 'none':
            timesteps = None
        else:
            raise NotImplementedError(f'Unknown time_type: {time_type}')
        model_kwargs['timesteps'] = timesteps
        if not self.measure_wall_clock_time:
            return model(x_t, **model_kwargs)

        self._sync_device_for_timing(x_t.device)
        start_time = time.perf_counter()
        output = model(x_t, **model_kwargs)
        self._sync_device_for_timing(x_t.device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print_rank_zero(f'[MaskedDiffusionLoss] model_forward wall clock: {elapsed_ms:.3f} ms')
        return output

    def forward(
        self,
        lightning_module: L.LightningModule,
        model: PreTrainedModel,
        batch: dict[str, Any],
        **model_kwargs,
    ) -> dict[str, torch.Tensor]:
        # get x_0
        x_0 = batch['input_ids']
        labels = batch.get('labels', x_0.detach().clone())
        attention_mask_0 = self.get_annealed_attention_mask(lightning_module, model, x_0)
        attention_mask_1 = batch.get('attention_mask')
        attention_mask_2 = torch.ones_like(x_0).long()
        # Use coalescing: prefer attention_mask_0, then attention_mask_1, then default to all ones
        if attention_mask_0 is not None:
            attention_mask = attention_mask_0
        elif attention_mask_1 is not None:
            attention_mask = attention_mask_1
        else:
            attention_mask = attention_mask_2
        if attention_mask is attention_mask_2:
            model_attention_mask = None
            loss_attention_mask = attention_mask_2
        else:
            model_attention_mask = attention_mask
            loss_attention_mask = attention_mask
        # This is for SFT training: if labels are not provided, all tokens are maskable
        maskable_mask = labels != -100
        t = self.time_sampler(x_0.size(0)).to(x_0.device)
        alphas = self.noise_sampler(t)
        x_t = self.masked_process.sample_forward(x_0, alphas, maskable_mask, model.config)
        model_output = self.model_forward(model, x_t, model_attention_mask, alphas, t, **model_kwargs)
        logits = self.extract_logits_from_output(model_output)
        # calculate the loss
        loss_dict = self.get_loss(
            logits=logits,
            x_0=x_0,
            x_t=x_t,
            attention_mask=loss_attention_mask,
            labels=labels,
            loss_weight=alphas.loss_weight,
            use_data_loss_elbo=self.use_data_loss_elbo,
        )
        assert 'loss' in loss_dict, 'Loss key not found in the loss dictionary'
        return loss_dict

    def get_loss(
        self,
        logits,
        x_0,
        x_t,
        attention_mask,
        labels,
        loss_weight,
        use_data_loss_elbo=False,
        example_mask=None,
        **kwargs: dict,
    ):
        if not torch.isfinite(logits).all():
            stats = summarize_nonfinite_tensor(logits)
            rank = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            prefix = f'[rank{rank}] ' if rank is not None else ''
            print(
                f'{prefix}Non-finite x_0_pred_logits in MaskedDiffusionLoss.get_loss: '
                f'shape={tuple(logits.shape)} dtype={logits.dtype} device={logits.device} '
                f'nonfinite={stats["num_total"] - stats["num_finite"]}/{stats["num_total"]} '
                f'nan={stats["num_nan"]} inf={stats["num_inf"]} '
                f'finite_min={stats["finite_min"]} finite_max={stats["finite_max"]} '
                f'finite_mean={stats["finite_mean"]}',
            )
            raise ValueError('Non-finite x_0_pred_logits detected in MaskedDiffusionLoss.get_loss')
        mask_indices = x_t == self.mask_token_id
        if self.shift_logits:
            logits = logits[:, :-1]
            x_0 = x_0[:, 1:]
            x_t = x_t[:, 1:]
            attention_mask = attention_mask[:, 1:]
            mask_indices = mask_indices[:, 1:]
            labels = labels[:, 1:]
        # Targets: prefer labels when provided (supports -100), else x_0
        targets = labels if labels is not None else x_0
        targets = targets.long()

        # --- numerically sensitive region in fp32 ---
        with torch.autocast(device_type='cuda', enabled=False):
            logits = self.masked_process.subs_parameterization(
                x=x_t,
                x0_logits=logits.float(),
                mask_indices=mask_indices,
                return_probs=False,
            )
            logprobs = logits.gather(dim=-1, index=x_0[:, :, None]).squeeze(-1)
            loss_diff = (
                logprobs * loss_weight[:, None] * attention_mask if use_data_loss_elbo else -logprobs * attention_mask
            )
            nlls = logprobs * loss_weight[:, None] * attention_mask
            seqlen = attention_mask.sum(dim=-1).clamp_min(1)
            per_example_loss = loss_diff.sum(dim=-1) / seqlen
            per_example_elbo = nlls.sum(dim=-1)
            if example_mask is not None:
                example_weights = example_mask.to(device=per_example_loss.device, dtype=per_example_loss.dtype)
                num_examples = example_weights.sum().clamp_min(1.0)
                loss = (per_example_loss * example_weights).sum() / num_examples
                elbo = (per_example_elbo * example_weights).sum() / num_examples
            else:
                loss = per_example_loss.mean()
                elbo = per_example_elbo.mean()
            loss_dict = {
                'loss': loss,  # training objective
                'loss_x': loss,  # kept for compatibility
                'elbo': elbo,  # sum over tokens, mean over batch
                'elbo_x': elbo,
            }
        return loss_dict


class MD4Loss(MaskedDiffusionLoss):
    """MD4 (Shi et al. 2024, https://arxiv.org/abs/2406.04329) Implementation"""

    def get_loss(self, logits, x_0, x_t, attention_mask, labels, loss_weight, **kwargs):
        # MD4 loss is the same as MaskedDiffusionLoss in this implementation,
        # potentially with different noise schedules or other configurations handled elsewhere.
        return super().get_loss(
            logits,
            x_0,
            x_t,
            attention_mask,
            labels,
            loss_weight,
            use_data_loss_elbo=self.use_data_loss_elbo,
            **kwargs,
        )


import numpy as np


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF using torch.erf (vectorized)."""
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


@torch.no_grad()
def precompute_T_mask_grid(
    K: int,
    lambda_: float,
    n_t: int = 10_000,
    n_gh: int = 64,
    device: torch.device | str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute T_mask(t'; lambda, K) on a uniform grid of t' in [0, 1].

    Returns:
        t_grid: shape [n_t], in [0, 1]
        T_grid: shape [n_t], where
                T_grid[i] ≈ P(mask | t' = t_grid[i]) = T_mask(t_grid[i])
    """
    device = torch.device(device)
    t_grid = torch.linspace(0.0, 1.0, n_t, device=device, dtype=dtype)

    # Avoid singularities at 0 and 1 for δ_0, δ_1
    eps = 1e-6
    t_clipped = t_grid.clamp(eps, 1.0 - eps)

    sigma = lambda_ * torch.sqrt(t_clipped * (1.0 - t_clipped))  # [n_t]
    delta0 = t_clipped / sigma  # [n_t]
    delta1 = (2.0 * t_clipped - 1.0) / sigma  # [n_t]

    # Gauss–Hermite nodes/weights from numpy
    x, w = np.polynomial.hermite.hermgauss(n_gh)  # nodes & weights for e^{-x^2}
    x = torch.tensor(x, device=device, dtype=dtype)  # [n_gh]
    w = torch.tensor(w, device=device, dtype=dtype)  # [n_gh]

    # Transform to standard normal: z = sqrt(2) * x, and
    # ∫ φ(z) f(z) dz ≈ (1 / sqrt(pi)) * Σ w_i f(z_i)
    z = x * np.sqrt(2.0)  # [n_gh]
    z = z.view(-1, 1)  # [n_gh, 1]
    delta0 = delta0.view(1, -1)  # [1, n_t]
    delta1 = delta1.view(1, -1)  # [1, n_t]

    Phi = _normal_cdf

    # term: Φ(z+δ0)^{K-2} Φ(z+δ1), shape [n_gh, n_t]
    term = Phi(z + delta0) ** (K - 2) * Phi(z + delta1)
    # Weighted sum over quadrature nodes
    T_grid = (1.0 / np.sqrt(np.pi)) * (w.view(-1, 1) * term).sum(dim=0)  # [n_t]

    # Fix endpoints exactly
    T_grid[0] = 0.0
    T_grid[-1] = 1.0

    return t_grid, T_grid


@torch.no_grad()
def invert_T_mask(
    gamma_bar: torch.Tensor,
    t_grid: torch.Tensor,
    T_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Invert T_mask approximately using a precomputed grid + linear interpolation.

    Args:
        gamma_bar: tensor of any shape, elements in [0, 1].
                   This is the discrete schedule \bar{γ}_t.
        t_grid:    shape [n_t], from precompute_T_mask_grid (increasing).
        T_grid:    shape [n_t], P(mask) values (increasing in t').

    Returns:
        t_prime: tensor with same shape as gamma_bar, such that
                 T_mask(t_prime) ≈ 1 - gamma_bar.
    """
    device = t_grid.device
    dtype = t_grid.dtype

    target = (1.0 - gamma_bar).to(device=device, dtype=dtype)  # desired mask prob
    T = T_grid  # [n_t]

    # Flatten for vectorized search
    target_flat = target.reshape(-1)

    # Clamp to [T[0], T[-1]] to avoid out-of-range issues
    target_flat = target_flat.clamp(T[0], T[-1])

    # Find indices i such that T[i-1] <= target < T[i]
    idx = torch.searchsorted(T, target_flat, right=False)
    idx = idx.clamp(min=1, max=T.numel() - 1)
    idx0 = idx - 1  # lower index

    T0 = T[idx0]  # [N]
    T1 = T[idx]  # [N]
    t0 = t_grid[idx0]
    t1 = t_grid[idx]

    # Linear interpolation weights
    denom = (T1 - T0).clamp_min(1e-12)
    w = (target_flat - T0) / denom  # in [0,1]

    t_prime_flat = t0 + w * (t1 - t0)  # [N]
    t_prime = t_prime_flat.view_as(gamma_bar)

    return t_prime


class DUOMaskedDiffusionLoss(MaskedDiffusionLoss):
    def __init__(
        self,
        tau_duo=1e-4,
        lambda_duo: float = 1.0,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.tau_duo = tau_duo
        self.lambda_duo = lambda_duo
        print_rank_zero('Assuming last token is mask token in DUO...')
        self.K = self.mask_token_id + 1
        # lazy initialization of the T_mask values because device and dtype are needed
        self.n_t = 10_000
        self.n_gh = 64
        self.t_grid, self.T_grid, self.t_prime = None, None, None

    @torch.no_grad()
    def sample_y_t_from_x0(
        self,
        x0: torch.Tensor,
        t_prime: torch.Tensor,
        K: int,
        lambda_: float,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Sample latent Y_t from the Brownian-bridge construction given discrete x0.

        Args:
            x0:      [B, S] integer tokens in {0, ..., K-1}.
                    We assume the mask token has index K-1.
            t_prime: scalar or tensor broadcastable to [B, S, 1],
                    continuous time values in [0, 1].
            K:       vocabulary size (including mask).
            lambda_: Brownian bridge noise scale (same λ as in T_mask).
            dtype:   optional dtype; if None, inferred from x0.

        Returns:
            y_t: [B, S, K] samples from N(μ_t^{(i)}, σ_t^2 I_K).
        """
        device = x0.device

        if dtype is None:
            dtype = torch.float32

        x0 = x0.to(device)
        B, S = x0.shape

        # One-hot encode x0 -> [B, S, K]
        x0_onehot = F.one_hot(x0, num_classes=K).to(dtype=dtype, device=device)

        # Mask basis vector e_K (index K-1)
        mask_vec = torch.zeros(K, device=device, dtype=dtype)
        mask_vec[-1] = 1.0
        # Broadcast to [1, 1, K] for broadcasting
        mask_vec = mask_vec.view(1, 1, K)

        # Prepare t' with shape [B, S, 1]
        t_prime = t_prime.to(device=device, dtype=dtype)
        while t_prime.dim() < 3:
            t_prime = t_prime.unsqueeze(-1)  # make it at least [..., 1]
        # Now broadcast to [B, S, 1]
        t_prime = t_prime.expand(B, S, 1)

        # Mean: μ_t^{(i)} = (1 - t') e_i + t' e_K
        mu = (1.0 - t_prime) * x0_onehot + t_prime * mask_vec  # [B, S, K]

        # Std: σ_t = λ sqrt(t'(1-t'))
        sigma2 = lambda_**2 * t_prime * (1.0 - t_prime)
        sigma = torch.sqrt(torch.clamp(sigma2, min=1e-12))  # [B, S, 1]

        # Gaussian noise
        eps = torch.randn_like(mu)  # [B, S, K]

        y_t = mu + sigma * eps  # broadcast σ over K

        return y_t

    def u_tau(
        self,
        x0: torch.Tensor,
        y_t: torch.Tensor,
        tau: float,
        K: int,
    ) -> torch.Tensor:
        """
        Compute the soft relaxation u_tau(x0, y_t) of shape [B, S, K].

        Args:
            x0:  [B, S] integer tokens in {0, ..., K-1}.
                We assume the mask token has index K-1.
            y_t: [B, S, K] latent Gaussian states (same as sample_y_t_from_x0 output).
            tau: temperature parameter > 0 for softmax.
            K:   vocabulary size (including mask).

        Returns:
            u: [B, S, K] relaxed masked state.
        """
        device = y_t.device
        dtype = y_t.dtype

        # One-hot representation of X_0: [B, S, K]
        x0_onehot = F.one_hot(x0.to(device), num_classes=K).to(dtype=dtype)

        # π_τ(Y_t) = softmax(Y_t / τ)
        pi_tau = torch.softmax(y_t / tau, dim=-1)  # [B, S, K]

        # g_τ(Y_t) = mask probability = last coordinate
        g_tau = pi_tau[..., -1:]  # [B, S, 1]

        # u_τ = g π_τ + (1 - g) X_0
        u = g_tau * pi_tau + (1.0 - g_tau) * x0_onehot  # [B, S, K]

        return u

    def forward(
        self,
        lightning_module: L.LightningModule,
        model: PreTrainedModel,
        batch: dict[str, Any],
        **model_kwargs,
    ) -> dict[str, torch.Tensor]:
        # get x_0
        x_0 = batch['input_ids']
        labels = batch.get('labels', x_0.detach().clone())
        attention_mask_0 = self.get_annealed_attention_mask(lightning_module, model, x_0)
        attention_mask_1 = batch.get('attention_mask')
        attention_mask_2 = torch.ones_like(x_0).long()
        # Use coalescing: prefer attention_mask_0, then attention_mask_1, then default to all ones
        if attention_mask_0 is not None:
            attention_mask = attention_mask_0
        elif attention_mask_1 is not None:
            attention_mask = attention_mask_1
        else:
            attention_mask = attention_mask_2
        # This is for SFT training: if labels are not provided, all tokens are maskable
        maskable_mask = labels != -100
        t = self.time_sampler(x_0.size(0)).to(x_0.device)
        alphas = self.noise_sampler(t)

        if self.t_grid is None:
            # compute T_mask grid once and for all
            # dtype is float32 for stability
            self.t_grid, self.T_grid = precompute_T_mask_grid(
                K=self.K,
                lambda_=self.lambda_duo,
                n_t=self.n_t,
                n_gh=self.n_gh,
                device=x_0.device,
                dtype=torch.float32,
            )
        self.t_prime = invert_T_mask(
            gamma_bar=alphas.alpha.float(),
            t_grid=self.t_grid,
            T_grid=self.T_grid,
        )

        y_t = self.sample_y_t_from_x0(
            x0=x_0,
            t_prime=self.t_prime,
            K=self.K,
            lambda_=self.lambda_duo,
            dtype=torch.float32,
        )

        x_t = self.u_tau(
            x0=x_0,
            y_t=y_t,
            tau=self.tau_duo,
            K=self.K,
        )

        assert model.config is not None, 'mask_token_id must be provided'
        assert hasattr(model.config, 'mask_token_id'), 'mask_token_id must be provided'
        assert model.config.mask_token_id == self.mask_token_id, 'Model mask_token_id does not match loss mask_token_id'
        if not maskable_mask.all():
            attention_mask = attention_mask * maskable_mask.to(dtype=attention_mask.dtype)

        model_output = self.model_forward(model, x_t, attention_mask, alphas, t, **model_kwargs)
        logits = self.extract_logits_from_output(model_output)
        # calculate the loss
        loss_dict = self.get_loss(
            logits=logits,
            x_0=x_0,
            x_t=x_t,
            attention_mask=attention_mask,
            labels=labels,
            loss_weight=alphas.loss_weight,
        )
        assert 'loss' in loss_dict, 'Loss key not found in the loss dictionary'
        return loss_dict

    def get_loss(
        self,
        logits,
        x_0,
        x_t,
        attention_mask,
        labels,
        loss_weight,
        use_data_loss_elbo=True,
        example_mask=None,
        **kwargs: dict,
    ):
        assert not self.shift_logits, 'shift_logits not supported in DUO loss'
        # Targets: prefer labels when provided (supports -100), else x_0
        targets = labels if labels is not None else x_0
        targets = targets.long()
        # --- numerically sensitive region in fp32 ---
        # with torch.autocast(device_type='cuda', enabled=False):
        logits = self.masked_process.subs_parameterization(
            x=x_t,
            x0_logits=logits, # logits.float(),
            mask_indices=None,  # we are not working on token space anymore
            return_probs=False,
        )
        logprobs = logits.gather(dim=-1, index=x_0[:, :, None]).squeeze(-1)
        loss_diff = (
            logprobs * loss_weight[:, None] * attention_mask if use_data_loss_elbo else -logprobs * attention_mask
        )
        nlls = logprobs * loss_weight[:, None] * attention_mask
        seqlen = attention_mask.sum(dim=-1)
        per_example_loss = loss_diff.sum(dim=-1) / seqlen
        per_example_elbo = nlls.sum(dim=-1)
        if example_mask is not None:
            example_weights = example_mask.to(device=per_example_loss.device, dtype=per_example_loss.dtype)
            num_examples = example_weights.sum().clamp_min(1.0)
            loss = (per_example_loss * example_weights).sum() / num_examples
            elbo = (per_example_elbo * example_weights).sum() / num_examples
        else:
            loss = per_example_loss.mean()
            elbo = per_example_elbo.mean()
        loss_dict = {
            'loss': loss,  # training objective
            'loss_x': loss,  # kept for compatibility
            'elbo': elbo,  # sum over tokens, mean over batch
            'elbo_x': elbo,
        }
        return loss_dict


class DistillMaskedDiffusionLoss(MaskedDiffusionLoss):
    def __init__(
        self,
        teacher_model_config: dict,
        distill_loss_weight: float = 1.0,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.teacher_model = instantiate_from_config_hf_pretrained(teacher_model_config).eval().requires_grad_(False)

        self.distill_loss_weight = distill_loss_weight

    def get_loss(self, logits, x_0, x_t, attention_mask, labels, loss_weight, **kwargs: dict):
        loss_dict = super().get_loss(
            logits,
            x_0,
            x_t,
            attention_mask,
            labels,
            loss_weight,
            **kwargs,
        )
        loss_diff = loss_dict.pop('loss')
        loss_dict.pop('bpc', None)
        loss_dict.pop('perplexity', None)

        loss_dict['loss_diff'] = loss_diff
        loss_distill = self.get_distill_loss(
            logits,
            x_0,
            x_t,
            attention_mask,
            labels,
            **kwargs,
        )
        loss_dict['loss_distill'] = loss_distill
        loss = loss_diff + self.distill_loss_weight * loss_distill
        loss_dict['loss'] = loss
        loss_dict['bpc'] = loss / math.log(2.0)
        return loss_dict

    def get_distill_loss(self, logits, x_0, x_t, attention_mask, labels, **kwargs):
        # forward KL
        # only compute loss on the masked tokens
        mask_indices = x_t == self.mask_token_id
        mask = (labels != -100).int() & mask_indices
        with torch.no_grad():
            outputs = self.teacher_model(x_0, attention_mask=attention_mask)
            logits_t = outputs.logits[:, :-1, :].contiguous()
        shift_logits = logits[:, 1:, :].contiguous()
        mask = mask[:, 1:].contiguous()
        inf_mask = torch.isinf(shift_logits)
        teacher_probs = F.softmax(logits_t, dim=-1, dtype=torch.float32)
        if teacher_probs.size(-1) != shift_logits.size(-1):
            # assume the model is extended with a mask token. So, we need to expand the teacher probs
            num_new_tokens = shift_logits.size(-1) - teacher_probs.size(-1)
            teacher_probs = torch.cat(
                [
                    teacher_probs,
                    torch.zeros(
                        *teacher_probs.shape[:2],
                        num_new_tokens,
                        device=teacher_probs.device,
                        dtype=teacher_probs.dtype,
                    ),
                ],
                dim=-1,
            )
        loss = teacher_probs * F.log_softmax(shift_logits, dim=-1, dtype=torch.float32)
        loss = torch.masked_fill(loss, inf_mask, 0.0)
        loss = loss.sum(dim=-1) * mask
        loss = -loss.sum() / mask.sum()
        return loss
