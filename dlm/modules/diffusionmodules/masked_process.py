import torch
from transformers import PretrainedConfig

from .nonfinite_utils import summarize_nonfinite_tensor
from .noise_sampling import NoiseOutput
from .utils import sample_categorical


class MaskedDiffusionProcess:
    def __init__(
        self,
        mask_token_id,
        neg_infty=-1e6,
    ):
        self.mask_token_id = mask_token_id
        self.neg_infty = neg_infty

    def sample_forward(
        self,
        x_0: torch.Tensor,
        alphas: NoiseOutput,
        maskable_mask: torch.Tensor,
        model_config: PretrainedConfig | None = None,
    ) -> torch.Tensor:
        if self.mask_token_id is None:
            assert model_config is not None, 'mask_token_id must be provided'
            assert hasattr(model_config, 'mask_token_id'), 'mask_token_id must be provided'
            self.mask_token_id = model_config.mask_token_id
        assert self.mask_token_id != model_config.pad_token_id, 'mask_token_id must be different from pad_token_id'
        mask_probs = 1 - alphas.alpha
        mask_indices = (torch.rand(*x_0.shape, device=x_0.device) < mask_probs.unsqueeze(1)) & maskable_mask

        # return x_t
        return torch.where(mask_indices, torch.tensor(self.mask_token_id, device=x_0.device, dtype=x_0.dtype), x_0)

    def subs_parameterization(self, x, x0_logits, mask_indices, return_probs: bool = True):
        """https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py#L592"""
        x0_logits[..., self.mask_token_id] = self.neg_infty
        x0_logits = x0_logits - torch.logsumexp(x0_logits, dim=-1, keepdim=True)
        if mask_indices is not None:
            # predict probability one for unmasked tokens
            x0_logits[~mask_indices] = self.neg_infty
            x0_logits[~mask_indices, x[~mask_indices]] = 0.0
        if not return_probs:
            return x0_logits
        return x0_logits.exp()

    def _assert_finite_logits(self, x0_logits: torch.Tensor, context: str) -> None:
        if not torch.isfinite(x0_logits).all():
            stats = summarize_nonfinite_tensor(x0_logits)
            rank = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            prefix = f'[rank{rank}] ' if rank is not None else ''
            print(
                f'{prefix}Non-finite x0_logits in {context}: '
                f'shape={tuple(x0_logits.shape)} dtype={x0_logits.dtype} device={x0_logits.device} '
                f'mask_token_id={self.mask_token_id} '
                f'nonfinite={stats["num_total"] - stats["num_finite"]}/{stats["num_total"]} '
                f'nan={stats["num_nan"]} inf={stats["num_inf"]} '
                f'finite_min={stats["finite_min"]} finite_max={stats["finite_max"]} '
                f'finite_mean={stats["finite_mean"]}',
            )
            raise ValueError(f'Non-finite x0_logits detected in {context}')

    def sample_bridge(self, xt, x0_logits, alpha_t, alpha_s, shift_logits):
        """
        Sample from the bridge distribution p(x_s | x_t, x_0). x_0 can be a
        probability distribution over the vocabulary, and xt is the current state.

        Args:
            xt: Current state (tensor of shape [batch_size, seq_len])
            x0_pred: Predicted distribution over the vocabulary at t=0. Can be a Dirac
            alpha_t: Noise schedule alpha at time t
            alpha_s: Noise schedule alpha at time s

        """
        self._assert_finite_logits(x0_logits, 'MaskedDiffusionProcess.sample_bridge')
        # 1. Get mask indices
        mask_indices = xt == self.mask_token_id

        # 2. Get the probabilities at x_0
        x0_probs = self.subs_parameterization(xt, x0_logits, mask_indices, return_probs=True)

        # 3. compute probs at t-1
        move_chance_t = 1 - alpha_t
        move_chance_t_next = 1 - alpha_s
        next_probs = x0_probs * (move_chance_t - move_chance_t_next)
        next_probs[..., self.mask_token_id] = move_chance_t_next
        next_probs = next_probs / move_chance_t

        # 4. Sample x_{t-1}
        x_next = sample_categorical(next_probs)
        if shift_logits:
            x_next = torch.cat([xt[:, 0:1], x_next[:, :-1]], dim=1)
        return xt.masked_scatter(mask_indices, x_next[mask_indices])

    def sample_last_step(self, xt, x0_logits):
        """
        Sample the last step of the diffusion process.

        Args:
            xt: Current state (tensor of shape [batch_size, seq_len])
            x0_logits: Predicted distribution over the vocabulary at t=0.

        Returns:
            Sampled tensor for the last step.
        """
        self._assert_finite_logits(x0_logits, 'MaskedDiffusionProcess.sample_last_step')
        mask_indices = xt == self.mask_token_id
        x0_probs = self.subs_parameterization(xt, x0_logits, mask_indices, return_probs=True)
        return sample_categorical(x0_probs)
