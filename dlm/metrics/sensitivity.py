from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchmetrics import Metric

if TYPE_CHECKING:
    from collections.abc import Callable


@torch.no_grad()
def self_position_distance(
    x_t: torch.Tensor,
    alpha_t: torch.Tensor,
    base_logits: torch.Tensor,
    forward_logits: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    positions_per_sample: int,
    perturbation_trials: int,
    temperature: float = 1.0,
    valid_position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate average TV distance induced by perturbing each token at its own position.

    This generalizes the simple `model(xt, t).logits` interface by accepting `forward_logits`,
    a callable that maps a perturbed `x_t` and 'alpha_t' tensor to logits with shape `(B, S, V)`.
    """
    expected_x_rank = 2
    expected_logits_rank = 3
    if x_t.ndim != expected_x_rank:
        raise ValueError(f'Expected x_t to have shape (B, S), got {tuple(x_t.shape)}')
    if base_logits.ndim != expected_logits_rank:
        raise ValueError(f'Expected base_logits to have shape (B, S, V), got {tuple(base_logits.shape)}')
    if base_logits.shape[:2] != x_t.shape:
        raise ValueError(f'base_logits shape {tuple(base_logits.shape[:2])} does not match x_t shape {tuple(x_t.shape)}')

    batch_size, seq_len = x_t.shape
    vocab_size = int(base_logits.size(-1))
    if seq_len == 0 or vocab_size <= 1:
        return base_logits.new_tensor(0.0, dtype=torch.float32)

    if valid_position_mask is None:
        valid_mask = torch.ones_like(x_t, dtype=torch.bool)
    else:
        if valid_position_mask.shape != x_t.shape:
            raise ValueError(
                f'valid_position_mask shape {tuple(valid_position_mask.shape)} does not match x_t shape {tuple(x_t.shape)}',
            )
        valid_mask = valid_position_mask.bool()

    positions_per_sample = max(1, min(int(positions_per_sample), seq_len))
    perturbation_trials = max(1, int(perturbation_trials))
    temperature = max(float(temperature), 1e-8)

    sampling_probs = valid_mask.float()
    no_valid_positions = sampling_probs.sum(dim=1) <= 0
    if torch.all(no_valid_positions):
        return base_logits.new_tensor(0.0, dtype=torch.float32)
    sampling_probs[no_valid_positions, 0] = 1.0
    sampling_probs = sampling_probs / sampling_probs.sum(dim=1, keepdim=True)

    total_distance = base_logits.new_tensor(0.0, dtype=torch.float32)
    for _ in range(perturbation_trials):
        positions = torch.multinomial(sampling_probs, num_samples=positions_per_sample, replacement=True)
        selected_valid = torch.gather(valid_mask, dim=1, index=positions).float()

        gather_index = positions.unsqueeze(-1).expand(-1, -1, vocab_size)
        base_selected = torch.gather(base_logits, dim=1, index=gather_index)

        x_t_perturbed = x_t.clone()
        old_tokens = torch.gather(x_t_perturbed, dim=1, index=positions)
        old_tokens_mod = old_tokens.remainder(vocab_size)
        token_delta = torch.randint(1, vocab_size, old_tokens.shape, device=x_t.device)
        new_tokens = (old_tokens_mod + token_delta) % vocab_size
        x_t_perturbed.scatter_(dim=1, index=positions, src=new_tokens)

        logits_perturbed = forward_logits(x_t_perturbed, alpha_t)
        if logits_perturbed.shape != base_logits.shape:
            raise ValueError(
                f'forward_logits returned shape {tuple(logits_perturbed.shape)}, expected {tuple(base_logits.shape)}',
            )
        pert_selected = torch.gather(logits_perturbed, dim=1, index=gather_index)

        distance = 0.5 * torch.abs(
            (base_selected.float() / temperature).softmax(dim=-1)
            - (pert_selected.float() / temperature).softmax(dim=-1),
        ).sum(dim=-1)
        weighted_distance = (distance * selected_valid).sum() / selected_valid.sum().clamp_min(1.0)
        total_distance += weighted_distance.to(total_distance.dtype)

    return total_distance / float(perturbation_trials)


class SensitivityMetric(Metric):
    """Running mean of scalar sensitivity values."""

    def __init__(self):
        super().__init__()
        self.add_state('sum_vals', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, value: torch.Tensor | float) -> None:
        val = torch.as_tensor(value, dtype=torch.float32, device=self.sum_vals.device)
        if val.ndim > 0:
            val = val.mean()
        self.sum_vals += val
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.sum_vals / self.count.clamp_min(1)
