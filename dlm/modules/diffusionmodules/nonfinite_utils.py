from typing import Any

import torch


def summarize_nonfinite_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    finite_mask = torch.isfinite(tensor)
    num_total = tensor.numel()
    num_finite = int(finite_mask.sum().item())
    num_nan = int(torch.isnan(tensor).sum().item())
    num_inf = int(torch.isinf(tensor).sum().item())

    if num_finite == 0:
        return {
            'num_total': num_total,
            'num_finite': num_finite,
            'num_nan': num_nan,
            'num_inf': num_inf,
            'finite_min': None,
            'finite_max': None,
            'finite_mean': None,
        }

    pos_inf = torch.tensor(torch.inf, device=tensor.device, dtype=tensor.dtype)
    neg_inf = torch.tensor(-torch.inf, device=tensor.device, dtype=tensor.dtype)
    zeros = torch.zeros((), device=tensor.device, dtype=tensor.dtype)
    finite_min = torch.where(finite_mask, tensor, pos_inf).amin().item()
    finite_max = torch.where(finite_mask, tensor, neg_inf).amax().item()
    finite_sum = torch.where(finite_mask, tensor, zeros).sum(dtype=torch.float64).item()
    finite_mean = finite_sum / num_finite
    return {
        'num_total': num_total,
        'num_finite': num_finite,
        'num_nan': num_nan,
        'num_inf': num_inf,
        'finite_min': finite_min,
        'finite_max': finite_max,
        'finite_mean': finite_mean,
    }
