from math import sqrt

import torch


class UniformSampling:
    def __init__(self, low_discrepancy: bool = False, eps: float = 1e-4, effective_dim_shift_alpha: float | None = None):
        """
        Args:
            eps (float, optional): sampling epsilon. Defaults to 1e-3.
        """
        self.low_discrepancy = low_discrepancy
        self.eps = eps
        self.effective_dim_shift_alpha = effective_dim_shift_alpha
        self.shift_alpha = sqrt(effective_dim_shift_alpha / 4096) if effective_dim_shift_alpha is not None else None

    def sample_rand(self, num_samples: int):
        rand = torch.rand(num_samples)
        if self.low_discrepancy:
            offset = torch.arange(num_samples) / num_samples
            rand = (rand / num_samples + offset) % 1.0
        return rand

    def __call__(self, num_samples: int):
        t = (1 - self.eps) * self.sample_rand(num_samples) + self.eps
        if self.shift_alpha is not None and self.shift_alpha != 1.0:
            alpha = torch.tensor(self.shift_alpha, device=t.device, dtype=t.dtype)
            t = (alpha * t) / (1 + (alpha - 1) * t)
            t = t.clamp(min=self.eps, max=1.0 - self.eps)
        return t


class LowDiscrepancySampling(UniformSampling):
    """
    Low discrepancy sampling proposed in VDM (see Appendix I.1 in https://arxiv.org/abs/2107.00630) and used in MDLM (https://arxiv.org/abs/2406.07524).
    """

    def __init__(self, eps: float = 1e-3, effective_dim_shift_alpha: float | None = None):
        super().__init__(low_discrepancy=True, eps=eps, effective_dim_shift_alpha=effective_dim_shift_alpha)


class EDMSampling:
    """
    Inspired by EDM framework (https://arxiv.org/abs/2206.00364)
    """

    def __init__(self, p_mean=-1.2, p_std=1.2, sigma_data=0.5, eps=1e-7):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.eps = eps

    def __call__(self, n_samples: int):
        tau = self.p_mean + self.p_std * torch.randn((n_samples,))
        # lognormal distribution
        tau = tau.exp() / self.sigma_data
        # convert tau to t \in [0, 1]
        t = tau / (1.0 + tau)
        # Numerical safety
        return t.clamp(min=self.eps, max=1.0 - self.eps)
