import math
from abc import ABC, abstractmethod

import torch


class TimeDiscretization(ABC):
    def __init__(self, t_eps: float = 1e-5):
        self.t_eps = t_eps

    def linspace(self, num_inference_steps: int, reverse: bool = False) -> torch.Tensor:
        if reverse:
            return torch.linspace(1, self.t_eps, num_inference_steps + 1)
        return torch.linspace(self.t_eps, 1, num_inference_steps + 1)

    @abstractmethod
    def __call__(self, num_inference_steps: int, reverse: bool = False) -> torch.Tensor:
        pass


class UniformDiscretization(TimeDiscretization):
    def __call__(self, num_inference_steps: int, reverse: bool = False) -> torch.Tensor:
        return self.linspace(num_inference_steps, reverse)


class CosineDiscretization(TimeDiscretization):
    def __call__(self, num_inference_steps: int, reverse: bool = False) -> torch.Tensor:
        uniform = self.linspace(num_inference_steps, reverse)
        return torch.cos((1 - uniform) * 0.5 * math.pi)


class EDMDiscretization(TimeDiscretization):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(self, num_inference_steps: int, reverse: bool = False) -> torch.Tensor:
        ramp = self.linspace(num_inference_steps, not reverse)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        ts = sigmas / (1.0 + sigmas)
        # numerical safety
        ts = ts.clamp(min=self.t_eps, max=1.0)
        # make sure the first and last values are 1.0 and eps
        if reverse:
            ts[0] = 1.0
            # ts[-1] = self.t_eps
        else:
            # ts[0] = self.t_eps
            ts[-1] = 1.0
        return ts
