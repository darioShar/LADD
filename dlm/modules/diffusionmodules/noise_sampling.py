import math
from abc import ABC
from dataclasses import dataclass

import torch


@dataclass
class NoiseOutput:
    alpha: torch.Tensor
    dalpha: torch.Tensor | None
    loss_weight: torch.Tensor


class NoiseSampling(ABC):
    noise_type: str = 'diffusion'

    def __init__(self, eps: float | None = None):
        """
        Base class for noise sampling.

        Args:
            eps: Proposed in Shi et al. (2024) to ensure numerical stability. They use 1e-4.
        """
        self.eps = eps

    def _get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.eps is not None:
            return (1.0 - 2.0 * self.eps) * self._get_alpha(t) + self.eps
        return self._get_alpha(t)

    def _get_dalpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_dalpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.eps is not None:
            return (1.0 - 2.0 * self.eps) * self._get_dalpha(t)
        return self._get_dalpha(t)

    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        alpha_prime/(1-alpha), alpha_prime is the derivative of alpha w.r.t. t
        """
        return self.get_dalpha(t) / (1.0 - self.get_alpha(t))

    def __call__(self, t: torch.Tensor) -> NoiseOutput:
        return NoiseOutput(
            alpha=self.get_alpha(t),
            dalpha=self.get_dalpha(t),
            loss_weight=self.get_loss_weight(t),
        )


class IdentitySampling(NoiseSampling):
    def _get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _get_dalpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


"""
MD4's implementations of noise sampling.
"""


class LinearSampling(NoiseSampling):
    def _get_alpha(self, t):
        return 1 - t

    def _get_dalpha(self, t):
        return -torch.ones_like(t)


class GeometricSampling(NoiseSampling):
    def __init__(self, beta_min: float = 1e-5, beta_max: float = 20, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _get_alpha(self, t):
        return torch.exp(-(self.beta_min ** (1 - t)) * self.beta_max**t)

    def _get_dalpha(self, t):
        alpha = self.get_alpha(t)
        return alpha * self.beta_min ** (1 - t) * self.beta_max**t * math.log(self.beta_min / self.beta_max)


class PolynomialSampling(NoiseSampling):
    def __init__(self, degree: int = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree

    def _get_alpha(self, t):
        return 1 - t**self.degree

    def _get_dalpha(self, t):
        return -self.degree * t ** (self.degree - 1)

    def get_loss_weight(self, t):
        return -self.degree / t


class CosineSampling(NoiseSampling):
    def _get_alpha(self, t):
        return 1.0 - torch.cos(math.pi / 2.0 * (1.0 - t))

    def _get_dalpha(self, t):
        return -math.pi / 2.0 * torch.sin(math.pi / 2.0 * (1.0 - t))


"""
SEDD's implementations of noise sampling.
"""


class SigmaSampling(NoiseSampling):
    def _get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_dsigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._get_sigma(t))

    def get_dalpha(self, t: torch.Tensor) -> torch.Tensor:
        return None

    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        return -self._get_dsigma(t) / torch.expm1(self._get_sigma(t))


class SigmaGeometricSampling(SigmaSampling):
    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def _get_sigma(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t

    def _get_dsigma(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())


class SigmaLogLinearSampling(SigmaSampling):
    def _get_sigma(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def _get_dsigma(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)


NOISE_SAMPLINGS = {
    'linear': LinearSampling,
    'geometric': GeometricSampling,
    'cosine': CosineSampling,
    'polynomial': PolynomialSampling,
}


def get_noise_sampling(name: str, **kwargs) -> NoiseSampling:
    return NOISE_SAMPLINGS[name](**kwargs)


def plot_noise_sampling():
    import matplotlib.pyplot as plt

    t = torch.linspace(0.0, 1.0, 100)
    # plot the alpha and loss weight respectively
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for name, sampler in NOISE_SAMPLINGS.items():
        noise_output = sampler()(t)
        axes[0].plot(t, noise_output.alpha, label=name)
        axes[1].plot(t, noise_output.loss_weight, label=name)
    axes[0].legend()
    axes[0].set_title('alpha')
    axes[1].legend()
    axes[1].set_ylim(-20, 0)
    axes[1].set_title('loss weight')
    plt.show()


if __name__ == '__main__':
    plot_noise_sampling()
