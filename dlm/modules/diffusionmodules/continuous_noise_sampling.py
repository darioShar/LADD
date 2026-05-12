from abc import ABC

import torch


class ContinuousNoiseSampling(ABC):
    noise_type: str = 'continuous_diffusion'

    def __init__(self):
        """
        Base class for noise sampling.
        """
        return

    def alpha_bar_squared(self, t):
        raise NotImplementedError('alpha_bar_squared')

    def sigma_bar_squared(self, t):
        raise NotImplementedError('sigma_bar_squared')

    def alpha_bar(self, t):
        return torch.sqrt(self.alpha_bar_squared(t))

    def alpha_squared(self, t, s):
        """Consider that s < t"""
        return self.alpha_bar_squared(t) / self.alpha_bar_squared(s)

    def alpha(self, t, s):
        """Consider that s < t"""
        return torch.sqrt(self.alpha_squared(t, s))

    def sigma_bar(self, t):
        return torch.sqrt(self.sigma_bar_squared(t))

    def sigma_squared(self, t, s):
        """Consider that s < t"""
        return self.sigma_bar_squared(t) - self.alpha_squared(t, s) * self.sigma_bar_squared(s)

    def sigma(self, t, s):
        """Consider that s < t"""
        return torch.sqrt(self.sigma_squared(t, s))

    def sigma_tilde(self, t, s):
        """Consider that s < t"""
        return self.sigma_squared(t, s) * self.sigma_bar_squared(s) / self.sigma_bar_squared(t)

    def loss_weight_mean_pred(self, t, s):
        return 1 / (2 * self.sigma_tilde(t, s))

    def loss_weight_eps_pred(self, t, s):
        return self.sigma_squared(t, s) / (2 * self.sigma_bar_squared(s) * self.alpha_squared(t, s))

    def loss_weight_y_0_pred(self, t, s):
        return (
            self.alpha_bar_squared(s)
            * self.sigma_squared(t, s)
            / (2 * self.sigma_bar_squared(s) * self.sigma_bar_squared(t))
        )

class CosineContinuousNoiseSampling(ContinuousNoiseSampling):
    def __init__(
        self,
        eps=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        if self.eps is not None:
            self.schedule = lambda t: (1.0 - 2.0 * self.eps) * self.cosine_schedule(t) + self.eps
        else:
            self.schedule = self.cosine_schedule

    def cosine_schedule(self, t):
        s = 0.008
        return torch.cos((t + s) / (1 + s) * (torch.pi / 2)) ** 2

    def alpha_bar_squared(self, t):
        return self.schedule(t)

    def sigma_bar_squared(self, t):
        return 1 - self.schedule(t)


class SqrtContinuousNoiseSampling(ContinuousNoiseSampling):
    def __init__(
        self,
        eps=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        if self.eps is not None:
            self.schedule = lambda t: (1.0 - 2.0 * self.eps) * self.sqrt_schedule(t) + self.eps
        else:
            self.schedule = self.sqrt_schedule

    def sqrt_schedule(self, t):
        return torch.sqrt(1 - t)

    def alpha_bar_squared(self, t):
        return self.schedule(t)

    def sigma_bar_squared(self, t):
        return 1 - self.schedule(t)


class LinearContinuousNoiseSampling(ContinuousNoiseSampling):
    def __init__(
        self,
        eps=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        if self.eps is not None:
            self.schedule = lambda t: (1.0 - 2.0 * self.eps) * self.linear_schedule(t) + self.eps
        else:
            self.schedule = self.linear_schedule

    def linear_schedule(self, t):
        return 1 - t

    def alpha_bar_squared(self, t):
        return self.schedule(t)

    def sigma_bar_squared(self, t):
        return 1 - self.schedule(t)

class PolynomialContinuousNoiseSampling(ContinuousNoiseSampling):
    def __init__(
        self,
        eps=1e-4,
        power: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.power = power
        self.eps = eps
        if self.eps is not None:
            self.schedule = lambda t: (1.0 - 2.0 * self.eps) * self.polynomial_schedule(t) + self.eps
        else:
            self.schedule = self.polynomial_schedule

    def polynomial_schedule(self, t):
        return 1 - t ** self.power

    def alpha_bar_squared(self, t):
        return self.schedule(t)

    def sigma_bar_squared(self, t):
        return 1 - self.schedule(t)
