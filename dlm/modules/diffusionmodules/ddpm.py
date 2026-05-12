import torch

from .continuous_noise_sampling import ContinuousNoiseSampling


class DDPMProcess:
    def __init__(
        self,
        continuous_noise_sampling: ContinuousNoiseSampling,
    ):
        self.cns = continuous_noise_sampling
    
    def y_0_to_eps(self, y_t, y_0, t):
        return (y_t - self.cns.alpha_bar(t) * y_0) / self.cns.sigma_bar(t)

    def eps_to_y_0(self, y_t, eps, t):
        return (y_t - self.cns.sigma_bar(t) * eps) / self.cns.alpha_bar(t)

    def sample_forward(
        self,
        y_0: torch.Tensor,
        t_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample y_t from y_0 and t_y.
        Args:
            y_0: Tensor of shape [batch_size, seq_len]
            t_y: [batch_size] Tensor containing the noise schedule parameters
        """
        z_t = torch.randn_like(y_0)
        return self.cns.alpha_bar(t_y[..., None]) * y_0 + self.cns.sigma_bar(t_y[..., None]) * z_t, z_t

    def sample_bridge_y_0_pred(
        self,
        y_t: torch.tensor,
        y_0_pred: torch.tensor,
        t: torch.tensor,
        s: torch.tensor,
    ) -> torch.tensor:
        """
        Sample from the bridge distribution p(y_s | y_t, y_0).
        Args:
            y_t: Current state (tensor of shape [batch_size, seq_len])
            y_0_logits: Predicted distribution over the vocabulary at t=0.
            t: Noise schedule alpha at time t
            s: Noise schedule alpha at time s, s < t
        """
        # Implement the bridge sampling logic here
        y_t_numerator = self.cns.alpha(t, s) * self.cns.sigma_bar_squared(s)
        y_t_denominator = self.cns.sigma_bar_squared(t)
        y_t_factor = y_t_numerator / y_t_denominator

        y_0_numerator = self.cns.alpha_bar(s) * self.cns.sigma_squared(t, s)
        y_0_denominator = self.cns.sigma_bar_squared(t)
        y_0_factor = y_0_numerator / y_0_denominator

        mean = y_t_factor * y_t + y_0_factor * y_0_pred
        std = torch.sqrt(torch.clamp(self.cns.sigma_tilde(t, s), min=0.0))
        return mean + std * torch.randn_like(y_t)

    def sample_last_step_y_0_pred(self, y_t: torch.tensor, y_0_pred: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        Sample the last step from the diffusion process.
        Args:
            y_t: Current state (tensor of shape [batch_size, seq_len])
            t: Noise schedule alpha at time t
        """
        return y_0_pred

    def sample_bridge_eps_pred(
        self,
        y_t: torch.tensor,
        eps_pred: torch.tensor,
        t: torch.tensor,
        s: torch.tensor,
    ) -> torch.tensor:
        """
        Sample from the bridge distribution p(y_s | y_t, eps_t(y_t, y_0)).
        """
        eps_factor = self.cns.sigma_squared(t, s) / self.cns.sigma_bar(t)
        mean = (y_t - eps_factor * eps_pred) / self.cns.alpha(t, s)
        std = torch.sqrt(torch.clamp(self.cns.sigma_tilde(t, s), min=0.0))
        return mean + std * torch.randn_like(y_t)

    def sample_bridge_eps_pred_ddim(
        self,
        y_t: torch.tensor,
        eps_pred: torch.tensor,
        t: torch.tensor,
        s: torch.tensor,
    ) -> torch.tensor:
        """Deterministic DDIM step using eps prediction."""
        y_0_pred = self.eps_to_y_0(y_t, eps_pred, t)
        return self.cns.alpha_bar(s) * y_0_pred + self.cns.sigma_bar(s) * eps_pred

    def sample_bridge_y_0_pred_ddim(
        self,
        y_t: torch.tensor,
        y_0_pred: torch.tensor,
        t: torch.tensor,
        s: torch.tensor,
    ) -> torch.tensor:
        """Deterministic DDIM step using y_0 prediction."""
        eps_pred = self.y_0_to_eps(y_t, y_0_pred, t)
        return self.cns.alpha_bar(s) * y_0_pred + self.cns.sigma_bar(s) * eps_pred

    def sample_last_step_eps_pred(
        self,
        y_t,
        eps_pred,
        t: torch.tensor,
    ) -> torch.tensor:
        """Sample the last step from the diffusion process."""
        # s = torch.ones_like(t) * 1e-5
        # eps_factor = self.cns.sigma_squared(t, s) / self.cns.sigma_bar(t)
        # return (y_t - eps_factor * eps_pred) / self.cns.alpha(t, s)
        return (y_t - self.cns.sigma_bar(t) * eps_pred) / self.cns.alpha_bar(t)
