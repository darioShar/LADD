import torch
from dataclasses import dataclass

@dataclass
class Output:
    """Output of the Latent Diffusion model."""

    logits: torch.Tensor | None = None
    y_pred: torch.Tensor | None = None
    y_std: torch.Tensor | None = None
