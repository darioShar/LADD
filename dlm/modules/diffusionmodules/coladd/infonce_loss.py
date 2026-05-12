"""InfoNCE contrastive loss for two-view invariance in latent diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with two-view invariance.

    This module implements InfoNCE loss for encouraging invariance between
    two corrupted views of the input. It includes:
    1. Masked mean pooling to compress sequence representations
    2. Cosine similarity with temperature scaling
    3. InfoNCE objective with in-batch negatives

    Args:
        temperature: Temperature parameter for InfoNCE (default: 0.1)
        normalize: Whether to L2-normalize embeddings before computing similarity
    """

    def __init__(
        self,
        temperature: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def masked_mean_pool(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool sequence embeddings using masked mean pooling.

        Args:
            embeddings: (B, S, D) tensor of sequence embeddings
            attention_mask: (B, S) tensor of attention mask (1 = keep, 0 = ignore)

        Returns:
            pooled: (B, D) tensor of pooled embeddings
        """
        if attention_mask is None:
            # Simple mean pooling without mask
            return embeddings.mean(dim=1)

        # Expand mask to match embedding dimension
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)

        # Masked sum
        masked_sum = (embeddings * mask_expanded).sum(dim=1)  # (B, D)

        # Count of non-masked tokens per sequence
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)

        # Masked mean
        pooled = masked_sum / mask_sum  # (B, D)

        return pooled

    def forward(
        self,
        embeddings_view1: torch.Tensor,
        embeddings_view2: torch.Tensor,
        attention_mask_view1: torch.Tensor | None = None,
        attention_mask_view2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute InfoNCE loss between two views.

        Args:
            embeddings_view1: (B, S, D) embeddings from first view
            embeddings_view2: (B, S, D) embeddings from second view
            attention_mask_view1: (B, S) attention mask for view 1
            attention_mask_view2: (B, S) attention mask for view 2

        Returns:
            dict with 'loss' (scalar) and 'logits' (B, B) similarity matrix
        """
        # Pool embeddings to (B, D)
        h1 = self.masked_mean_pool(embeddings_view1, attention_mask_view1)
        h2 = self.masked_mean_pool(embeddings_view2, attention_mask_view2)

        # L2 normalize if requested
        if self.normalize:
            h1 = F.normalize(h1, p=2, dim=-1)
            h2 = F.normalize(h2, p=2, dim=-1)

        # Compute cosine similarity matrix (B, B)
        # Each row i: similarity between h1[i] and all h2[j]
        logits = torch.matmul(h1, h2.t()) / self.temperature  # (B, B)

        # InfoNCE: positive pairs are on the diagonal
        # Labels are just indices [0, 1, 2, ..., B-1]
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Compute cross-entropy loss (InfoNCE)
        # Loss encourages logits[i, i] to be high and logits[i, j≠i] to be low
        loss = F.cross_entropy(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
        }
