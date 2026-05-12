# dlm/transformers/vqvae/lucidrain_vector_quantizer.py

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def _ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def _ddp_broadcast_module_state_(module: nn.Module, src: int = 0) -> None:
    """
    Broadcast all tensors in module.state_dict() from src to all ranks.

    This is a blunt but reliable way to keep codebooks identical across ranks
    without relying on any internal collectives inside the quantizer.
    """
    if not _ddp_is_initialized():
        return

    state = module.state_dict()
    for _, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        if tensor.numel() == 0:
            continue
        dist.broadcast(tensor, src=src)


class LucidRainVectorQuantizer(nn.Module):
    """
    Wrapper around lucidrains/vector-quantize-pytorch quantizers.

    Key design:
      - Codebook size = (codebook_size - 1)
      - The final index (codebook_size - 1) is reserved for a learnable z_mask vector
        that bypasses quantization and is trained independently.

    Returns (to match existing interface):
      z_q_st, vq_loss, indices, one_hot

    Notes:
      - commitment_cost maps to lucidrains' commitment_weight
      - If you pass mask_positions, masked positions will be forced to:
          indices = mask_index (K_total - 1)
          z_q_st = z_mask (broadcasted)
        and excluded from quantizer updates via the quantizer's mask arg.
    """

    def __init__(
        self,
        codebook_size: int, # codebook size including mask token
        vqvae_latent_dim: int,
        commitment_cost: float = 0.25,
        # choose which lucidrains quantizer class to instantiate
        quantizer_type: str = 'VectorQuantize',  # VectorQuantize | ResidualVQ | GroupedResidualVQ | SimVQ
        # common VQ params
        decay: float = 0.95,
        codebook_dim: Optional[int] = None, # if activated, uses separate mlp for projected dim
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        use_cosine_sim: bool = False,
        rotation_trick: bool = False,
        threshold_ema_dead_code: Optional[float] = None,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_max_codes: int = 128,
        orthogonal_reg_active_codes_only: bool = False,
        # multi-headed VQ params
        heads: int = 1,
        separate_codebook_per_head: bool = False,
        accept_image_fmap: bool = False,
        # residual VQ params
        num_quantizers: int = 1,
        shared_codebook: bool = False,
        stochastic_sample_codes: bool = False,
        sample_codebook_temp: float = 0.1,
        # grouped residual VQ param
        groups: int = 1,
        # distributed behavior
        ddp_sync_mode: str = 'lucidrains',  # 'broadcast' | 'lucidrains' | 'none'
        sync_codebook: Optional[bool] = True,
        sync_kmeans: Optional[bool] = True,
        broadcast_src_rank: int = 0,
        broadcast_every: int = 1,
        # mask behavior
        require_quantizer_mask_support: bool = True,
    ) -> None:
        super().__init__()

        if codebook_size < 2:
            raise ValueError('codebook_size must be >= 2 (1+ codes plus 1 mask token).')

        self.codebook_size = int(codebook_size)
        self.vqvae_latent_dim = int(vqvae_latent_dim)
        self.commitment_cost = float(commitment_cost)

        assert _ddp_is_initialized() or ddp_sync_mode == 'none', (
            f"ddp_sync_mode cannot be 'broadcast' or 'lucidrains' when DDP is not initialized. Got {ddp_sync_mode}"
        )
        self.ddp_sync_mode = ddp_sync_mode
        if (self.ddp_sync_mode == 'broadcast') or (self.ddp_sync_mode == 'none'):
            sync_codebook = False
            sync_kmeans = False
        self.broadcast_src_rank = int(broadcast_src_rank)
        self.broadcast_every = int(broadcast_every)
        self.require_quantizer_mask_support = bool(require_quantizer_mask_support)

        # learned mask vector trained independently of the codebook
        self.z_mask = nn.Parameter(torch.zeros(self.vqvae_latent_dim))

        # track steps (for broadcast cadence)
        self.register_buffer('_vq_step', torch.zeros((), dtype=torch.long), persistent=False)

        # instantiate lucidrains quantizer
        from vector_quantize_pytorch import (
            GroupedResidualVQ,
            ResidualVQ,
            SimVQ,
            VectorQuantize,
        )

        quantizer_cls_map = {
            'VectorQuantize': VectorQuantize,
            'ResidualVQ': ResidualVQ,
            'GroupedResidualVQ': GroupedResidualVQ,
            'SimVQ': SimVQ,
        }

        if quantizer_type not in quantizer_cls_map:
            raise ValueError(
                f"Unknown quantizer_type='{quantizer_type}'. Expected one of {sorted(quantizer_cls_map.keys())}."
            )

        self.quantizer_type = quantizer_type
        quantizer_cls = quantizer_cls_map[quantizer_type]

        # codebook excludes the mask token
        codebook_size = self.codebook_size - 1

        # Build kwargs based on quantizer type - NO FILTERING, so wrong args will fail
        # Common args for all quantizer types
        init_kwargs: Dict[str, Any] = {
            'dim': self.vqvae_latent_dim,
            'codebook_size': codebook_size,
        }

        # VectorQuantize specific args
        if quantizer_type == 'VectorQuantize':
            init_kwargs.update(
                {
                    'decay': decay,
                    'commitment_weight': self.commitment_cost,
                    'kmeans_init': kmeans_init,
                    'kmeans_iters': kmeans_iters,
                    'use_cosine_sim': use_cosine_sim,
                    'rotation_trick': rotation_trick,
                    'orthogonal_reg_weight': orthogonal_reg_weight,
                    'orthogonal_reg_max_codes': orthogonal_reg_max_codes,
                    'orthogonal_reg_active_codes_only': orthogonal_reg_active_codes_only,
                    'heads': heads,
                    'separate_codebook_per_head': separate_codebook_per_head,
                    'accept_image_fmap': accept_image_fmap,
                    'stochastic_sample_codes': stochastic_sample_codes,
                    'sample_codebook_temp': sample_codebook_temp,
                }
            )
            if codebook_dim is not None:
                init_kwargs['codebook_dim'] = int(codebook_dim)
            if threshold_ema_dead_code is not None:
                init_kwargs['threshold_ema_dead_code'] = float(threshold_ema_dead_code)
            if sync_codebook is not None:
                init_kwargs['sync_codebook'] = bool(sync_codebook)
            if sync_kmeans is not None:
                init_kwargs['sync_kmeans'] = bool(sync_kmeans)

        # ResidualVQ and GroupedResidualVQ specific args
        elif quantizer_type in ['ResidualVQ', 'GroupedResidualVQ']:
            init_kwargs.update(
                {
                    'num_quantizers': num_quantizers,
                    'shared_codebook': shared_codebook,
                    'heads': heads,
                    'accept_image_fmap': accept_image_fmap,
                }
            )
            if codebook_dim is not None:
                init_kwargs['codebook_dim'] = int(codebook_dim)

            # GroupedResidualVQ also needs groups
            if quantizer_type == 'GroupedResidualVQ':
                init_kwargs['groups'] = groups

        # SimVQ specific args (minimal for now, extend as needed)
        elif quantizer_type == 'SimVQ':
            init_kwargs.update(
                {
                    'decay': decay,
                    'commitment_weight': self.commitment_cost,
                }
            )
            if codebook_dim is not None:
                init_kwargs['codebook_dim'] = int(codebook_dim)

        # Instantiate without filtering - will fail if wrong args are passed
        self.vqvae = quantizer_cls(**init_kwargs)

        # stats cache
        self._last_stats: Dict[str, Any] = {}

    @property
    def mask_index(self) -> int:
        return self.codebook_size - 1

    def get_last_stats(self) -> Dict[str, Any]:
        """
        Returns last forward's lightweight diagnostics.
        """
        return dict(self._last_stats)

    def _compute_stats(
        self,
        indices: torch.Tensor,
        vq_loss: Union[torch.Tensor, float],
    ) -> None:
        """
        Compute lightweight stats on code usage (excluding mask token).

        Updates self._last_stats with:
          - 'prop_top_1_token': proportion of assignments to most used code (excl. mask)
          - 'vq_loss': float value of vq_loss
        """
        with torch.no_grad():
            k_no_mask = self.codebook_size - 1
            flat_idx = indices.reshape(-1) if indices.ndim > 1 else indices

            valid = flat_idx != self.mask_index
            used = flat_idx[valid]
            if used.numel() > 0:
                counts = torch.bincount(used.clamp(min=0, max=k_no_mask - 1), minlength=k_no_mask).float()
                prop_top_1 = (counts.max() / counts.sum()).item() if counts.sum() > 0 else 0.0
            else:
                prop_top_1 = 0.0

            self._last_stats = {
                'prop_top_1_token': prop_top_1,
                'vq_loss': float(vq_loss.detach().item()) if torch.is_tensor(vq_loss) else float(vq_loss),
            }

    def _ddp_synchronization_step(self) -> None:
        """
        Perform DDP synchronization step if configured.
        """
        if not _ddp_is_initialized():
            return

        if self.training and _ddp_is_initialized():
            self._vq_step += 1

            if self.ddp_sync_mode == 'broadcast':
                if (int(self._vq_step.item()) % self.broadcast_every) == 0:
                    _ddp_broadcast_module_state_(self.vqvae, src=self.broadcast_src_rank)

            elif self.ddp_sync_mode == 'lucidrains':
                # Rely on lucidrains internal collectives (sync_codebook/sync_kmeans).
                # Nothing to do here.
                pass

            elif self.ddp_sync_mode == 'none':
                # No syncing (expect codebooks to diverge across ranks).
                pass

            else:
                raise ValueError(
                    f"Unknown ddp_sync_mode='{self.ddp_sync_mode}'. Expected 'broadcast', 'lucidrains', or 'none'."
                )

    def codebook(self) -> torch.Tensor:
        return self.vqvae.codebook
    
    def forward(
        self,
        z_e: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
        return_one_hot: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z_e: encoder outputs, shape (B, ..., D)
            mask_positions: boolean tensor, shape (B, ...), True where you want to use z_mask
            return_one_hot: whether to return one-hot assignments

        Returns:
            z_q_st: quantized outputs with straight-through behavior
            vq_loss: commitment + any aux losses returned by lucidrains
            indices: code indices, shape (B, ...)
            one_hot: one-hot encodings over K_total=codebook_size (mask token is last)
        """
        # Store original shape and flatten to (N, D)
        z_e_shape = z_e.shape
        assert z_e_shape[-1] == self.vqvae_latent_dim, f'Expected last dim={self.vqvae_latent_dim}, got {z_e_shape[-1]}'
        flat_z_e = z_e.reshape(-1, self.vqvae_latent_dim)  # (N, D)

        # Flatten mask if provided
        flat_mask: Optional[torch.Tensor] = None
        if mask_positions is not None:
            assert mask_positions.shape == z_e_shape[:-1], f'mask_positions shape {mask_positions.shape} != {z_e_shape[:-1]}'
            flat_mask_positions = mask_positions.reshape(-1).bool()
            flat_valid_mask = ~flat_mask_positions

            # Check if quantizer supports mask argument
            vq_forward = self.vqvae.forward
            supports_mask = 'mask' in inspect.signature(vq_forward).parameters

            if not supports_mask and self.require_quantizer_mask_support:
                raise RuntimeError(
                    'mask_positions was provided, but lucidrains quantizer forward() '
                    "does not accept a 'mask' argument. "
                    'Either upgrade vector-quantize-pytorch or set require_quantizer_mask_support=False.'
                )
            if supports_mask:
                flat_mask = flat_valid_mask

        # Call lucidrains quantizer
        if flat_mask is not None:
            out = self.vqvae(flat_z_e, mask=flat_mask)
        else:
            out = self.vqvae(flat_z_e)

        # Parse output: (quantized, indices, loss)
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            raise RuntimeError(
                'Unexpected output from lucidrains quantizer. Expected at least 3-tuple: (quantized, indices, loss).'
            )

        flat_z_q : torch.Tensor = out[0]  # (N, D)
        flat_indices : torch.Tensor = out[1]  # (N,)
        loss : torch.Tensor = out[2]
        
        # compute loss mean
        vq_loss = loss.mean()

        # Apply mask vector to masked positions
        if mask_positions is not None:
            flat_mask_positions = mask_positions.reshape(-1).bool()
            flat_z_q = torch.where(flat_mask_positions.unsqueeze(-1), self.z_mask.unsqueeze(0).expand_as(flat_z_q), flat_z_q)
            flat_indices = torch.where(flat_mask_positions, torch.full_like(flat_indices, self.mask_index), flat_indices)

        # Reshape back to original shape
        z_q = flat_z_q.reshape(z_e_shape)  # (B, ..., D)
        indices = flat_indices.reshape(*z_e_shape[:-1])  # (B, ...)

        # Straight-through (lucidrains already handles this internally, but keep interface)
        z_q_st = z_q

        # One-hot encoding
        one_hot: Optional[torch.Tensor] = None
        if return_one_hot:
            k_total = self.codebook_size
            one_hot = F.one_hot(indices.clamp(min=0, max=k_total - 1), num_classes=k_total)

        # Compute stats
        self._compute_stats(indices, vq_loss)

        # DDP synchronization
        self._ddp_synchronization_step()

        return z_q_st, vq_loss, indices, one_hot
