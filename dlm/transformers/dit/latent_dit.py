"""Copied from https://github.com/kuleshov-group/mdlm/blob/master/models/dit.py"""

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import PreTrainedModel

from .configuration_latent_dit import LatentDiTConfig

# Import shared components from dit.py to avoid redundancy
from .dit import (
    _HAS_FLASH_ATTN,
    DDiTBlock,
    DDitFinalLayer,
    EmbeddingLayer,
    LayerNorm,
    Rotary,
    TimestepEmbedder,
    apply_rotary_pos_emb,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_inference,
    bias_dropout_add_scale_train,
    flash_attn,
    get_bias_dropout_add_scale,
    modulate,
    modulate_fused,
)
from .hybrid_flash import HybridFlashAttention
from .output import Output


@torch.no_grad()
def _init_linear_zero_(layer: nn.Linear) -> nn.Linear:
    """Zero-out a Linear layer so it outputs exactly 0 at init."""
    layer.weight.data.zero_()
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


# def _build_joint_attention_mask(
#     attention_mask: torch.Tensor | None,
#     y_seq_len: int,
# ) -> torch.Tensor | None:
#     if attention_mask is None:
#         return None
#     if attention_mask.dtype != torch.bool:
#         attention_mask = attention_mask != 0
#     y_mask = torch.ones(
#         (attention_mask.size(0), y_seq_len),
#         device=attention_mask.device,
#         dtype=torch.bool,
#     )
#     joint_mask = torch.cat([attention_mask, y_mask], dim=1)
#     return ~joint_mask[:, None, None, :]

def _build_joint_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    x_seq_len: int,
    y_seq_len: int,
    device: torch.device,
    keep_y: bool = True,
) -> torch.Tensor:
    # 1. Check if we actually have padding in X
    x_has_padding = False
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
             # Check if there are any zeros
            if (attention_mask == 0).any():
                x_has_padding = True
        elif (~attention_mask).any():
            x_has_padding = True

    # 2. If no padding in X, and we keep all Y, return None
    # This triggers the fast paths in SDPA and Flash
    if not x_has_padding and keep_y:
        return None

    # x_mask: True = keep token
    if attention_mask is None:
        x_mask = torch.ones((batch_size, x_seq_len), device=device, dtype=torch.bool)
    else:
        x_mask = attention_mask.to(device=device)
        if x_mask.dtype != torch.bool:
            x_mask = x_mask != 0
        # sanity: ensure correct shape
        assert x_mask.shape[0] == batch_size and x_mask.shape[1] == x_seq_len, (x_mask.shape, batch_size, x_seq_len)

    if keep_y:
        y_mask = torch.ones((batch_size, y_seq_len), device=device, dtype=torch.bool)
    else:
        y_mask = torch.zeros((batch_size, y_seq_len), device=device, dtype=torch.bool)

    joint_keep = torch.cat([x_mask, y_mask], dim=1)  # (B, x+y), True=keep
    return joint_keep

class WideTransformerHead(nn.Module):
    """Lightweight wide head stacked after the DiT backbone."""

    def __init__(
        self,
        in_dim: int,
        head_dim: int,
        num_layers: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float,
        dropout: float,
        attention_backend: str = 'auto',
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, head_dim) if in_dim != head_dim else nn.Identity()
        self.rotary_emb = Rotary(head_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    head_dim,
                    num_heads,
                    cond_dim,
                    mlp_ratio,
                    dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ],
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(x)
        rope = self.rotary_emb(x)
        for block in self.blocks:
            x = block(x, rope, c, attention_mask=attention_mask)
        return x


class MM_WideTransformerHead(nn.Module):
    """Multi-modal Lightweight wide head stacked after the DiT backbone."""

    def __init__(
        self,
        in_dim: int,
        head_dim: int,
        num_layers: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float,
        dropout: float,
        attention_backend: str = 'auto',
        use_rope_for_latents: bool = True,
    ):
        super().__init__()
        self.in_proj_x = nn.Linear(in_dim, head_dim) if in_dim != head_dim else nn.Identity()
        self.in_proj_y = nn.Linear(in_dim, head_dim) if in_dim != head_dim else nn.Identity()
        self.rotary_emb = Rotary(head_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                MM_DDiTBlock(
                    head_dim,
                    num_heads,
                    cond_dim,
                    mlp_ratio,
                    dropout,
                    attention_backend=attention_backend,
                    use_rope_for_latents=use_rope_for_latents,
                )
                for _ in range(num_layers)
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj_x(x)
        y = self.in_proj_y(y)
        joint_embeds = torch.cat([x, y], dim=1)
        rope = self.rotary_emb(joint_embeds)
        for block in self.blocks:
            x, y = block(x, y, rope, c, attn_mask=attn_mask)
        return x, y


class DiTPretrainedModel(PreTrainedModel):
    """Base class for DiT models."""

    config_class = LatentDiTConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['DDiTBlock']
    _supports_flash_attn_2 = _HAS_FLASH_ATTN
    _supports_sdpa = False

    def __init__(self, *inputs, **kwargs) -> None:
        """Initialize the DiTPretrainedModel."""
        super().__init__(*inputs, **kwargs)


#################################################################################
#                                 MM-DDiTBlock                              #
#################################################################################


class MM_DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4,
        dropout: float = 0.1,
        attention_backend: str = 'auto',
        use_rope_for_latents: bool = True,  # If False, latents use learned slot embeddings only
    ):
        """Initialize the MM_DDiTBlock.

        :param dim: The dimension of the embedding.
        :param n_heads: The number of attention heads.
        :param cond_dim: The dimension of the conditioning vector.
        :param use_rope_for_latents: If True, apply RoPE to latents (default). If False, latents use learned absolute positions only.
        """
        super().__init__()
        self.n_heads = n_heads
        self.use_rope_for_latents = use_rope_for_latents

        # Always use separate QKV matrices for each modality
        self.qkv_x = nn.Linear(dim, 3 * dim, bias=False)
        self.qkv_y = nn.Linear(dim, 3 * dim, bias=False)

        # Separate output projections for each modality
        self.attn_out_x = nn.Linear(dim, dim, bias=False)
        self.attn_out_y = nn.Linear(dim, dim, bias=False)

        # Choose attention mechanism based on positional encoding strategy
        if not use_rope_for_latents:
            # Use hybrid flash attention with split positional encoding
            self.hybrid_attn = HybridFlashAttention(dim, n_heads)
        else:
            # Standard unified attention (RoPE on both streams)
            self.hybrid_attn = None

        self.dropout = dropout

        # x modality
        self.norm1_x = LayerNorm(dim)

        self.norm2_x = LayerNorm(dim)
        self.mlp_x = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim), bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(int(mlp_ratio * dim), dim, bias=True),
        )

        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),  # added SiLU activation
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        self.adaLN_modulation_x[1].weight.data.zero_()
        self.adaLN_modulation_x[1].bias.data.zero_()


        # y modality
        self.norm1_y = LayerNorm(dim)
        if self.use_rope_for_latents:
            # attn_out_y already defined above for standard mode
            pass
        # else: attn_out_y handled by hybrid_attn

        self.norm2_y = LayerNorm(dim)
        self.mlp_y = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim), bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(int(mlp_ratio * dim), dim, bias=True),
        )


        self.adaLN_modulation_y = nn.Sequential(
            nn.SiLU(),  # added SiLU activation
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        self.adaLN_modulation_y[1].weight.data.zero_()
        self.adaLN_modulation_y[1].bias.data.zero_()
        self.gradient_checkpointing = False
        self.attention_backend = attention_backend
        self._attn_fn = self._resolve_attention_backend()


    def _get_modulate(self):
        if torch.cuda.is_available():
            return modulate_fused
        return modulate

    def _get_bias_dropout_scale(self):
        if getattr(self, 'gradient_checkpointing', False):
            if self.training:
                return bias_dropout_add_scale_train
            return bias_dropout_add_scale_inference
        if not torch.cuda.is_available():
            if self.training:
                return bias_dropout_add_scale_train
            return bias_dropout_add_scale_inference
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def _resolve_attention_backend(self):
        backend = self.attention_backend
        if backend == 'flash':
            if not _HAS_FLASH_ATTN:
                raise RuntimeError('attention_backend=flash requested, but flash-attn is not available')
            return self._flash_attention
        if backend == 'sdpa':
            return self._sdpa_attention
        if backend == 'auto':
            if _HAS_FLASH_ATTN:
                return self._flash_attention
            return self._sdpa_attention
        raise ValueError(f'Unknown attention_backend: {backend}')

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the DDiTBlock.

        :param x: The input tensor.
        :param rotary_cos_sin: The rotary cosine and sine tensors.
        :param c: The conditioning vector.
        :param attn_mask: The attention mask.
        :param seqlens: The sequence lengths.
        :return: The output tensor.
        """
        batch_size, seq_len_x, dim = x.shape[0], x.shape[1], x.shape[2]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        modulate_fn = self._get_modulate()

        (shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x) = (
            self.adaLN_modulation_x(c)[:, None].chunk(6, dim=2)
        )

        (shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y) = (
                self.adaLN_modulation_y(c)[:, None].chunk(6, dim=2)
            )

        # attention operation
        x_skip = x
        x_norm = modulate_fn(self.norm1_x(x), shift_msa_x, scale_msa_x)

        y_skip = y
        y_norm = modulate_fn(self.norm1_y(y), shift_msa_y, scale_msa_y)
        seq_len_y = y_norm.shape[1]

        # Hybrid positional encoding strategy:
        # - Text stream (x): Apply RoPE to preserve relative positional information
        # - Latent stream (y): Skip RoPE (hybrid) or also apply RoPE (standard)

        # Project each modality separately with its own QKV matrix
        qkv_x = self.qkv_x(x_norm)
        qkv_y = self.qkv_y(y_norm)

        # Reshape to (B, S, 3, H, D)
        qkv_x = rearrange(qkv_x, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        qkv_y = rearrange(qkv_y, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        cos, sin = rotary_cos_sin

        if self.use_rope_for_latents:
            # Standard path: Apply RoPE uniformly to both streams
            # Concatenate and apply RoPE to joint sequence
            qkv_joint = torch.cat([qkv_x, qkv_y], dim=1)
            qkv_joint = apply_rotary_pos_emb(
                qkv_joint,
                cos,
                sin,
                use_flash_attn=self.attention_backend != 'sdpa',
            )
            joint_embeds = self._attn_fn(qkv_joint, attn_mask, batch_size, seq_len_x + seq_len_y, dim)
            x = joint_embeds[:, :seq_len_x, :]
            y = joint_embeds[:, seq_len_x:, :]
        elif self.attention_backend == 'sdpa':
            qkv_x = apply_rotary_pos_emb(
                qkv_x,
                cos[:, :seq_len_x],
                sin[:, :seq_len_x],
                use_flash_attn=False,
            )
            qkv_joint = torch.cat([qkv_x, qkv_y], dim=1)
            joint_attn_mask = None
            if attn_mask is not None:
                y_attn_mask = torch.ones(
                    (batch_size, seq_len_y),
                    device=attn_mask.device,
                    dtype=attn_mask.dtype,
                )
                joint_attn_mask = torch.cat([attn_mask, y_attn_mask], dim=1)
            joint_embeds = self._sdpa_attention(qkv_joint, joint_attn_mask, batch_size, seq_len_x + seq_len_y, dim)
            x = joint_embeds[:, :seq_len_x, :]
            y = joint_embeds[:, seq_len_x:, :]
        else:
            # Hybrid path - Pass QKV and RoPE components directly
            # Note: We pass raw qkv_x/qkv_y. Hybrid module handles specific rotations internally.
            x, y = self.hybrid_attn(
                qkv_text=qkv_x,
                qkv_latent=qkv_y,
                rope_cos=cos,
                rope_sin=sin,
                attn_mask=attn_mask,
            )

        # x: attn out and mlp operation
        x = bias_dropout_scale_fn(
            self.attn_out_x(x),
            None,
            gate_msa_x,
            x_skip,
            self.dropout,
        )
        x = bias_dropout_scale_fn(
            self.mlp_x(modulate_fn(self.norm2_x(x), shift_mlp_x, scale_mlp_x)),
            None,
            gate_mlp_x,
            x,
            self.dropout,
        )

        # y: attn out and mlp operation
        y = bias_dropout_scale_fn(
            self.attn_out_y(y),
            None,
            gate_msa_y,
            y_skip,
            self.dropout,
        )
        y = bias_dropout_scale_fn(
            self.mlp_y(modulate_fn(self.norm2_y(y), shift_mlp_y, scale_mlp_y)),
            None,
            gate_mlp_y,
            y,
            self.dropout,
        )

        return x, y

    def _flash_attention(self, qkv, attn_mask, batch_size, seq_len, dim):
        if attn_mask is None:
            qkv_flat = rearrange(qkv, 'b s three h d -> (b s) three h d').contiguous()
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                device=qkv.device,
                dtype=torch.int32,
            )
            out = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv_flat,
                cu_seqlens,
                seq_len,
                0.0,
                causal=False,
            )
            return rearrange(out, '(b s) h d -> b s (h d)', b=batch_size, s=seq_len)
        if attn_mask.dim() != 2:
            raise ValueError('Flash attention only supports None or 2D padding masks')
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask != 0
        seqlens = attn_mask.sum(dim=1, dtype=torch.int32)
        qkv_flat = rearrange(qkv, 'b s three h d -> (b s) three h d').contiguous()
        mask = attn_mask.reshape(-1)
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        qkv_unpad = qkv_flat.index_select(0, indices)
        cu_seqlens = torch.zeros(batch_size + 1, device=qkv.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
        max_seqlen = int(seqlens.max().item())
        out = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens,
            max_seqlen,
            0.0,
            causal=False,
        )
        padded = torch.zeros(
            batch_size * seq_len,
            self.n_heads,
            qkv.shape[-1],
            device=qkv.device,
            dtype=out.dtype,
        )
        padded.index_copy_(0, indices, out)
        return rearrange(padded, '(b s) h d -> b s (h d)', b=batch_size, s=seq_len)

    def _sdpa_attention(self, qkv, attn_mask, batch_size, seq_len, dim):
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_bias = None
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask != 0
            if attn_mask.dim() == 2:
                attn_mask = (~attn_mask)[:, None, None, :]
            else:
                attn_mask = ~attn_mask
            attn_bias = attn_mask.to(dtype=q.dtype) * torch.finfo(q.dtype).min

        joint_embeds = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
        return joint_embeds.transpose(1, 2).reshape(batch_size, seq_len, dim)


#################################################################################
#                                 LatentDiscreteDiTModel                        #
#################################################################################


class JointDiscreteDiTModel(DiTPretrainedModel):
    """Multi-modal Joint discrete DiT model."""

    def __init__(self, config: LatentDiTConfig) -> None:
        """Initialize the JointDiscreteDiTModel.

        :param config: The configuration object.
        """
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.cond_size = config.cond_hidden_size
        self.in_channels = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.y_latent_dim = config.y_latent_dim
        self.y_num_patches = config.y_num_patches
        self.normalize_embeddings = config.normalize_embeddings
        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches
        self.y_output_dim = (
            config.y_output_dim if hasattr(config, 'y_output_dim') and (config.y_output_dim is not None) else self.patch_dim
        )
        self.head_hidden_size = (
            config.head_hidden_size
            if hasattr(config, 'head_hidden_size') and (config.head_hidden_size is not None)
            else None
        )
        self.head_num_layers = getattr(config, 'head_num_layers', 0)
        self.head_num_heads = getattr(config, 'head_num_attention_heads', None) or self.num_heads
        self.use_wide_head = self.head_hidden_size is not None and self.head_num_layers > 0
        if self.head_hidden_size is None:
            self.head_hidden_size = self.in_channels
        assert self.head_hidden_size % self.head_num_heads == 0, 'head_hidden_size must be divisible by head_num_heads'


        self.t_embedder = TimestepEmbedder(self.cond_size)
        self.t_y_embedder = TimestepEmbedder(self.cond_size)
        self.y_cond_embedder = nn.Sequential(
            nn.Linear(self.patch_dim * self.y_num_patches, self.cond_size),
            nn.SiLU(),
            nn.Linear(self.cond_size, self.cond_size),
        )

        if self.config.soft_inputs:
            # Since the first transformer block applies LayerNorm, we prefer an MLP rather than a linear layer
            # otherwise the model is scale invariant with respect to the y input embeddings.
            self.y_patch_embedder = nn.Sequential(
                nn.Linear(self.patch_dim, self.in_channels * 4),
                nn.SiLU(),
                nn.Linear(self.in_channels * 4, self.in_channels),
            )
        else:
            self.y_patch_embedder = EmbeddingLayer(
                self.in_channels,
                self.patch_dim,
            )  # in this case, patch_dim is vocab size for y. Forward should output as many dims as patch_dim.

        self.w = EmbeddingLayer(self.in_channels, self.vocab_size)
        self.rotary_emb = Rotary(self.in_channels // self.num_heads)

        # Learned absolute slot embeddings for latent stream
        # These identify each latent as belonging to a specific semantic slot
        self.use_rope_for_latents = getattr(config, 'use_rope_for_latents', True)
        if not self.use_rope_for_latents:
            self.y_slot_embeddings = nn.Parameter(
                torch.randn(self.y_num_patches, self.in_channels) * 0.02,
            )
        else:
            self.y_slot_embeddings = None

        mm_dit_blocks = [
            MM_DDiTBlock(
                self.in_channels,
                self.num_heads,
                self.cond_size,
                config.mlp_ratio,
                attention_backend=config.attn_backend,
                use_rope_for_latents=self.use_rope_for_latents,
            )
            for _ in range(config.num_hidden_layers)
        ]
        # keep previous ordering (ema loading method stored params as lists)
        self.blocks = nn.ModuleList(mm_dit_blocks)
        if self.use_wide_head:
            self.y_head = WideTransformerHead(
                self.in_channels,
                self.head_hidden_size,
                self.head_num_layers,
                self.head_num_heads,
                self.cond_size,
                config.mlp_ratio,
                config.dropout,
                attention_backend=config.attn_backend,
            )

        self.final_layer_x = DDitFinalLayer(
            self.in_channels,
            self.vocab_size,
            self.cond_size,
        )

        final_layer_y_input_dim = self.head_hidden_size if self.use_wide_head else self.in_channels
        self.final_layer_y = DDitFinalLayer(
            final_layer_y_input_dim,
            self.y_output_dim,
            self.cond_size,
        )

        self._get_bias_dropout_scale()
        self.gradient_checkpointing = False

    def _get_bias_dropout_scale(self):
        self.bias_dropout_add_scale = get_bias_dropout_add_scale(self.training)

    def forward(
        self,
        x_input_ids: torch.Tensor,
        y_input_embeds: torch.Tensor | None,
        timesteps_x: torch.Tensor | None = None,
        timesteps_y: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        keep_y_attention: bool = True,
        **kwargs: dict[str, Any],
    ) -> Output:
        """Forward pass for the JointDiscreteDiTModel.

        :param x_input_ids: The input ids of the x tokens [B, S].
        :param y_input_embeds: The input embeddings of the y tokens [B, D=y_num_patches * patch_dim].
        :param timesteps_x: The timesteps of the x tokens [B].
        :param timesteps_y: The timesteps of the y tokens [B].
        """
        # 10% of the time, replace y_input_embeds with zeros
        # if self.training and torch.rand(1).item() < 0.1:
        #     y_input_embeds = torch.zeros_like(y_input_embeds)

        with torch.autocast(device_type=x_input_ids.device.type, enabled=False):
            self._get_bias_dropout_scale()
            # 1. Embed x
            x_embed = self.w(x_input_ids)
            x_seq_len = x_embed.shape[1]

            if self.config.soft_inputs:
                # 2. Patch and embed y
                # patch
                target_dtype = next(self.y_patch_embedder.parameters()).dtype
                y_input_embeds = y_input_embeds.to(target_dtype)
                y_input_embeds_patched = y_input_embeds.view(y_input_embeds.shape[0], self.y_num_patches, -1)
                y_embed = self.y_patch_embedder(y_input_embeds_patched)  # residual connection
            else:
                y_embed = self.y_patch_embedder(y_input_embeds)
            y_seq_len = y_embed.shape[1]

            # Add learned slot embeddings to latent stream (if not using RoPE for latents)
            if self.y_slot_embeddings is not None:
                y_embed = y_embed + self.y_slot_embeddings.unsqueeze(0)  # (B, y_num_patches, D)

            # Create conditioning vector c for y
            # y_cond = self.y_cond_embedder(y_input_embeds)
            # c += y_cond

            # 4. Create conditioning vector c from x and y timestep
            if timesteps_x is None:
                timesteps_x = torch.zeros(x_embed.shape[0], device=x_embed.device)
            c = self.t_embedder(timesteps_x)
            if timesteps_y is None:
                timesteps_y = torch.zeros(y_embed.shape[0], device=y_embed.device)
            c += self.t_y_embedder(timesteps_y)

            c = F.silu(c)

            # Create attention mask (mask pads in x, keep all y tokens)
            x_only_attn_mask = attention_mask

            B = x_embed.shape[0]
            joint_attn_mask = _build_joint_attention_mask(
                attention_mask=attention_mask,
                batch_size=B,
                x_seq_len=x_seq_len,
                y_seq_len=y_seq_len,
                device=x_embed.device,
                keep_y=keep_y_attention,
            )

            # 3. Create joint sequence for rotary embedding
            joint_embeds = torch.cat([x_embed, y_embed], dim=1)

            # 4. Create rotary embedding
            rope = self.rotary_emb(joint_embeds)

        # 5. Pass through DiT blocks
        mask_to_pass = joint_attn_mask if self.use_rope_for_latents else x_only_attn_mask
        if self.gradient_checkpointing and self.training:
            for i, block in enumerate(self.blocks):
                # checkpoint requires a tuple of args
                x_embed, y_embed = checkpoint(block, x_embed, y_embed, rope, c, mask_to_pass, use_reentrant=False)
        else:
            for i, block in enumerate(self.blocks):
                x_embed, y_embed = block(x_embed, y_embed, rope, c, attn_mask=mask_to_pass)

        if self.use_wide_head:
            # WideTransformerHead only processes Y, so pass Y-only mask (all ones since Y has no padding)
            y_embed = self.y_head(y_embed, c, attention_mask=None)

        logits = self.final_layer_x(x_embed, c)
        y_pred = self.final_layer_y(y_embed, c)
        y_std = torch.zeros_like(y_pred)

        if self.config.flatten_outputs:
            # glue y_pred patches together in order to get full y_pred
            y_pred = y_pred.view(y_pred.shape[0], self.y_latent_dim)
            y_std = y_std.view(y_std.shape[0], self.y_latent_dim)

        # Compare the embedding of the same token
        # tm = x_input_ids == 11
        # embeds_same_token = x_embed[tm]
        # print_rank_zero('Emebddings of token 11:', embeds_same_token[..., :10])

        return Output(logits=logits, y_pred=y_pred, y_std=y_std)


class EncoderDiTModel(DiTPretrainedModel):
    """Encoder DiT model."""

    def __init__(self, config: LatentDiTConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.in_channels = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.cond_size = config.cond_hidden_size
        self.y_latent_dim = config.y_latent_dim
        self.normalize_embeddings = config.normalize_embeddings
        self.y_num_patches = config.y_num_patches
        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches

        self.w = EmbeddingLayer(self.in_channels, self.vocab_size)
        self.time_embedder = TimestepEmbedder(self.cond_size)
        self.rotary_emb = Rotary(self.in_channels // self.num_heads)
        dit_blocks = [
            DDiTBlock(
                self.in_channels,
                self.num_heads,
                cond_dim=self.cond_size,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_backend=config.attn_backend,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.blocks = nn.ModuleList(dit_blocks)

        self.final_layer = DDitFinalLayer(
            self.in_channels,
            self.patch_dim,
            cond_dim=self.cond_size,
        )

        # Xavier uniform init for the last projection
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

        self.gradient_checkpointing = False

    def extract_encoding_from_output(self, output: Output) -> torch.Tensor:
        """Extract the encoding from the output of the EncoderDiTModel.

        :param output: The output of the EncoderDiTModel.
        :return: The encoding.
        """
        return output.y_pred

    def forward(
        self,
        x_input_ids: torch.Tensor,
        masked_tokens_pos: torch.Tensor | None = None,
        p_r: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> Output:
        """Forward pass for the EncoderDiTModel.

        :param x_input_ids: The input ids of the x tokens.
        :param masked_tokens_pos: Boolean mask indicating which tokens are masked.
        :param p_r: Probability for p_r masking.
        :param attention_mask: Attention mask for the input sequence.
        """
        with torch.autocast(device_type=x_input_ids.device.type, enabled=False):
            B = x_input_ids.size(0)

            # 1. Embed x
            x_embed = self.w(x_input_ids)

            # 2. p_r masking (zero embeddings at masked token positions) BEFORE the model
            if p_r is not None:
                # torch.rand_like over (B,) then compare
                apply_mask = (torch.rand_like(p_r) < p_r).to(torch.bool)  # (B,)
                # Expand to tokens and combine with masked positions
                token_zero_mask = masked_tokens_pos & apply_mask.unsqueeze(1)  # (B, L) bool
                # Zero-IN the embeddings at those token positions
                x_embed = x_embed.masked_fill(token_zero_mask.unsqueeze(-1), 0.0)

            c = self.time_embedder(torch.zeros(B, device=x_input_ids.device))
            c = F.silu(c)

            # 3. Pass through DiT blocks
            rope = self.rotary_emb(x_embed)



        if self.gradient_checkpointing and self.training:
            for block in self.blocks:
                x_embed = checkpoint(
                    block,
                    x_embed,
                    rope,
                    c,
                    attention_mask,
                    None,
                    use_reentrant=False,
                )
        else:
            for block in self.blocks:
                x_embed = block(x_embed, rope, c=c, attention_mask=attention_mask)

        # 4. Apply final layer and extract last y_num_patches tokens
        output = self.final_layer(x_embed, c)  # (B, seq_len, patch_dim)

        # Select the last y_num_patches tokens as the encoded representation
        y_pred = output[:, -self.y_num_patches :, :]  # (B, y_num_patches, patch_dim)

        # 5. Post-processing
        if self.normalize_embeddings:
            y_pred = F.normalize(y_pred, dim=-1, p=2)

        y_std = torch.zeros_like(y_pred)

        if self.config.flatten_outputs:
            y_pred = y_pred.reshape(B, self.y_latent_dim)
            y_std = y_std.reshape(B, self.y_latent_dim)

        # 6. p_r masking AFTER the model forward (zero continuous latents at masked positions)
        if p_r is not None and not self.config.flatten_outputs:
            # Expand token_zero_mask to match last y_num_patches positions
            # Take last y_num_patches positions from token_zero_mask
            y_token_mask = token_zero_mask[:, -self.y_num_patches :]
            y_pred = y_pred.masked_fill(y_token_mask.unsqueeze(-1), 0.0)
        elif p_r is not None and self.config.flatten_outputs:
            # If flattened, we need to expand the mask accordingly
            y_token_mask = token_zero_mask[:, -self.y_num_patches :].unsqueeze(-1).expand(-1, -1, self.patch_dim)
            y_token_mask = y_token_mask.reshape(B, self.y_latent_dim)
            y_pred = y_pred.masked_fill(y_token_mask, 0.0)

        return Output(logits=None, y_pred=y_pred, y_std=y_std)


class ContinuousDiTModel(DiTPretrainedModel):
    """Continuous DiT model, typically used to drive the latent diffusion process."""

    def __init__(self, config: LatentDiTConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.cond_size = config.cond_hidden_size
        self.in_channels = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.y_latent_dim = config.y_latent_dim
        self.y_num_patches = config.y_num_patches
        self.normalize_embeddings = config.normalize_embeddings

        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches
        self.head_hidden_size = (
            config.head_hidden_size
            if hasattr(config, 'head_hidden_size') and (config.head_hidden_size is not None)
            else None
        )
        self.head_num_layers = getattr(config, 'head_num_layers', 0)
        self.head_num_heads = getattr(config, 'head_num_attention_heads', None) or self.num_heads
        self.use_wide_head = self.head_hidden_size is not None and self.head_num_layers > 0
        if self.head_hidden_size is None:
            self.head_hidden_size = self.in_channels
        assert self.head_hidden_size % self.head_num_heads == 0, 'head_hidden_size must be divisible by head_num_heads'
        self.y_patch_embedder = nn.Linear(self.patch_dim, self.in_channels)

        # torch.nn.init.xavier_uniform_(self.y_patch_embedder[0].weight)
        # torch.nn.init.constant_(self.y_patch_embedder[0].bias, 0.0)
        # torch.nn.init.xavier_uniform_(self.y_patch_embedder[2].weight)
        # torch.nn.init.constant_(self.y_patch_embedder[2].bias, 0.0)

        self.t_y_embedder = TimestepEmbedder(self.cond_size)

        self.rotary_emb = Rotary(self.in_channels // self.num_heads)
        dit_blocks = [
            DDiTBlock(
                self.in_channels,
                self.num_heads,
                self.cond_size,
                config.mlp_ratio,
                attention_backend=config.attn_backend,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.blocks = nn.ModuleList(dit_blocks)

        if self.use_wide_head:
            self.y_head = WideTransformerHead(
                self.in_channels,
                self.head_hidden_size,
                self.head_num_layers,
                self.head_num_heads,
                self.cond_size,
                config.mlp_ratio,
                config.dropout,
                attention_backend=config.attn_backend,
            )

        self.final_layer_y = DDitFinalLayer(
            self.head_hidden_size if self.use_wide_head else self.in_channels,
            self.patch_dim,
            self.cond_size,
        )

        self._get_bias_dropout_scale()
        self.gradient_checkpointing = False

    def _get_bias_dropout_scale(self):
        self.bias_dropout_add_scale = get_bias_dropout_add_scale(self.training)

    def forward(
        self,
        y_input_embeds: torch.Tensor,
        timesteps_y: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> Output:
        """Forward pass for the ContinuousDiTModel.

        :param y_input_embeds: The input embeddings of the y tokens.
        :param timesteps_y: The timesteps of the y tokens.
        """
        with torch.autocast(device_type=y_input_embeds.device.type, enabled=False):
            self._get_bias_dropout_scale()

            # 1. Patch and embed y
            target_dtype = next(self.y_patch_embedder.parameters()).dtype
            y_input_embeds = y_input_embeds.to(target_dtype)
            y_input_embeds_patched = y_input_embeds.view(y_input_embeds.shape[0], self.y_num_patches, -1)
            y_embed = self.y_patch_embedder(y_input_embeds_patched)

            # 4. Create conditioning vector c from x and y timestep
            assert timesteps_y is not None, 'timesteps_y must be provided'
            c = self.t_y_embedder(timesteps_y)
            c = F.silu(c)

            # 6. Pass through DiT blocks
            rope = self.rotary_emb(y_embed)

        if self.gradient_checkpointing and self.training:
            for block in self.blocks:
                y_embed = checkpoint(block, y_embed, rope, c, use_reentrant=False)
        else:
            for block in self.blocks:
                y_embed = block(y_embed, rope, c)

        if self.use_wide_head:
            y_embed = self.y_head(y_embed, c)

        y_pred = self.final_layer_y(y_embed, c)
        y_pred = y_pred.view(y_pred.shape[0], self.y_latent_dim)
        y_std = torch.zeros_like(y_pred)

        return Output(logits=None, y_pred=y_pred, y_std=y_std)
