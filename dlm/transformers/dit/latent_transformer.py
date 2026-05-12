"""Seq2Seq-based Latent Discrete Diffusion Models."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import PreTrainedModel

from .configuration_latent_transformer import LatentTransformerConfig
from .dit import (
    DDiTBlock,
    DDitFinalLayer,
    EmbeddingLayer,
    Rotary,
    TimestepEmbedder,
)
from .dit import (
    LayerNorm as DiTLayerNorm,
)
from .output import Output


@torch.no_grad()
def _init_linear_zero_(layer: nn.Linear) -> nn.Linear:
    """Zero-out a Linear layer so it outputs exactly 0 at init."""
    layer.weight.data.zero_()
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


# ------------------------------------------------------------------------------------
# Core utils (RMSNorm, RoPE, attention helpers)
# ------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    # x: (bsz, seqlen, n_heads, head_dim)
    xr, xi = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
    # Reshape freqs for broadcasting
    freqs_cos = freqs_cos.view(1, freqs_cos.shape[0], 1, freqs_cos.shape[1])
    freqs_sin = freqs_sin.view(1, freqs_sin.shape[0], 1, freqs_sin.shape[1])
    xo_r = xr * freqs_cos - xi * freqs_sin
    xo_i = xr * freqs_sin + xi * freqs_cos
    x_out = torch.stack([xo_r, xo_i], dim=-1).flatten(3)
    return x_out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, s, h_kv, d = x.shape
    return x[:, :, :, None, :].expand(b, s, h_kv, n_rep, d).reshape(b, s, h_kv * n_rep, d)


# ------------------------------------------------------------------------------------
# Attention blocks
# ------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.1,
        is_causal: bool = False,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cos[:seqlen], freqs_sin[:seqlen])
        xk = apply_rotary_emb(xk, freqs_cos[:seqlen], freqs_sin[:seqlen])

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(out)
        return self.dropout(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, enc_dim: int, n_heads: int, n_kv_heads: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(enc_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(enc_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        freqs_cos_q: torch.Tensor,
        freqs_sin_q: torch.Tensor,
        freqs_cos_k: torch.Tensor,
        freqs_sin_k: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, tgt_len, _ = x_q.shape
        src_len = x_kv.shape[1]

        xq = self.wq(x_q).view(b, tgt_len, self.n_heads, self.head_dim)
        xk = self.wk(x_kv).view(b, src_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x_kv).view(b, src_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cos_q[:tgt_len], freqs_sin_q[:tgt_len])
        xk = apply_rotary_emb(xk, freqs_cos_k[:src_len], freqs_sin_k[:src_len])

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Create attention mask from key_padding_mask if provided
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (1.0 - key_padding_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9

        out = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(b, tgt_len, -1)
        out = self.wo(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, multiple_of: int = 256, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ------------------------------------------------------------------------------------
# Transformer blocks
# ------------------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, n_kv_heads, dropout, is_causal=False)
        self.feed_forward = FeedForward(dim, hidden_dim, multiple_of, dropout)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads, n_kv_heads, dropout, is_causal=False)
        self.cross_attn = CrossAttention(dim, dim, n_heads, n_kv_heads, dropout)
        self.feed_forward = FeedForward(dim, hidden_dim, multiple_of, dropout)

        self.attn_norm = RMSNorm(dim, norm_eps)
        self.cross_attn_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        freqs_cos_dec: torch.Tensor,
        freqs_sin_dec: torch.Tensor,
        freqs_cos_enc: torch.Tensor,
        freqs_sin_enc: torch.Tensor,
        enc_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.self_attn(self.attn_norm(x), freqs_cos_dec, freqs_sin_dec)
        h = h + self.cross_attn(
            self.cross_attn_norm(h),
            enc_out,
            freqs_cos_dec,
            freqs_sin_dec,
            freqs_cos_enc,
            freqs_sin_enc,
            enc_key_padding_mask,
        )
        h = h + self.feed_forward(self.ffn_norm(h))
        return h


# ------------------------------------------------------------------------------------
# Joint decoder block built on top of the original DiT architecture
# ------------------------------------------------------------------------------------
class JointDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        n_kv_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dit_block = DDiTBlock(dim, n_heads, cond_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.cross_attn = CrossAttention(dim, dim, n_heads, n_kv_heads, dropout)
        self.cross_attn_norm = DiTLayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
        cond: torch.Tensor,
        enc_out: torch.Tensor | None,
        freqs_cos_dec: torch.Tensor,
        freqs_sin_dec: torch.Tensor,
        freqs_cos_enc: torch.Tensor | None,
        freqs_sin_enc: torch.Tensor | None,
        enc_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.dit_block(x, rotary_cos_sin, cond, seqlens=None)

        if enc_out is not None and freqs_cos_enc is not None and freqs_sin_enc is not None:
            cross = self.cross_attn(
                self.cross_attn_norm(x),
                enc_out,
                freqs_cos_dec,
                freqs_sin_dec,
                freqs_cos_enc,
                freqs_sin_enc,
                enc_key_padding_mask,
            )
            x = x + cross

        return x


# ------------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------------
def _zero_decoder_cross_attention(blocks: Iterable[nn.Module]) -> None:
    """Set all cross-attention projection weights to zero."""

    with torch.no_grad():
        for block in blocks:
            cross_attn = getattr(block, 'cross_attn', None)
            if cross_attn is None:
                continue
            cross_attn.wq.weight.zero_()
            cross_attn.wk.weight.zero_()
            cross_attn.wv.weight.zero_()
            cross_attn.wo.weight.zero_()


# ------------------------------------------------------------------------------------
# Model base class
# ------------------------------------------------------------------------------------
class LatentPretrainedModel(PreTrainedModel):
    """Base class for Transformer models."""

    config_class = LatentTransformerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['EncoderBlock', 'DecoderBlock']
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)


# ------------------------------------------------------------------------------------
# Main Models
# ------------------------------------------------------------------------------------
class JointDiscreteDiTModelV2(LatentPretrainedModel):
    """Joint Seq2Seq model for discrete diffusion with latent conditioning."""

    def __init__(self, config: LatentTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.zero_init_cross_attention = getattr(config, 'zero_init_cross_attention', False)

        # Model dimensions
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.n_kv_heads
        self.vocab_size = config.vocab_size
        self.y_latent_dim = config.y_latent_dim
        self.y_num_patches = config.y_num_patches
        self.mlp_intermediate_size = int(config.hidden_size * config.mlp_ratio)

        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Y latent projection
        self.y_patch_embedder = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

        # Timestep embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Encoder layers
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    self.dim,
                    self.n_heads,
                    self.n_kv_heads,
                    hidden_dim=self.mlp_intermediate_size,
                    multiple_of=256,
                    norm_eps=1e-5,
                    dropout=config.dropout,
                )
                for _ in range(config.num_hidden_layers // 2)
            ],
        )

        # Decoder layers
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.dim,
                    self.n_heads,
                    self.n_kv_heads,
                    hidden_dim=self.mlp_intermediate_size,
                    multiple_of=256,
                    norm_eps=1e-5,
                    dropout=config.dropout,
                )
                for _ in range(config.num_hidden_layers // 2)
            ],
        )

        self.norm = RMSNorm(self.dim, eps=1e-5)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        # Final layers for y patches (mean and std) - project back to patch_dim
        self.final_layer_y = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.SiLU(),
            nn.Linear(self.dim * 2, self.patch_dim),
        )

        self.final_layer_y_std = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.SiLU(),
            nn.Linear(self.dim * 2, self.patch_dim),
        )

        # RoPE
        max_seq_len = config.max_sequence_length
        rope_len = max_seq_len + self.y_num_patches
        freqs_cos, freqs_sin = precompute_freqs_cis(self.dim // self.n_heads, rope_len)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        self.gradient_checkpointing = False
        self._init_weights()
        if self.zero_init_cross_attention:
            _zero_decoder_cross_attention(self.decoder_blocks)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_input_ids: torch.Tensor,
        y_input_embeds: torch.Tensor,
        timesteps_x: torch.Tensor | None = None,
        timesteps_y: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> Output:
        bsz, seq_len = x_input_ids.shape

        # Embed tokens
        x_embed = self.token_embedding(x_input_ids)

        # Process y latents as patches
        y_patches = y_input_embeds.view(bsz, self.y_num_patches, -1)
        y_embed = self.y_patch_embedder(y_patches)

        # Concatenate: [y_patches, x_tokens]
        enc_input = torch.cat([y_embed, x_embed], dim=1)
        enc_input = self.dropout(enc_input)

        # Encoder forward pass
        total_enc_len = enc_input.size(1)
        for block in self.encoder_blocks:
            if self.gradient_checkpointing and self.training:
                enc_input = checkpoint(
                    block,
                    enc_input,
                    self.freqs_cos[:total_enc_len],
                    self.freqs_sin[:total_enc_len],
                    use_reentrant=False,
                )
            else:
                enc_input = block(enc_input, self.freqs_cos[:total_enc_len], self.freqs_sin[:total_enc_len])

        # Split encoder output
        y_encoded = enc_input[:, : self.y_num_patches]  # Keep y patches for final prediction
        enc_memory = enc_input  # Use all encoder output as memory for decoder

        # Decoder: process x tokens with cross-attention to encoder memory
        dec_input = x_embed

        # Add timestep conditioning
        if timesteps_x is not None:
            t_norm = timesteps_x.float().view(bsz, 1, 1) / 1000.0
            t_bias = self.time_mlp(t_norm)
            dec_input = dec_input + t_bias

        dec_input = self.dropout(dec_input)

        # Decoder forward pass
        for block in self.decoder_blocks:
            if self.gradient_checkpointing and self.training:
                dec_input = checkpoint(
                    block,
                    dec_input,
                    enc_memory,
                    self.freqs_cos[:seq_len],
                    self.freqs_sin[:seq_len],
                    self.freqs_cos[:total_enc_len],
                    self.freqs_sin[:total_enc_len],
                    None,
                    use_reentrant=False,
                )
            else:
                dec_input = block(
                    dec_input,
                    enc_memory,
                    self.freqs_cos[:seq_len],
                    self.freqs_sin[:seq_len],
                    self.freqs_cos[:total_enc_len],
                    self.freqs_sin[:total_enc_len],
                    None,
                )

        dec_input = self.norm(dec_input)

        # Output predictions for x
        logits = self.output(dec_input)

        # Predict y from encoded y patches - project back to patch_dim and reassemble
        y_pred_patches = self.final_layer_y(y_encoded)  # (B, y_num_patches, patch_dim)
        y_pred = y_pred_patches.view(bsz, self.y_latent_dim)

        y_std_patches = self.final_layer_y_std(y_encoded)  # (B, y_num_patches, patch_dim)
        y_std = F.softplus(y_std_patches.view(bsz, self.y_latent_dim))

        return Output(logits=logits, y_pred=y_pred, y_std=y_std)


class JointDiscreteDiTModel(LatentPretrainedModel):
    """
    MODIFIED: A Seq2Seq model for discrete diffusion that uses direct cross-attention
    for conditioning, rather than joint encoding.

    In this architecture, the model functions as a decoder that processes the noisy
    discrete tokens 'x' and cross-attends to the continuous latent conditioning 'y'
    at each layer. This aligns with standard Transformer encoder-decoder patterns.
    """

    def __init__(self, config: LatentTransformerConfig) -> None:
        super().__init__(config)
        self.config = config

        # Model dimensions
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.n_kv_heads
        self.vocab_size = config.vocab_size
        self.y_latent_dim = config.y_latent_dim
        self.y_num_patches = config.y_num_patches
        self.mlp_intermediate_size = int(config.hidden_size * config.mlp_ratio)

        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Y latent projection
        self.y_patch_embedder = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

        # Timestep embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        # --- MODIFICATION: Use a single stack of DecoderBlocks ---
        # The model is now a pure decoder stack, not an encoder-decoder.
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.dim,
                    self.n_heads,
                    self.n_kv_heads,
                    hidden_dim=self.mlp_intermediate_size,
                    multiple_of=256,
                    norm_eps=1e-5,
                    dropout=config.dropout,
                )
                for _ in range(config.num_hidden_layers)  # Use full number of layers
            ],
        )

        self.norm = RMSNorm(self.dim, eps=1e-5)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        # --- MODIFICATION: Separate RoPE for decoder (x) and memory (y) ---
        max_seq_len = config.max_sequence_length

        # RoPE for the decoder sequence 'x'
        freqs_cos_dec, freqs_sin_dec = precompute_freqs_cis(self.dim // self.n_heads, max_seq_len)
        self.register_buffer('freqs_cos_dec', freqs_cos_dec, persistent=False)
        self.register_buffer('freqs_sin_dec', freqs_sin_dec, persistent=False)

        # RoPE for the encoder memory 'y'
        freqs_cos_enc, freqs_sin_enc = precompute_freqs_cis(self.dim // self.n_heads, self.y_num_patches)
        self.register_buffer('freqs_cos_enc', freqs_cos_enc, persistent=False)
        self.register_buffer('freqs_sin_enc', freqs_sin_enc, persistent=False)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_input_ids: torch.Tensor,
        y_input_embeds: torch.Tensor,
        timesteps_x: torch.Tensor | None = None,
        timesteps_y: torch.Tensor | None = None,  # Unused, but kept for API consistency
        **kwargs: dict[str, Any],
    ) -> Output:
        bsz, seq_len = x_input_ids.shape

        with torch.autocast(device_type='cuda', enabled=False):
            # 1. Prepare input sequences
            # The main sequence to be processed/denoised
            dec_input = self.token_embedding(x_input_ids)

            # The conditioning sequence (memory) for cross-attention
            target_dtype = next(self.y_patch_embedder.parameters()).dtype
            y_input_embeds = y_input_embeds.to(target_dtype)
            y_patches = y_input_embeds.view(bsz, self.y_num_patches, -1)
            enc_memory = self.y_patch_embedder(y_patches)

            # 2. Add timestep conditioning to the main sequence
            if timesteps_x is not None:
                t_norm = timesteps_x.float().view(bsz, 1, 1) / 1000.0
                t_bias = self.time_mlp(t_norm)
                dec_input = dec_input + t_bias

            dec_input = self.dropout(dec_input)

        # 3. Forward through the decoder stack
        for block in self.decoder_blocks:
            if self.gradient_checkpointing and self.training:
                dec_input = checkpoint(
                    block,
                    dec_input,
                    enc_memory,
                    self.freqs_cos_dec[:seq_len],
                    self.freqs_sin_dec[:seq_len],
                    self.freqs_cos_enc,  # Use full freqs for y
                    self.freqs_sin_enc,  # Use full freqs for y
                    None,
                    use_reentrant=False,
                )
            else:
                dec_input = block(
                    x=dec_input,
                    enc_out=enc_memory,
                    freqs_cos_dec=self.freqs_cos_dec[:seq_len],
                    freqs_sin_dec=self.freqs_sin_dec[:seq_len],
                    freqs_cos_enc=self.freqs_cos_enc,
                    freqs_sin_enc=self.freqs_sin_enc,
                    enc_key_padding_mask=None,
                )

        dec_input = self.norm(dec_input)

        # 4. Output predictions for x
        logits = self.output(dec_input)

        # Note: y_pred and y_std are no longer predicted as y is only used for conditioning.
        return Output(logits=logits, y_pred=None, y_std=None)


class CompatibleJointDiscreteDiTModel(LatentPretrainedModel):
    """Joint decoder that extends the pretrained DiscreteDiT with optional cross-attention."""

    def __init__(self, config: LatentTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.zero_init_cross_attention = getattr(config, 'zero_init_cross_attention', False)

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.y_latent_dim = config.y_latent_dim
        self.y_num_patches = config.y_num_patches
        self.mlp_ratio = config.mlp_ratio
        self.cond_hidden_size = config.cond_hidden_size

        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches

        # Pretrained decoder components
        self.token_embedding = EmbeddingLayer(self.dim, self.vocab_size)
        self.sigma_map = TimestepEmbedder(self.cond_hidden_size)
        self.rotary_emb = Rotary(self.dim // self.n_heads)

        # Conditioning projection (kept for compatibility with latent cross-attention)
        self.y_patch_embedder = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

        self.decoder_blocks = nn.ModuleList(
            [
                JointDecoderBlock(
                    self.dim,
                    self.n_heads,
                    self.cond_hidden_size,
                    n_kv_heads=self.n_kv_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_hidden_layers)
            ],
        )

        self.output_layer = DDitFinalLayer(self.dim, self.vocab_size, self.cond_hidden_size)

        max_seq_len = config.max_sequence_length
        freqs_cos_dec, freqs_sin_dec = precompute_freqs_cis(self.dim // self.n_heads, max_seq_len)
        self.register_buffer('freqs_cos_dec', freqs_cos_dec, persistent=False)
        self.register_buffer('freqs_sin_dec', freqs_sin_dec, persistent=False)

        freqs_cos_enc, freqs_sin_enc = precompute_freqs_cis(self.dim // self.n_heads, self.y_num_patches)
        self.register_buffer('freqs_cos_enc', freqs_cos_enc, persistent=False)
        self.register_buffer('freqs_sin_enc', freqs_sin_enc, persistent=False)

        self.gradient_checkpointing = False
        self._init_weights()
        if self.zero_init_cross_attention:
            _zero_decoder_cross_attention(self.decoder_blocks)

    def _init_weights(self) -> None:
        for module in self.y_patch_embedder:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        for block in self.decoder_blocks:
            torch.nn.init.normal_(block.cross_attn.wq.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(block.cross_attn.wk.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(block.cross_attn.wv.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(block.cross_attn.wo.weight, mean=0.0, std=0.02)
            block.cross_attn_norm.weight.data.fill_(1.0)

    def forward(
        self,
        x_input_ids: torch.Tensor,
        y_input_embeds: torch.Tensor,
        timesteps_x: torch.Tensor | None = None,
        timesteps_y: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> Output:
        # del timesteps_y, kwargs

        bsz, seq_len = x_input_ids.shape

        with torch.autocast(device_type='cuda', enabled=False):
            sigma = timesteps_x
            if sigma is None:
                sigma = torch.zeros(bsz, device=x_input_ids.device)
            sigma = sigma.view(-1)

            cond = F.silu(self.sigma_map(sigma))

            x_embed = self.token_embedding(x_input_ids)
            rotary_cos_sin = self.rotary_emb(x_embed)

            enc_memory = None
            if y_input_embeds is not None:
                target_dtype = next(self.y_patch_embedder.parameters()).dtype
                y_input_embeds = y_input_embeds.to(target_dtype)
                y_patches = y_input_embeds.view(bsz, self.y_num_patches, -1)
                enc_memory = self.y_patch_embedder(y_patches)

        for block in self.decoder_blocks:
            if self.gradient_checkpointing and self.training:
                x_embed = checkpoint(
                    block,
                    x_embed,
                    rotary_cos_sin,
                    cond,
                    enc_memory,
                    self.freqs_cos_dec[:seq_len],
                    self.freqs_sin_dec[:seq_len],
                    self.freqs_cos_enc if enc_memory is not None else None,
                    self.freqs_sin_enc if enc_memory is not None else None,
                    None,
                    use_reentrant=False,
                )
            else:
                x_embed = block(
                    x_embed,
                    rotary_cos_sin,
                    cond,
                    enc_memory,
                    self.freqs_cos_dec[:seq_len],
                    self.freqs_sin_dec[:seq_len],
                    self.freqs_cos_enc if enc_memory is not None else None,
                    self.freqs_sin_enc if enc_memory is not None else None,
                    None,
                )

        logits = self.output_layer(x_embed, cond)

        return Output(logits=logits, y_pred=None, y_std=None)


class EncoderDiTModel(LatentPretrainedModel):
    """Encoder model that produces latent representations from discrete tokens."""

    def __init__(self, config: LatentTransformerConfig):
        super().__init__(config)
        self.config = config

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.n_kv_heads
        self.vocab_size = config.vocab_size
        self.y_latent_dim = config.y_latent_dim
        self.y_num_patches = config.y_num_patches
        self.mlp_intermediate_size = int(config.hidden_size * config.mlp_ratio)
        self.tanh_out = config.tanh_out

        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Learnable y tokens (similar to original code)
        self.y_tokens = nn.Parameter(torch.zeros(self.y_num_patches, self.dim))
        nn.init.normal_(self.y_tokens, std=0.02)

        # Encoder blocks
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    self.dim,
                    self.n_heads,
                    self.n_kv_heads,
                    hidden_dim=self.mlp_intermediate_size,
                    multiple_of=256,
                    norm_eps=1e-5,
                    dropout=config.dropout,
                )
                for _ in range(config.num_hidden_layers)
            ],
        )

        self.norm = RMSNorm(self.dim, eps=1e-5)

        # Output heads that project to patch_dim
        self.final_layer_y = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            _init_linear_zero_(nn.Linear(self.dim * 4, self.patch_dim)),
        )

        self.final_layer_y_std = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            _init_linear_zero_(nn.Linear(self.dim * 4, self.patch_dim)),
        )
        # Initialize std bias for reasonable initial values
        self.final_layer_y_std[-1].bias.data.fill_(0.541)  # softplus^-1(1.0) ≈ 0.541

        # RoPE
        max_seq_len = config.max_sequence_length
        rope_len = max_seq_len + self.y_num_patches
        freqs_cos, freqs_sin = precompute_freqs_cis(self.dim // self.n_heads, rope_len)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_input_ids: torch.Tensor,
        masked_tokens_pos: torch.Tensor | None = None,
        p_r: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> Output:
        bsz, seq_len = x_input_ids.shape

        # Embed tokens
        x_embed = self.token_embedding(x_input_ids)
        if p_r is not None:
            # torch.rand_like over (B,) then compare
            apply_mask = (torch.rand_like(p_r) < p_r).to(torch.bool)  # (B,)
            # Expand to tokens and combine with masked positions
            token_zero_mask = masked_tokens_pos & apply_mask.unsqueeze(1)  # (B, L) bool
            # Zero-IN the embeddings at those token positions
            x_embed = x_embed.masked_fill(token_zero_mask.unsqueeze(-1), 0.0)

        # Add learnable y tokens
        y_embed = self.y_tokens.unsqueeze(0).expand(bsz, -1, -1)

        # Concatenate: [x_tokens, y_tokens]
        h = torch.cat([x_embed, y_embed], dim=1)
        h = self.dropout(h)

        # Forward through blocks
        total_len = h.size(1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, self.freqs_cos[:total_len], self.freqs_sin[:total_len], use_reentrant=False)
            else:
                h = block(h, self.freqs_cos[:total_len], self.freqs_sin[:total_len])

        h = self.norm(h)

        # Extract y tokens
        y_output = h[:, seq_len:]  # (B, y_num_patches, dim)

        # Project to patch dimension and reassemble
        y_pred_patches = self.final_layer_y(y_output)  # (B, y_num_patches, patch_dim)
        y_pred = y_pred_patches.view(bsz, self.y_latent_dim)
        # apply tanh to bound the outputs
        if self.tanh_out:
            y_pred = torch.tanh(y_pred)

        y_std_patches = self.final_layer_y_std(y_output)  # (B, y_num_patches, patch_dim)
        y_std = F.softplus(y_std_patches.view(bsz, self.y_latent_dim))

        # 3) pr masking (zero continuous latents at masked token positions) AFTER the model
        if p_r is not None:
            # assumes that y_pred and x_0 have the same shape (B, S, D') and (B, S, D).
            y_pred = y_pred.masked_fill(token_zero_mask.unsqueeze(-1), 0.0)

        return Output(logits=None, y_pred=y_pred, y_std=y_std)


class ContinuousDiTModel(LatentPretrainedModel):
    """Continuous DiT model for latent diffusion."""

    def __init__(self, config: LatentTransformerConfig):
        super().__init__(config)
        self.config = config

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.n_kv_heads
        self.y_latent_dim = config.y_latent_dim
        self.y_num_patches = config.y_num_patches
        self.mlp_intermediate_size = int(config.hidden_size * config.mlp_ratio)

        assert self.y_latent_dim % self.y_num_patches == 0, 'y_latent_dim must be divisible by y_num_patches'
        self.patch_dim = self.y_latent_dim // self.y_num_patches

        # Y patch embedder
        self.y_patch_embedder = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    self.dim,
                    self.n_heads,
                    self.n_kv_heads,
                    hidden_dim=self.mlp_intermediate_size,
                    multiple_of=256,
                    norm_eps=1e-5,
                    dropout=config.dropout,
                )
                for _ in range(config.num_hidden_layers)
            ],
        )

        self.norm = RMSNorm(self.dim, eps=1e-5)

        # Output heads that project back to patch_dim
        self.final_layer_y = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.SiLU(),
            _init_linear_zero_(nn.Linear(self.dim * 2, self.patch_dim)),
        )

        self.final_layer_y_std = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.SiLU(),
            _init_linear_zero_(nn.Linear(self.dim * 2, self.patch_dim)),
        )

        # Initialize std bias for reasonable initial values
        self.final_layer_y_std[-1].bias.data.fill_(0.541)  # softplus^-1(1.0) ≈ 0.541

        # RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(self.dim // self.n_heads, self.y_num_patches)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        y_input_embeds: torch.Tensor,
        timesteps_y: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> Output:
        bsz = y_input_embeds.shape[0]

        # Process y as patches
        y_patches = y_input_embeds.view(bsz, self.y_num_patches, -1)
        h = self.y_patch_embedder(y_patches)

        # Add timestep conditioning
        if timesteps_y is not None:
            t_norm = timesteps_y.float().view(bsz, 1, 1)
            t_bias = self.time_mlp(t_norm)
            h = h + t_bias

        # Forward through blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, self.freqs_cos, self.freqs_sin, use_reentrant=False)
            else:
                h = block(h, self.freqs_cos, self.freqs_sin)

        h = self.norm(h)

        # Predict mean and std - project back to patch_dim and reassemble
        y_pred_patches = self.final_layer_y(h)  # (B, y_num_patches, patch_dim)
        y_pred = y_pred_patches.view(bsz, self.y_latent_dim)

        y_std_patches = self.final_layer_y_std(h)  # (B, y_num_patches, patch_dim)
        y_std = F.softplus(y_std_patches.view(bsz, self.y_latent_dim))

        return Output(logits=None, y_pred=y_pred, y_std=y_std)
