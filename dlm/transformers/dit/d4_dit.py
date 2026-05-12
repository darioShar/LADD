"""Copied from https://github.com/kuleshov-group/mdlm/blob/master/models/dit.py"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.modeling_utils import PreTrainedModel

from .configuration_d4_dit import D4DiTConfig
from .output import Output

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)



def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    # The unsqueeze is needed for broadcasting with shape (B, T, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Rotary(torch.nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        """Initialize the Rotary.

        :param dim: The dimension of the embedding.
        :param base: The base of the exponential.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Rotary.

        :param x: The input tensor.
        :param seq_dim: The dimension of the sequence.
        :return: The output tensor.
        """
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the tensor.

    :param x: The input tensor.
    :return: The output tensor.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding to the query, key, and value tensors.

    :param qkv: The query, key, and value tensors.
    :param cos: The cosine tensor.
    :param sin: The sine tensor.
    :return: The output tensor.
    """
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_qkv_

        cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        return apply_rotary_emb_qkv_(qkv, cos, sin)  # type: ignore[attr-defined]
    except ImportError:
        # Fallback for non-flash-attn
        q, k, v = qkv.unbind(2)
        q = (q * cos.to(q.dtype)) + (rotate_half(q) * sin.to(q.dtype))
        k = (k * cos.to(k.dtype)) + (rotate_half(k) * sin.to(k.dtype))
        return torch.stack([q, k, v], dim=2)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    """LayerNorm module."""

    def __init__(self, dim: int):
        """Initialize the LayerNorm.

        :param dim: The dimension of the embedding.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the LayerNorm.

        :param x: The input tensor.
        :return: The output tensor.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(
    x: torch.Tensor,
    W: torch.Tensor,
    x_skip: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """Residual linear operation.

    :param x: The input tensor.
    :param W: The weight tensor.
    :param x_skip: The skip tensor.
    :param residual_scale: The residual scale.
    :return: The output tensor.
    """
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale,
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """Initialize the TimestepEmbedder.

        :param hidden_size: The dimension of the hidden size.
        :param frequency_embedding_size: The dimension of the frequency embedding.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half,
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes: int, cond_size: int):
        """Initialize the LabelEmbedder.

        :param num_classes: The number of classes.
        :param cond_size: The dimension of the conditioning vector.
        """
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass for the LabelEmbedder.

        :param labels: The labels.
        :return: The output tensor.
        """
        return self.embedding_table(labels)


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4,
        dropout: float = 0.1,
    ):
        """Initialize the DDiTBlock.

        :param dim: The dimension of the embedding.
        :param n_heads: The number of attention heads.
        :param cond_dim: The dimension of the conditioning vector.
        """
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim), bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(int(mlp_ratio * dim), dim, bias=True),
        )
        self.dropout = dropout

        self.adaLN_modulation = nn.Sequential(
            # nn.SiLU(),  # added SiLU activation
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        self.adaLN_modulation[-1].weight.data.zero_()
        self.adaLN_modulation[-1].bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the DDiTBlock.

        :param x: The input tensor.
        :param rotary_cos_sin: The rotary cosine and sine tensors.
        :param c: The conditioning vector.
        :param attn_mask: The attention mask.
        :param seqlens: The sequence lengths.
        :return: The output tensor.
        """
        batch_size, seq_len, dim = x.shape[0], x.shape[1], x.shape[2]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)

        # attention operation
        x_skip = x
        x_norm = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads,
        )
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(qkv, cos, sin)

        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        x = x.transpose(1, 2).reshape(batch_size, seq_len, dim)

        x = bias_dropout_scale_fn(
            self.attn_out(x),
            None,
            gate_msa.unsqueeze(1),
            x_skip,
            self.dropout,
        )

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp.unsqueeze(1),
            x,  # or x_skip ??
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):
    """The embedding layer of DiT."""

    def __init__(self, dim: int, vocab_dim: int):
        """Initialize the EmbeddingLayer.

        :param dim: The dimension of the embedding.
        :param vocab_dim: The dimension of the vocabulary.
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the EmbeddingLayer.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        """Initialize the DDitFinalLayer.

        :param hidden_size: The dimension of the hidden size.
        :param out_channels: The dimension of the output channels.
        :param cond_dim: The dimension of the conditioning vector.
        """
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            # nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass for the DDitFinalLayer.

        :param x: The input tensor.
        :param c: The conditioning vector.
        :return: The output tensor.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTPretrainedModel(PreTrainedModel):
    """Base class for DiT models."""

    config_class = D4DiTConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['DDiTBlock']
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def __init__(self, *inputs, **kwargs) -> None:
        """Initialize the DiTPretrainedModel."""
        super().__init__(*inputs, **kwargs)


#################################################################################
#                                 D4DiTModel                        #
#################################################################################


class D4DiTModel(DiTPretrainedModel):
    """D4 DiT model."""

    def __init__(self, config: D4DiTConfig) -> None:
        """Initialize the D4DiTModel.

        :param config: The configuration object.
        """
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.cond_size = config.cond_hidden_size
        self.in_channels = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.proba_input = config.proba_input  # will expect a proba input rather than tokens

        if self.proba_input:
            self.w: nn.Module = nn.Sequential(
                nn.Linear(self.vocab_size, self.in_channels * 4),
                nn.SiLU(),
                nn.Linear(self.in_channels * 4, self.in_channels),
            )
        else:
            self.w: nn.Module = EmbeddingLayer(self.in_channels, self.vocab_size)

        self.t_embedder = TimestepEmbedder(self.cond_size)
        self.rotary_emb = Rotary(self.in_channels // self.num_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    self.in_channels,
                    self.num_heads,
                    self.cond_size,
                    config.mlp_ratio,
                )
                for _ in range(config.num_hidden_layers)
            ],
        )
        self.final_layer_x = DDitFinalLayer(
            self.in_channels,
            self.vocab_size,
            self.cond_size,
        )
        self._get_bias_dropout_scale()
        self.gradient_checkpointing = False

    def _get_bias_dropout_scale(self):
        self.bias_dropout_add_scale = get_bias_dropout_add_scale(self.training)

    def forward(
        self,
        x_input_ids: torch.Tensor,
        timesteps_x: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> Output:
        """Forward pass for the D4DiTModel.

        :param x_input_ids: The input ids of the x tokens.
        :param timesteps_x: The timesteps of the x tokens.
        """
        self._get_bias_dropout_scale()

        # 1. Embed x
        x_embed = self.w(x_input_ids)

        # Create conditioning vector c for y

        # Create conditioning vector c from x timestep
        batch_size = x_embed.shape[0]
        timesteps_x = torch.zeros(batch_size, device=x_embed.device) if timesteps_x is None else timesteps_x
        c_x = self.t_embedder(timesteps_x)
        # Create final conditioning vector c
        c = F.silu(c_x)

        # Create attention mask
        attn_mask = None

        # 3. Create rotary embedding
        rope = self.rotary_emb(x_embed)

        # 5. Pass through DiT blocks
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                # checkpoint requires a tuple of args
                x_embed = checkpoint(
                    block,
                    x_embed,
                    rope,
                    c,
                    attn_mask,
                    use_reentrant=False,
                )
            else:
                x_embed = block(x_embed, rope, c, attn_mask=attn_mask)

        # x_embed = joint_embeds[:, :x_seq_len, :]
        # y_embed = joint_embeds[:, x_seq_len:, :]

        logits = self.final_layer_x(x_embed, c_x)
        return Output(logits=logits)
