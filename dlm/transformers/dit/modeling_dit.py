import math

import torch
from torch import nn
from torch.nn import functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.utils.import_utils import is_flash_attn_2_available

from einops import rearrange
from .rotary import Rotary, apply_rotary_pos_emb
from .fused_add_dropout_scale import (
    modulate_fused,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
)
from .configuration_dit import DiTConfig

if is_flash_attn_2_available():
    import flash_attn


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


# copied from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/model/transformer.py#L56
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # Consider matching the original DiT 0.02 std initialization here.

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class SelfAttention(nn.Module, ModuleUtilsMixin):
    # Using SDPA attention
    def __init__(self, config: DiTConfig, layer_idx: int | None = None):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.dropout
        self.config = config
        self.scaling = 1 / math.sqrt(self.head_dim)
        self.is_causal = config.is_causal
        if self.is_causal:
            raise NotImplementedError("Causal attention is not implemented")
        self.layer_idx = layer_idx
        self.attn_qkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias
        )
        self.proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        x,
        attention_mask,
        position_embeddings,
    ):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b t (three h d) -> b t three h d", three=3, h=self.num_heads
        )
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            cos, sin = position_embeddings
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        if self.config._attn_implementation == "sdpa":
            q, k, v = (
                qkv[:, :, 0].transpose(1, 2),
                qkv[:, :, 1].transpose(1, 2),
                qkv[:, :, 2].transpose(1, 2),
            )
            attn_output = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                # attn_mask=attention_mask,
                dropout_p=0.0,
                # scale=self.scaling,
                is_causal=False,
            )
            attn_output = rearrange(attn_output, "(b t) h d -> b t (h d)", b=B)

        elif self.config._attn_implementation == "flash_attention_2":
            qkv = rearrange(qkv, "b t ... -> (b t) ...")
            cu_seqlens = torch.arange(
                0,
                (B + 1) * T,
                step=T,
                dtype=torch.int32,
                device=qkv.device,
            )
            attn_output = (
                flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, T, 0.0, causal=False
                )
            )
        else:
            raise NotImplementedError(
                f"Attention implementation {self.config._attn_implementation} not implemented"
            )
        attn_output = rearrange(attn_output, "(b s) h d -> b s (h d)", b=B)
        attn_output = self.proj(attn_output)
        return attn_output


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        config: DiTConfig,
        enable_ada_norm: bool = True,
        layer_idx: int | None = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        cond_hidden_size = config.cond_hidden_size
        self.dropout = config.dropout
        mlp_ratio = config.mlp_ratio
        self.enable_ada_norm = enable_ada_norm
        self.norm1 = LayerNorm(hidden_size)
        self.attn = SelfAttention(
            config,
            layer_idx=layer_idx,
        )
        self.norm2 = LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, mlp_ratio)
        if enable_ada_norm:
            self.adaLN_modulation = nn.Linear(
                cond_hidden_size, 6 * hidden_size, bias=True
            )
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c, attention_mask, position_embeddings):
        bias_dropout_scale_fn = (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )
        # attention operation
        if self.enable_ada_norm:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            )  # (bsz, hidden_size)
            residual = self.attn(
                x=modulate_fused(self.norm1(x), shift_msa, scale_msa),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
        else:
            gate_msa = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            residual = self.attn(
                x=self.norm1(x),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
        x = bias_dropout_scale_fn(
            residual,
            bias=None,
            scale=gate_msa,
            residual=residual,
            prob=self.dropout,
        )
        # mlp operation
        if self.enable_ada_norm:
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                bias=None,
                scale=gate_mlp,
                residual=x,
                prob=self.dropout,
            )
        else:
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)),
                bias=None,
                scale=gate_msa,
                residual=x,
                prob=self.dropout,
            )
        return x


class DiTLMHead(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, vocab_size, enable_ada_norm, cond_hidden_size):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.enable_ada_norm = enable_ada_norm
        if enable_ada_norm:
            self.adaLN_modulation = nn.Linear(
                cond_hidden_size,
                2 * hidden_size,
                bias=True,
            )
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm(x)
        if self.enable_ada_norm:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


class DiTPretrainedModel(PreTrainedModel):
    config_class = DiTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DiTBlock"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class DiTModel(DiTPretrainedModel):
    """
    Time-conditioned transformer w/ self-attn for discrete diffusion models
    """

    def __init__(self, config: DiTConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        torch.nn.init.kaiming_normal_(self.embed_tokens.weight, a=math.sqrt(5))
        enable_ada_norm = True
        self.time_embedder = TimestepEmbedder(config.cond_hidden_size)
        self.layers = nn.ModuleList(
            [
                DiTBlock(config, enable_ada_norm, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = Rotary(config.hidden_size // config.num_attention_heads)
        self.lm_head = DiTLMHead(
            config.hidden_size,
            config.vocab_size,
            enable_ada_norm,
            config.cond_hidden_size,
        )

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (bsz, seq_len, hidden_size)
        input_shape = inputs_embeds.shape[:-1]
        _, seq_length = input_shape
        if timesteps is None:
            timesteps = torch.zeros(
                input_shape[0], device=input_ids.device, dtype=torch.long
            )
        c = F.silu(self.time_embedder(timesteps))  # (bsz, hidden_size_ada_norm)
        # create position embeddings to be shared across the decoder layers
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.size(1), device=inputs_embeds.device
            ).expand(inputs_embeds.size(0), -1)
        position_embeddings = self.rotary_emb(
            inputs_embeds
        )  # (bsz, seq_len, hidden_size)
        # prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=inputs_embeds.device)
        if self.config._attn_implementation == "sdpa" and attention_mask.dim() == 2:
            attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask,
                inputs_embeds.dtype,
                tgt_len=seq_length,
            )
        else:
            attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape=input_shape
            )

        x = inputs_embeds
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x,
                    c,
                    attention_mask,
                    position_embeddings,
                )
            else:
                x = layer(
                    x=x,
                    c=c,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                )

        # # scale by sigma
        # esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
        # x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0

        # x = torch.scatter(x, -1, input_ids[..., None], torch.zeros_like(x[..., :1]))
        logits = self.lm_head(x, c)
        return CausalLMOutputWithPast(logits=logits)
