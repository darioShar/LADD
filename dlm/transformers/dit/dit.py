"""
Copied from https://github.com/kuleshov-group/mdlm/blob/master/models/dit.py
"""

import math

try:
    import flash_attn
    from flash_attn.layers import rotary as flash_rotary

    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn = None
    flash_rotary = None
    _HAS_FLASH_ATTN = False
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from .configuration_dit import DiTConfig
from .output import Output
from ...utils import print_rank_zero





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


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def bias_dropout_add_scale_train(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


def bias_dropout_add_scale_inference(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


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
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
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


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin, use_flash_attn: bool = True):
    # qkv: (batch, seq_len, 3, n_heads, head_dim)
    # cos, sin: (1, seq_len, 1, 1, head_dim)
    if _HAS_FLASH_ATTN and use_flash_attn:
        cos: torch.Tensor = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin: torch.Tensor = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        return flash_rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
    cos: torch.Tensor = cos[:, :, 0, :, :]
    sin: torch.Tensor = sin[:, :, 0, :, :]
    q, k, v = qkv.unbind(2)
    q = (q * cos.to(q.dtype)) + (rotate_half(q) * sin.to(q.dtype))
    k = (k * cos.to(k.dtype)) + (rotate_half(k) * sin.to(k.dtype))
    return torch.stack([q, k, v], dim=2)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
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

    def forward(self, t):
        if t.dim() == 0:
            t = t[None]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(dtype=self.mlp[0].weight.dtype))
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # Consider matching the original DiT 0.02 std initialization here.

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1, attention_backend: str = 'auto'):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.attention_backend = attention_backend

        
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
        self.gradient_checkpointing = False
        self._attn_fn = self._resolve_attention_backend()

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

    def forward(self, x, rotary_cos_sin, c, attention_mask=None, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        modulate_fn = self._get_modulate()
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(
            6,
            dim=2,
        )

        # attention operation
        x_skip = x
        x = modulate_fn(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads,
        )
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(
            qkv,
            cos.to(qkv.dtype),
            sin.to(qkv.dtype),
            use_flash_attn=self.attention_backend != 'sdpa',
        )

        x = self._attn_fn(qkv, attention_mask, batch_size, seq_len)

        # qkv = qkv.transpose(1, 3)
        # x = F.scaled_dot_product_attention(
        #     query=qkv[:, :, 0],
        #     key=qkv[:, :, 1],
        #     value=qkv[:, :, 2],
        #     attn_mask=None,
        #     is_causal=False,
        #     scale=None,
        # )
        # x = x.transpose(1, 2)
        # # x = rearrange(x, "b s h d -> b s (h d)")
        # x = x.reshape(batch_size, seq_len, -1)

        
        x = bias_dropout_scale_fn(
            self.attn_out(x),
            None,
            gate_msa,
            x_skip,
            self.dropout,
        )
        
        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fn(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x

    def _flash_attention(self, qkv, attention_mask, batch_size, seq_len):
        if attention_mask is None:
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
        if attention_mask.dim() != 2:
            raise ValueError('Flash attention only supports None or 2D padding masks')
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask != 0
        seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
        qkv_flat = rearrange(qkv, 'b s three h d -> (b s) three h d').contiguous()
        mask = attention_mask.reshape(-1)
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

    def _sdpa_attention(self, qkv, attention_mask, batch_size, seq_len):
        # FlashAttention note: for unmasked fixed-length inputs, swap this SDPA block
        # with flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func.
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_bias = None
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask != 0
            attn_bias = (~attention_mask)[:, None, None, :]
            attn_bias = attn_bias.to(dtype=q.dtype) * torch.finfo(q.dtype).min
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_bias,
            is_causal=False,
        )
        return x.transpose(1, 2).reshape(batch_size, seq_len, -1)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class ContinuousEmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Linear(vocab_dim, dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.constant_(self.embedding.bias, 0.0)

    def forward(self, x):
        return self.embedding(x)


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTPretrainedModel(PreTrainedModel):
    config_class = DiTConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['DDiTBlock']
    _supports_flash_attn_2 = _HAS_FLASH_ATTN
    _supports_sdpa = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class DiscreteDiTModel(DiTPretrainedModel):
    def __init__(self, config: DiTConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size

        if self.config.soft_inputs:
            # raise NotImplementedError('soft_inputs=True not implemented yet')
            print_rank_zero('Using soft inputs with ContinuousEmbeddingLayer')
            self.vocab_embed = ContinuousEmbeddingLayer(
                config.hidden_size,
                config.vocab_size,
            )
        else:
            self.vocab_embed = EmbeddingLayer(config.hidden_size, config.vocab_size)
        # self.vocab_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        # torch.nn.init.kaiming_normal_(self.vocab_embed.weight, a=math.sqrt(5))

        self.sigma_map = TimestepEmbedder(config.cond_hidden_size)
        self.rotary_emb = Rotary(config.hidden_size // config.num_attention_heads)

        blocks = [
            DDiTBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.cond_hidden_size,
                dropout=config.dropout,
                attention_backend=config.attn_backend,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.hidden_size,
            config.vocab_size,
            config.cond_hidden_size,
        )
        # self.scale_by_sigma = config.scale_by_sigma
        self.gradient_checkpointing = False

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

    def forward(self, input_ids, timesteps=None, latent=None, attention_mask=None, **kwargs: dict):
        # run vocab embed and conditioning in fp32
        # with torch.autocast(device_type=input_ids.device.type, enabled=False):
        sigma = timesteps
        if sigma is None:
            sigma = torch.zeros(input_ids.shape[0], device=input_ids.device)
        c = self.sigma_map(sigma)
        if latent is not None:
            # latent shape is (B, D), where B is the batch size and D is its dimension
            if latent.shape[-1] < self.config.cond_hidden_size:
                latent = F.pad(latent, (0, self.config.cond_hidden_size - latent.shape[-1]))
            elif latent.shape[-1] > self.config.cond_hidden_size:
                latent = latent[..., : self.config.cond_hidden_size]
            c += latent
        c = F.silu(c)
        if self.config.soft_inputs and len(input_ids.shape) == 2:
            # in the case where inputs are soft and we might be in the middle of the sampling process
            # with tokens as input_ids, we use one hot encoding to get the embeddings
            # print_rank_zero('Using one-hot encoding to make soft inputs')
            input_ids = F.one_hot(input_ids, num_classes=self.config.vocab_size).to(self.dtype)
        x = self.vocab_embed(input_ids)
        rotary_cos_sin = self.rotary_emb(x)

        for i in range(len(self.blocks)):
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    self.blocks[i].__call__,
                    x,
                    rotary_cos_sin,
                    c,
                    attention_mask,
                    None,
                )
            else:
                x = self.blocks[i](x, rotary_cos_sin, c, attention_mask=attention_mask, seqlens=None)

        # with torch.autocast(device_type='cuda', enabled=False):
        x = self.output_layer(x, c)

        return Output(logits=x)


class ContinuousDiTModel(DiTPretrainedModel):
    """
    inpus are continuous
    https://github.com/nnaisense/bayesian-flow-networks/blob/main/networks/adapters.py#L25
    """

    def __init__(self, config: DiTConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size

        self.vocab_embed = ContinuousEmbeddingLayer(
            config.hidden_size,
            config.vocab_size,
        )

        self.sigma_map = TimestepEmbedder(config.cond_hidden_size)
        self.rotary_emb = Rotary(config.hidden_size // config.num_attention_heads)

        blocks = [
            DDiTBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.cond_hidden_size,
                dropout=config.dropout,
                attention_backend=config.attn_backend,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.hidden_size,
            config.vocab_size,
            config.cond_hidden_size,
        )
        # self.scale_by_sigma = config.scale_by_sigma
        self.gradient_checkpointing = False

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

    def forward(self, x, timesteps=None, attention_mask=None, **kwargs):
        sigma = timesteps
        if sigma is None:
            sigma = torch.zeros(x.shape[0], device=x.device)
        x = self.vocab_embed(x.to(self.dtype))
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                if self.gradient_checkpointing and self.training:
                    x = self._gradient_checkpointing_func(
                        self.blocks[i].__call__,
                        x,
                        rotary_cos_sin,
                        c,
                        attention_mask,
                        None,
                    )
                else:
                    x = self.blocks[i](x, rotary_cos_sin, c, attention_mask=attention_mask, seqlens=None)

            x = self.output_layer(x, c)

        return Output(logits=x)


class MLPLatentEncoder(nn.Module):
    """Simple MLP Latent Encoder model. Takes existing latents as input"""

    def __init__(
        self,
        latent_dim,
        cond_hidden_size,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_hidden_size = cond_hidden_size
        self.mlp_ratio = mlp_ratio

        @torch.no_grad()
        def _init_linear_zero_(layer: nn.Linear) -> nn.Linear:
            """Zero-out a Linear layer so it outputs exactly 0 at init."""
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
            return layer

        self.latent_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 4),
            nn.SiLU(),
            nn.Linear(self.latent_dim * 4, self.cond_hidden_size),
            nn.SiLU(),
            _init_linear_zero_(nn.Linear(self.cond_hidden_size, self.cond_hidden_size)),
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        latent: torch.Tensor,
        **kwargs: dict,
    ) -> Output:
        """Forward pass for the model.

        :param latent: The input latent representation.
        """
        latent = self.latent_encoder(latent)

        return Output(logits=None, y_pred=latent, y_std=None)
