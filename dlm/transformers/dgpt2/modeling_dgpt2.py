"""
Same as GPT2, but with a self attention layer.

ref: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from .configuration_dgpt2 import DGPT2Config

import flash_attn


@dataclass
class Output:
    logits: torch.Tensor


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
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
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


class DGPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = Conv1D(3 * config.hidden_size, config.hidden_size)
        self.c_proj = Conv1D(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation in ["flash_attention_2", "sdpa"]

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        if self._attn_implementation == "flash_attention_2":
            x = flash_attn.flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=False,
            )
        elif self._attn_implementation == "sdpa":
            q = q.transpose(1, 2)  # (B, nh, T, hs)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )
            x = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                f"Invalid attention implementation: {self._attn_implementation}"
            )
        x = x.view(B, T, -1)
        x = self.resid_dropout(self.c_proj(x))
        return x


class DGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DGPT2Block(nn.Module):
    def __init__(self, config: DGPT2Config, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = DGPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = DGPT2MLP(4 * config.hidden_size, config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class DGPT2PreTrainedModel(PreTrainedModel):
    config_class = DGPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DGPT2Block"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        GPT2PreTrainedModel._init_weights(self, module)


class DGPT2Model(DGPT2PreTrainedModel):
    def __init__(self, config: DGPT2Config):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        if config.continuous_input:
            self.wte = nn.Linear(config.vocab_size, self.embed_dim)
        else:
            self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        if config.time_type != "none":
            self.time_embed = TimestepEmbedder(self.embed_dim)
        else:
            self.time_embed = None
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [DGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(self, input_ids, attention_mask=None, timesteps=None, **kwargs):
        input_shape = input_ids.size()
        position_ids = torch.arange(
            0, input_shape[-1], dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0)  # (1, seq_len)
        x = self.wte(input_ids) + self.wpe(position_ids)
        if self.time_embed is not None and timesteps is not None:
            x = x + self.time_embed(timesteps)
        x = self.drop(x)

        for block in self.h:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    block.__call__,
                    x,
                    attention_mask,
                )
            else:
                x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        return x


class DGPT2LMHeadModel(DGPT2PreTrainedModel):
    def __init__(self, config: DGPT2Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.transformer = DGPT2Model(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids, attention_mask=None, timesteps=None, **kwargs):
        x = self.transformer(
            input_ids, attention_mask=attention_mask, timesteps=timesteps
        )
        x = self.lm_head(x)
        return Output(logits=x)
