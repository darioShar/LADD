import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    KwargsForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .configuration_qwen3dit import Qwen3DiTConfig

try:
    from liger_kernel.transformers import (
        LigerRMSNorm,
        LigerSwiGLUMLP,
        liger_rotary_pos_emb,
    )

    enable_liger = True
except ImportError:
    enable_liger = False


logger = logging.get_logger(__name__)


@dataclass
class DiTModelOutputWithPast(BaseModelOutputWithPast):
    conditioning: torch.Tensor | None = None


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# @torch.jit.script
def modulate_fused(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return modulate(x, shift, scale)


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


# @torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


# @torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


class Qwen3DiTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3DiTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            'head_dim',
            config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0  # config.attention_dropout
        self.is_causal = False  # True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        rmsnorm_cls = LigerRMSNorm if config.use_liger_kernel else Qwen3RMSNorm
        self.q_norm = rmsnorm_cls(
            self.head_dim,
            eps=config.rms_norm_eps,
        )  # unlike olmo, only on the head dim!
        self.k_norm = rmsnorm_cls(
            self.head_dim,
            eps=config.rms_norm_eps,
        )  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, 'sliding_window', None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None
        if config.use_liger_kernel:
            self.apply_rotary_f = liger_rotary_pos_emb
        else:
            self.apply_rotary_f = apply_rotary_pos_emb

    def get_qkv(self, hidden_states, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape),
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape),
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_f(
            query_states,
            key_states,
            cos,
            sin,
        )
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        enable_diffusion_mask: bool | None = False,
        sampling_mode: bool | None = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        if enable_diffusion_mask and not sampling_mode:
            seq_len = attention_mask.shape[-1] // 2
            query_states, key_states, value_states = self.get_qkv(
                hidden_states[:, :seq_len],
                position_embeddings,
            )
            query_states_clean, key_states_clean, value_states_clean = self.get_qkv(
                hidden_states[:, seq_len:],
                position_embeddings,
            )
            # concatenate across the sequence length
            query_states = torch.cat([query_states, query_states_clean], dim=2)
            key_states = torch.cat([key_states, key_states_clean], dim=2)
            value_states = torch.cat([value_states, value_states_clean], dim=2)
        else:
            query_states, key_states, value_states = self.get_qkv(
                hidden_states,
                position_embeddings,
            )

        cos, sin = position_embeddings
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, 'sliding_window', None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != 'eager':
            if self.config._attn_implementation == 'sdpa' and kwargs.get(
                'output_attentions',
                False,
            ):
                logger.warning_once(
                    '`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to '
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.',
                )
            elif self.config._attn_implementation == 'flash_attention_2':
                if attention_mask is not None and len(attention_mask.shape) == 4:
                    attention_mask = None
                    self.is_causal = False
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DiTDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3DiTConfig, layer_idx: int):
        super().__init__()
        self.time_conditioning = config.time_conditioning
        self.hidden_size = config.hidden_size
        self.dropout = config.attention_dropout
        self.self_attn = Qwen3DiTAttention(config=config, layer_idx=layer_idx)
        if config.use_liger_kernel:
            self.mlp = LigerSwiGLUMLP(config)
        else:
            self.mlp = Qwen3MLP(config)
        rmsnorm_cls = LigerRMSNorm if config.use_liger_kernel else Qwen3RMSNorm
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        if config.sliding_window and config._attn_implementation != 'flash_attention_2':
            logger.warning_once(
                f'Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; '
                'unexpected results may be encountered.',
            )
        if config.time_conditioning:
            self.adaLN_modulation = nn.Linear(
                config.cond_hidden_size,
                6 * config.hidden_size,
                bias=True,
            )
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        enable_diffusion_mask: bool | None = False,
        sampling_mode: bool | None = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor,
        tuple[torch.FloatTensor, torch.FloatTensor] | None,
    ]:
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        residual = hidden_states
        if self.time_conditioning:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
                6, dim=1
            )

        hidden_states = self.input_layernorm(hidden_states)
        if self.time_conditioning:
            hidden_states = modulate_fused(hidden_states, shift_msa, scale_msa)
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            enable_diffusion_mask=enable_diffusion_mask,
            sampling_mode=sampling_mode,
            **kwargs,
        )
        if self.time_conditioning:
            hidden_states = bias_dropout_scale_fn(
                hidden_states,
                bias=None,
                scale=gate_msa.unsqueeze(1),
                residual=residual,
                prob=self.dropout,
            )
        else:
            hidden_states = residual + F.dropout(hidden_states, p=self.dropout)
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.time_conditioning:
            hidden_states = modulate_fused(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        if self.time_conditioning:
            hidden_states = bias_dropout_scale_fn(
                hidden_states,
                bias=None,
                scale=gate_mlp.unsqueeze(1),
                residual=residual,
                prob=self.dropout,
            )
        else:
            hidden_states = residual + F.dropout(hidden_states, p=self.dropout)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half,
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype,
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class Qwen3DiTModel(Qwen3Model):
    def __init__(self, config: Qwen3DiTConfig):
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        if config.time_conditioning:
            self.time_embedder = TimestepEmbedder(config.cond_hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DiTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)],
        )
        rmsnorm_cls = LigerRMSNorm if config.use_liger_kernel else Qwen3RMSNorm
        self.norm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timesteps: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        enable_diffusion_mask: bool | None = False,
        clean_embeds: torch.FloatTensor | None = None,
        sampling_mode: bool | None = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | DiTModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                'You must specify exactly one of input_ids or inputs_embeds',
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.',
            )
            use_cache = False

        # Kept for users that pass a legacy cache.
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                'The `past_key_values` should be either a `Cache` object or `None`.',
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None and len(attention_mask.shape) == 4:
            causal_mask = attention_mask
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
            )

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if timesteps is None:
            timesteps = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
        if self.config.time_conditioning:
            conditioning = F.silu(self.time_embedder(timesteps))
        else:
            conditioning = None
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    conditioning,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    conditioning=conditioning,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return DiTModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            conditioning=conditioning,
        )


class Qwen3DiTForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ['lm_head.weight']
    _tp_plan = {'lm_head': 'colwise_rep'}
    _pp_plan = {'lm_head': (['hidden_states'], ['logits'])}

    def __init__(self, config: Qwen3DiTConfig):
        super().__init__(config)
        if config.use_liger_kernel:
            assert enable_liger, 'Liger kernel is not installed'
        self.model = Qwen3DiTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.lm_head.weight.data.zero_()
        if config.time_conditioning:
            self.adaLN_modulation = nn.Linear(
                config.cond_hidden_size,
                2 * config.hidden_size,
                bias=True,
            )
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        timesteps: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        conditioning = outputs.conditioning
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if self.config.time_conditioning:
            shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
            hidden_states = modulate(hidden_states, shift, scale)

        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
