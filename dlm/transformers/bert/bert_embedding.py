import torch
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

from ...utils import resolve_local_files_only
from ..dit import Output
from ..qwen3dit.qwen_embedding import (
    _adaptive_chunk_mean_keep_eos,
    _adaptive_chunk_mean_pool,
    _position_ids_from_attention_mask,
)


class BertEmbeddingConfig(PretrainedConfig):
    model_type = 'bert_embedding'

    def __init__(
        self,
        y_latent_dim: int = 512,
        y_latent_len: int | None = None,
        pretrained_model_name_or_path: str = 'bert-base-uncased',
        freeze: bool = True,
        local_files_only: bool = False,
        pooling_strategy: str = 'last_layer',
        normalize_embeddings: bool = True,
        replace_oov_with_unk: bool = True,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.y_latent_dim = y_latent_dim
        self.y_latent_len = y_latent_len
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.freeze = freeze
        self.local_files_only = local_files_only
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        self.replace_oov_with_unk = replace_oov_with_unk


class BertEmbeddingModel(PreTrainedModel):
    config_class = BertEmbeddingConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: BertEmbeddingConfig | None = None, **kwargs: dict):
        if config is None:
            config = BertEmbeddingConfig(**kwargs)
        super().__init__(config)

        self.pretrained_model_name_or_path = config.pretrained_model_name_or_path
        self.y_latent_dim = config.y_latent_dim
        self.y_latent_len = config.y_latent_len
        self.local_files_only = resolve_local_files_only(config.local_files_only)
        self.pooling_strategy = config.pooling_strategy
        self.normalize_embeddings = config.normalize_embeddings
        self.replace_oov_with_unk = config.replace_oov_with_unk

        self.bert_model = AutoModel.from_pretrained(
            self.pretrained_model_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.hidden_size = self.bert_model.config.hidden_size

        self._encoder_frozen = config.freeze
        if config.freeze:
            self.freeze_pretrained_model()

        self.gradient_checkpointing = False

    def freeze_pretrained_model(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.bert_model.eval()
        self._encoder_frozen = True

    def unfreeze_pretrained_model(self):
        for param in self.bert_model.parameters():
            param.requires_grad = True
        self.bert_model.train()
        self._encoder_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.bert_model.eval()
            if mode:
                super().train(False)
        return self

    def _sanitize_input_ids_for_encoder(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.replace_oov_with_unk:
            return input_ids
        vocab_size = self.bert_model.get_input_embeddings().weight.shape[0]
        if torch.max(input_ids) < vocab_size and torch.min(input_ids) >= 0:
            return input_ids
        unk_id = getattr(self.bert_model.config, 'unk_token_id', 0)
        return torch.where(
            (input_ids >= 0) & (input_ids < vocab_size),
            input_ids,
            torch.full_like(input_ids, unk_id),
        )

    def _apply_pooling_strategy(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == 'cls':
            pooled_output = hidden_states[:, 0].unsqueeze(1)
        elif self.pooling_strategy == 'mean':
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden_states.dtype)
            pooled_output = (masked_hidden.sum(dim=1) / denom).unsqueeze(1)
        elif self.pooling_strategy == 'last_layer':
            pooled_output = hidden_states
        elif self.pooling_strategy == 'adaptive_chunk_mean':
            assert self.y_latent_len is not None, 'y_latent_len (S\') must be set for adaptive_chunk_mean'
            assert (self.y_latent_dim % self.y_latent_len) == 0, (
                f'y_latent_dim must be divisible by y_latent_len. got {self.y_latent_dim=} and {self.y_latent_len=}'
            )
            pooled_output = _adaptive_chunk_mean_pool(hidden_states, attention_mask, out_len=self.y_latent_len)
        elif self.pooling_strategy == 'adaptive_chunk_mean_eos':
            assert self.y_latent_len is not None, 'y_latent_len (S\') must be set for adaptive_chunk_mean_eos'
            assert (self.y_latent_dim % self.y_latent_len) == 0, (
                f'y_latent_dim must be divisible by y_latent_len. got {self.y_latent_dim=} and {self.y_latent_len=}'
            )
            pooled_output = _adaptive_chunk_mean_keep_eos(hidden_states, attention_mask, out_len=self.y_latent_len)
        else:
            raise ValueError(f'Unknown pooling strategy: {self.pooling_strategy}')
        return pooled_output

    def _clip_pooled_output(self, pooled_output: torch.Tensor) -> torch.Tensor:
        s_pool = pooled_output.shape[-2]
        h_pool = pooled_output.shape[-1]

        if self.pooling_strategy in ('adaptive_chunk_mean', 'adaptive_chunk_mean_eos'):
            d_per_token = self.y_latent_dim // s_pool
            if d_per_token > h_pool:
                raise ValueError(
                    f'Requested per-token dim ({d_per_token}) exceeds hidden size ({h_pool}). '
                    f'Lower y_latent_dim or increase encoder hidden size.',
                )
            pooled_output = pooled_output[..., :d_per_token]
        else:
            total_available = s_pool * h_pool
            if self.y_latent_dim < total_available:
                assert (self.y_latent_dim % s_pool) == 0, (
                    f'y_latent_dim must be divisible by pooled seq len. got {self.y_latent_dim=} and {s_pool=}'
                )
                d_per_token = self.y_latent_dim // s_pool
                pooled_output = pooled_output[..., :d_per_token]
            elif self.y_latent_dim > total_available:
                raise ValueError(
                    f'y_latent_dim ({self.y_latent_dim}) > available ({total_available}) '
                    f'from (S_pool={s_pool}, H_pool={h_pool}).',
                )
        return pooled_output

    def post_process_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_output = self._apply_pooling_strategy(hidden_states, attention_mask)
        pooled_output = self._clip_pooled_output(pooled_output)

        if self.normalize_embeddings:
            pooled_output = F.normalize(pooled_output, p=2, dim=-1)

        return pooled_output

    def forward(
        self,
        x_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        masked_tokens_pos: torch.Tensor | None = None,
        p_r: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> Output:
        x_input_ids = self._sanitize_input_ids_for_encoder(x_input_ids)

        inputs_embeds = None
        token_zero_mask = None
        position_ids = kwargs.pop('position_ids', None)
        use_inputs_embeds = p_r is not None

        if use_inputs_embeds:
            emb_layer = self.bert_model.get_input_embeddings()
            inputs_embeds = emb_layer(x_input_ids)

            if p_r is not None:
                assert masked_tokens_pos is not None, 'masked_tokens_pos must be provided when p_r is used'
                apply_mask = (torch.rand_like(p_r) < p_r).to(torch.bool)
                token_zero_mask = masked_tokens_pos & apply_mask.unsqueeze(-1)
                inputs_embeds = inputs_embeds.masked_fill(token_zero_mask.unsqueeze(-1), 0.0)

            if position_ids is None and attention_mask is not None:
                position_ids = _position_ids_from_attention_mask(attention_mask)

        if use_inputs_embeds:
            bert_kwargs = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                **kwargs,
            }
        else:
            bert_kwargs = {
                'input_ids': x_input_ids,
                'attention_mask': attention_mask,
                **kwargs,
            }

        if position_ids is not None:
            bert_kwargs['position_ids'] = position_ids

        bert_outputs = self.bert_model(**bert_kwargs)
        hidden_states = bert_outputs.last_hidden_state

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        pooled_output = self.post_process_hidden_states(hidden_states, attention_mask)

        if (p_r is not None) and (self.pooling_strategy == 'last_layer'):
            pooled_output = pooled_output.masked_fill(token_zero_mask.unsqueeze(-1), 0.0)

        y_pred = pooled_output.reshape(pooled_output.shape[0], -1)
        return Output(logits=None, y_pred=y_pred, y_std=None)
