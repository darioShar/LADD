import torch
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

from ...utils import print_rank_zero, resolve_local_files_only
from ..dit import Output


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-pad token, accounting for left/right padding."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _effective_ranks_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute per-token rank among valid tokens.

    Returns ranks in [0, L_eff-1] for valid tokens, and -1 for pad tokens.
    Works for both left and right padding.
    """
    # attention_mask: (B, L) in {0,1}
    am = attention_mask.to(torch.long)
    ranks = torch.cumsum(am, dim=1) - 1  # pads become -1 before first valid for left pad too
    ranks = torch.where(am.bool(), ranks, torch.full_like(ranks, -1))
    return ranks


def _adaptive_chunk_mean_pool(
    hidden_states: torch.Tensor,  # (B, L, H)
    attention_mask: torch.Tensor,  # (B, L), bool/int
    out_len: int,  # S'
) -> torch.Tensor:
    """Adaptive masked pooling into `out_len` contiguous chunks per sample.

    For each sample with effective length L_eff, assign each valid token with rank r in [0..L_eff-1]
    to bin: floor(r * out_len / L_eff). This works when L_eff is not divisible by out_len.

    Output shape: (B, out_len, H). For samples where L_eff < out_len, returns all zeros for that
    specific sample only (per-sample handling). Valid samples are processed normally.
    """
    if out_len <= 0:
        raise ValueError(f'out_len must be > 0, got {out_len}')

    b, l, h = hidden_states.shape
    am = attention_mask.to(hidden_states.device)
    am_bool = am.bool()
    lengths = am_bool.sum(dim=1)  # (B,)

    if (lengths == 0).any():
        raise ValueError('Found empty sequence (all pads) in batch; cannot pool.')

    # Initialize output tensor
    pooled = torch.zeros((b, out_len, h), device=hidden_states.device, dtype=hidden_states.dtype)

    # Identify samples with sufficient length
    valid_samples = lengths >= out_len  # (B,)

    # If no valid samples, return all zeros
    if not valid_samples.any():
        return pooled

    # Process only valid samples
    valid_indices = torch.where(valid_samples)[0]

    # Extract valid samples
    hs_valid = hidden_states[valid_samples]  # (B_valid, L, H)
    am_valid = am_bool[valid_samples]  # (B_valid, L)
    lengths_valid = lengths[valid_samples]  # (B_valid,)

    # ranks: (B_valid, L) with -1 for pads, 0..L_eff-1 for valids
    ranks = _effective_ranks_from_attention_mask(am_valid)

    # bin = floor(rank * out_len / L_eff)
    # use integer math: (rank * out_len) // L_eff
    lengths_safe = lengths_valid.clamp_min(1).unsqueeze(1)  # (B_valid, 1)
    bin_ids = (ranks.clamp_min(0) * out_len) // lengths_safe  # (B_valid, L)
    bin_ids = torch.where(am_valid, bin_ids, torch.full_like(bin_ids, -1))  # keep pads at -1

    # scatter-add sums
    b_valid = hs_valid.shape[0]
    sums = torch.zeros((b_valid, out_len, h), device=hidden_states.device, dtype=hidden_states.dtype)
    counts = torch.zeros((b_valid, out_len, 1), device=hidden_states.device, dtype=hidden_states.dtype)

    idx = bin_ids.clamp_min(0).unsqueeze(-1).expand(-1, -1, h)  # (B_valid, L, H)
    masked_hs = hs_valid * am_valid.unsqueeze(-1).to(hidden_states.dtype)

    sums.scatter_add_(dim=1, index=idx, src=masked_hs)

    idx_c = bin_ids.clamp_min(0).unsqueeze(-1)  # (B_valid, L, 1)
    counts.scatter_add_(
        dim=1,
        index=idx_c,
        src=am_valid.unsqueeze(-1).to(hidden_states.dtype),
    )

    pooled_valid = sums / counts.clamp_min(1.0)

    # Place valid pooled results back into output tensor
    pooled[valid_samples] = pooled_valid

    # print the proportion of valid_samples in the batch
    # proportion_valid = valid_samples.float().mean().item()
    # print_rank_zero(f"[QwenEmbedding] Proportion of valid samples for adaptive_chunk_mean_pool: {proportion_valid:.4f} ({valid_samples.sum().item()}/{b})")


    return pooled


def _adaptive_chunk_mean_keep_eos(
    hidden_states: torch.Tensor,  # (B, L, H)
    attention_mask: torch.Tensor,  # (B, L)
    out_len: int,  # S'
) -> torch.Tensor:
    """Adaptive masked pooling where the last output token is the EOS embedding.

    Assumes the *last valid token* (according to attention_mask) is EOS (as in your dataset formatting).
    Pools all valid tokens except that last one into (out_len - 1) tokens, and appends EOS embedding.

    Output: (B, out_len, H). For samples where L_eff < out_len or L_eff < 2, returns all zeros for
    that specific sample only (per-sample handling). Valid samples are processed normally.
    """
    if out_len < 2:
        raise ValueError('adaptive_chunk_mean_eos requires out_len >= 2')

    b, l, h = hidden_states.shape
    am = attention_mask.bool()
    lengths = am.sum(dim=1)  # (B,)

    if (lengths == 0).any():
        raise ValueError('Found empty sequence (all pads) in batch; cannot pool.')

    # Initialize output tensor
    pooled = torch.zeros((b, out_len, h), device=hidden_states.device, dtype=hidden_states.dtype)

    # Identify samples with sufficient length (and at least 2 tokens for prefix + eos)
    valid_samples = (lengths >= out_len) & (lengths >= 2)  # (B,)

    # If no valid samples, return all zeros
    if not valid_samples.any():
        return pooled

    # Process only valid samples
    hs_valid = hidden_states[valid_samples]  # (B_valid, L, H)
    am_valid = am[valid_samples]  # (B_valid, L)
    lengths_valid = lengths[valid_samples]  # (B_valid,)

    b_valid, l_valid, h_valid = hs_valid.shape
    ranks = _effective_ranks_from_attention_mask(am_valid)  # (B_valid, L), pads -1, valids 0..L_eff-1
    eos_rank = lengths_valid - 1  # (B_valid,)
    # eos position = where rank == eos_rank
    eos_pos = (ranks == eos_rank.unsqueeze(1))  # (B_valid, L) bool
    if not eos_pos.any():
        raise ValueError('Could not locate EOS position from attention_mask.')

    eos_emb = (hs_valid * eos_pos.unsqueeze(-1).to(hs_valid.dtype)).sum(dim=1, keepdim=True)  # (B_valid,1,H)

    # Build prefix mask by dropping eos position
    prefix_mask = am_valid & (~eos_pos)  # (B_valid, L)
    prefix_lengths = prefix_mask.sum(dim=1)
    if (prefix_lengths == 0).any():
        raise ValueError('Some samples have no prefix tokens after removing EOS.')

    prefix_pooled = _adaptive_chunk_mean_pool(hs_valid, prefix_mask, out_len=out_len - 1)  # (B_valid,S'-1,H)
    pooled_valid = torch.cat([prefix_pooled, eos_emb], dim=1)  # (B_valid,S',H)

    # Place valid pooled results back into output tensor
    pooled[valid_samples] = pooled_valid

    return pooled


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.to(torch.long).cumsum(dim=1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids


class QwenEmbeddingConfig(PretrainedConfig):
    model_type = 'qwen_embedding'

    def __init__(
        self,
        y_latent_dim: int = 512,  # total flattened dimension (B, D_total)
        y_latent_len: int | None = None,  # S' for strategies producing sequences
        pretrained_model_name_or_path: str = 'Qwen/Qwen3-Embedding-0.6B',
        freeze: bool = True,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        pooling_strategy: str = 'last_token',
        normalize_embeddings: bool = True,
        attn_backend: str = 'flash_attention_2',  # auto, flash, sdpa -> flash_attention_2, eager, sdpa
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.y_latent_dim = y_latent_dim
        self.y_latent_len = y_latent_len
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.freeze = freeze
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        if attn_backend == 'auto':
            # check if flash attention is available; if not, default to sdpa for stability
            try:
                from transformers.utils import is_flash_attn_2_available
                if is_flash_attn_2_available():
                    self.attn_backend = 'flash_attention_2'
                else:
                    print_rank_zero(
                        '[QwenEmbeddingConfig] Flash Attention not available; defaulting to sdpa backend.',
                    )
                    self.attn_backend = 'sdpa'
            except ImportError:
                print_rank_zero(
                    '[QwenEmbeddingConfig] Flash Attention not installed; defaulting to sdpa backend.',
                )
                self.attn_backend = 'sdpa'
        elif attn_backend in ('flash', 'flash_attention_2'):
            self.attn_backend = 'flash_attention_2'
        elif attn_backend in ('eager', 'sdpa'):
            self.attn_backend = attn_backend
        else:
            raise ValueError(f'Unknown attn_backend: {attn_backend}')


class QwenEmbeddingModel(PreTrainedModel):
    config_class = QwenEmbeddingConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: QwenEmbeddingConfig | None = None, **kwargs: dict):
        if config is None:
            config = QwenEmbeddingConfig(**kwargs)
        super().__init__(config)

        self.pretrained_model_name_or_path = config.pretrained_model_name_or_path
        self.y_latent_dim = config.y_latent_dim
        self.y_latent_len = config.y_latent_len
        self.trust_remote_code = config.trust_remote_code
        self.local_files_only = resolve_local_files_only(config.local_files_only)
        self.pooling_strategy = config.pooling_strategy
        self.normalize_embeddings = config.normalize_embeddings

        # Flash Attention 2 requires bf16/fp16 dtype, not fp32
        # When using Lightning bf16-mixed precision, load model in bf16
        self.qwen_model = AutoModel.from_pretrained(
            self.pretrained_model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=config.attn_backend,  # flash_attention_2, eager, sdpa
            local_files_only=self.local_files_only,
        )

        self.hidden_size = self.qwen_model.config.hidden_size

        self._encoder_frozen = config.freeze
        if config.freeze:
            self.freeze_pretrained_model()

        self.gradient_checkpointing = False

    def freeze_pretrained_model(self):
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        self.qwen_model.eval()
        self._encoder_frozen = True

    def unfreeze_pretrained_model(self):
        for param in self.qwen_model.parameters():
            param.requires_grad = True
        self.qwen_model.train()
        self._encoder_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.qwen_model.eval()
            if mode:
                super().train(False)
        return self

    def _apply_pooling_strategy(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to hidden states.
        hidden_states: (B, L, H)
        attention_mask: (B, L)
        """
        # Pool to (B, S_pool, H_pool)
        if self.pooling_strategy == 'last_token':
            pooled_output = last_token_pool(hidden_states, attention_mask).unsqueeze(1)  # (B, 1, H)

        elif self.pooling_strategy == 'mean':
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden_states.dtype)
            pooled_output = (masked_hidden.sum(dim=1) / denom).unsqueeze(1)  # (B, 1, H)

        elif self.pooling_strategy == 'last_layer':
            pooled_output = hidden_states  # (B, L, H)

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
        """Clip pooled output to match y_latent_dim.
        pooled_output: (B, S_pool, H_pool)
        """
        # Clip last dim so that flattened output is exactly y_latent_dim when S_pool == y_latent_len
        s_pool = pooled_output.shape[-2]
        h_pool = pooled_output.shape[-1]

        if self.pooling_strategy in ('adaptive_chunk_mean', 'adaptive_chunk_mean_eos'):
            # here s_pool == y_latent_len by construction
            d_per_token = self.y_latent_dim // s_pool
            if d_per_token > h_pool:
                raise ValueError(
                    f'Requested per-token dim ({d_per_token}) exceeds hidden size ({h_pool}). '
                    f'Lower y_latent_dim or increase encoder hidden size.',
                )
            pooled_output = pooled_output[..., :d_per_token]  # (B, S', d_per_token)

        else:
            # keep your existing “flatten then clip” semantics for other strategies
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
        """Pool and normalize hidden states according to config.
        hidden_states: (B, L, H)
        attention_mask: (B, L)
        """

        pooled_output = self._apply_pooling_strategy(hidden_states, attention_mask)  # (B, S_pool, H_pool)
        pooled_output = self._clip_pooled_output(pooled_output)  # (B, S', d_per_token)

        # Report non-finite pooled outputs before normalization.
        if torch.isnan(pooled_output).any():
            print_rank_zero(f"[QwenEmbedding] NaN in pooled_output BEFORE normalization! Shape: {pooled_output.shape}")
            print_rank_zero(f"[QwenEmbedding] NaN count: {torch.isnan(pooled_output).sum().item()}")
            print_rank_zero(f"[QwenEmbedding] First few values: {pooled_output.flatten()[:20]}")

        if self.normalize_embeddings:
            pooled_output = F.normalize(pooled_output, p=2, dim=-1)

        return pooled_output

    def forward(
        self,
        x_input_ids: torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor | None = None,  # (B, L)
        masked_tokens_pos: torch.Tensor | None = None,
        p_r: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> Output:

        inputs_embeds = None
        token_zero_mask = None
        position_ids = kwargs.pop('position_ids', None)
        use_inputs_embeds = p_r is not None

        if use_inputs_embeds:
            emb_layer = self.qwen_model.get_input_embeddings()
            inputs_embeds = emb_layer(x_input_ids)  # (B, L, H)

            if p_r is not None:
                apply_mask = (torch.rand_like(p_r) < p_r).to(torch.bool)  # (B,)
                token_zero_mask = masked_tokens_pos & apply_mask.unsqueeze(-1)  # (B, L) bool
                inputs_embeds = inputs_embeds.masked_fill(token_zero_mask.unsqueeze(-1), 0.0)

            if position_ids is None and attention_mask is not None:
                position_ids = _position_ids_from_attention_mask(attention_mask)

        if use_inputs_embeds:
            qwen_kwargs = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                **kwargs,
            }
            if position_ids is not None:
                qwen_kwargs['position_ids'] = position_ids
            qwen_outputs = self.qwen_model(**qwen_kwargs)
        else:
            qwen_kwargs = {
                'input_ids': x_input_ids,
                'attention_mask': attention_mask,
                **kwargs,
            }
            if position_ids is not None:
                qwen_kwargs['position_ids'] = position_ids
            qwen_outputs = self.qwen_model(**qwen_kwargs)
        hidden_states = qwen_outputs.last_hidden_state  # (B, L, H)

        if attention_mask is None:
            # If you want variable-length safe pooling, you should always pass attention_mask.
            # We keep existing behavior for backwards compatibility.
            attention_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        inputs_has_nan = torch.isnan(inputs_embeds).any() if inputs_embeds is not None else False
        if torch.isnan(hidden_states).any() or inputs_has_nan:
            hidden_has_nan = torch.isnan(hidden_states).any()
            print_rank_zero(
                f'[QwenEmbedding] NaN detected pre-pool: inputs_embeds={inputs_has_nan}, hidden_states={hidden_has_nan}',
            )
            if attention_mask is not None:
                valid_mask = attention_mask.to(torch.bool)
                pad_mask = ~valid_mask
                hidden_token_nan = torch.isnan(hidden_states).any(dim=-1)
                valid_nan = (hidden_token_nan & valid_mask).sum().item()
                pad_nan = (hidden_token_nan & pad_mask).sum().item()
                print_rank_zero(f'[QwenEmbedding] NaN tokens: valid={valid_nan}, pad={pad_nan}')
            if inputs_embeds is not None:
                inputs_dtype = inputs_embeds.dtype
            else:
                inputs_dtype = None
            print_rank_zero(f'[QwenEmbedding] dtypes: inputs_embeds={inputs_dtype}, hidden_states={hidden_states.dtype}')

        # Flash Attention can return NaNs for masked (padding) tokens; clear them before pooling.
        # pad_mask = attention_mask == 0
        # if pad_mask.any():
        #     hidden_states = hidden_states.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        pooled_output = self.post_process_hidden_states(hidden_states, attention_mask)

        # --- Post-model masking (ONLY valid for last_layer, since it preserves token alignment) ---
        if (p_r is not None) and (self.pooling_strategy == 'last_layer'):
            # Zero-out last-layer hidden states at those token positions
            # pooled_output: (B, S, H)
            pooled_output = pooled_output.masked_fill(token_zero_mask.unsqueeze(-1), 0.0)

        y_pred = pooled_output.reshape(pooled_output.shape[0], -1)  # (B, y_latent_dim)

        # Report non-finite Qwen outputs.
        if torch.isnan(y_pred).any():
            print_rank_zero(f"[QwenEmbedding] NaN detected in y_pred! Shape: {y_pred.shape}")
            print_rank_zero(f"[QwenEmbedding] NaN count: {torch.isnan(y_pred).sum().item()}")
            print_rank_zero(f"[QwenEmbedding] pooling_strategy: {self.pooling_strategy}")
            print_rank_zero(f"[QwenEmbedding] normalize_embeddings: {self.normalize_embeddings}")
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1)
                print_rank_zero(f"[QwenEmbedding] Sequence lengths: min={lengths.min().item()}, max={lengths.max().item()}, mean={lengths.float().mean().item():.1f}")
                print_rank_zero(f"[QwenEmbedding] y_latent_len: {self.y_latent_len}")

        return Output(logits=None, y_pred=y_pred, y_std=None)
