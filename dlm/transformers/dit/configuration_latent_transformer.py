from transformers.configuration_utils import PretrainedConfig


class LatentTransformerConfig(PretrainedConfig):
    model_type = 'transformer'

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        cond_hidden_size: int = 128,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        mask_token_id: int | None = None,
        x_time_type: str = 'time',
        y_time_type: str = 'time',
        y_latent_dim: int | None = None,
        y_num_patches: int = 1,
        zero_init_cross_attention: bool = False,
        # y_attends_x_only: bool = False,
        is_causal: bool = False,
        attention_bias: bool = False,
        normalize_embeddings: bool = False,
        n_kv_heads: int | None = None,
        max_sequence_length: int = 1024,
        tanh_out: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.cond_hidden_size = cond_hidden_size
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

        if mask_token_id is None:
            mask_token_id = vocab_size
        self.mask_token_id = mask_token_id
        # condition on probability of noise (alpha) or raw time (t)
        assert x_time_type.lower() in ['noise', 'time', 'none']
        self.x_time_type = x_time_type
        assert y_time_type.lower() in ['noise', 'time', 'none']
        self.y_time_type = y_time_type
        self.y_latent_dim = y_latent_dim
        self.y_num_patches = y_num_patches
        self.zero_init_cross_attention = zero_init_cross_attention
        # self.y_attends_x_only = y_attends_x_only
        self.is_causal = is_causal
        self.attention_bias = attention_bias
        self.normalize_embeddings = normalize_embeddings
        self.n_kv_heads = n_kv_heads
        self.max_sequence_length = max_sequence_length
        self.tanh_out = tanh_out

        if 'bos_token_id' not in kwargs:
            kwargs['bos_token_id'] = mask_token_id
        if 'eos_token_id' not in kwargs:
            kwargs['eos_token_id'] = mask_token_id

        assert self.normalize_embeddings is False, 'not implemented on these latent transformer yet.'

        super().__init__(**kwargs)
