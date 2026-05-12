from transformers.configuration_utils import PretrainedConfig


class LatentDiTConfig(PretrainedConfig):
    model_type = 'dit'

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
        # y_attends_x_only: bool = False,
        is_causal: bool = False,
        attention_bias: bool = False,
        normalize_embeddings: bool = False,
        flatten_outputs: bool = True,
        soft_inputs: bool = False,
        y_output_dim: int | None = None,
        head_hidden_size: int | None = None,
        head_num_layers: int | None = None,
        head_num_attention_heads: int | None = None,
        attn_backend: str = 'auto',
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
        # self.y_attends_x_only = y_attends_x_only
        self.is_causal = is_causal
        self.attention_bias = attention_bias
        self.normalize_embeddings = normalize_embeddings
        self.flatten_outputs = flatten_outputs
        self.soft_inputs = soft_inputs
        self.y_output_dim = y_output_dim
        self.head_hidden_size = head_hidden_size
        self.head_num_layers = head_num_layers
        self.head_num_attention_heads = head_num_attention_heads
        self.attn_backend = attn_backend

        super().__init__(
            bos_token_id=mask_token_id,
            eos_token_id=mask_token_id,
            **kwargs,
        )
