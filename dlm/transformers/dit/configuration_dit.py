from transformers.configuration_utils import PretrainedConfig


class DiTConfig(PretrainedConfig):
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
        time_type: str = 'time',
        is_causal: bool = False,
        attention_bias: bool = False,
        soft_inputs: bool = False,
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
        assert time_type.lower() in ['noise', 'time', 'none']
        # Convert string 'none' to actual None
        self.time_type = time_type # None if time_type.lower() == 'none' else time_type
        self.is_causal = is_causal
        self.attention_bias = attention_bias
        self.soft_inputs = soft_inputs
        self.attn_backend = attn_backend
        super().__init__(
            bos_token_id=mask_token_id,
            eos_token_id=mask_token_id,
            **kwargs,
        )
