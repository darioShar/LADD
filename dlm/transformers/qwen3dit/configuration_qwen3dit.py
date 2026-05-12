from transformers.models.qwen3 import Qwen3Config


class Qwen3DiTConfig(Qwen3Config):
    def __init__(
        self,
        time_conditioning: bool = True,
        cond_hidden_size: int | None = None,
        use_liger_kernel: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_conditioning = time_conditioning
        if cond_hidden_size is None:
            self.cond_hidden_size = self.hidden_size // 6
        else:
            self.cond_hidden_size = cond_hidden_size
        self.use_liger_kernel = use_liger_kernel
