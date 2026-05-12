from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel, get_scheduler

from ..modules.ema import EMAModel
from ..utils import (
    get_optimizer_params,
    instantiate_from_config,
    instantiate_model_from_config,
    instantiate_optimizer_from_config,
    print_rank_zero,
)
from .baselm import BaseLM
from .output import SampleOutput




class LM(BaseLM):
    """Language model.

    This is a simple language model base class that can be used to generate text.
    """

    def __init__(
        self,
        model_config: dict,
        loss_config: dict,
        optimizer_config: dict,
        scheduler_config: dict | None = None,
        gradient_checkpointing: bool = False,
        peft_config: dict | None = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        torch_compile: bool = False,
        not_hf_pretrained: bool = False,
        **kwargs: dict,
    ) -> None:
        """Initialize the LM.

        Args:
            model_config: The model configuration.
            loss_config: The loss configuration.
            optimizer_config: The optimizer configuration.
            scheduler_config: The scheduler configuration.
            gradient_checkpointing: Whether to use gradient checkpointing.
            peft_config: The PEFT configuration.
            use_ema: Whether to use EMA.
            ema_decay: The EMA decay.
            torch_compile: Whether to use torch.compile.
            not_hf_pretrained: Whether to use non-HF pretrained models.

        """
        super().__init__(**kwargs)

        # save local args to self.hparams
        self.save_hyperparameters()

    def on_train_start(self) -> None:
        """Call when the training starts."""
        super().on_train_start()
        # Only initialize weights if we're not loading from a checkpoint
        if hasattr(self.model, 'init_weights') and (not hasattr(self.hparams, 'weights_only_path')):
            print_rank_zero('Initializing model weights...')
            self.model.init_weights()

    def configure_model(self) -> None:
        """Configure the model."""
        print_fn = self.print if self._trainer is not None else print
        self.model: PreTrainedModel = instantiate_model_from_config(
            self.hparams['model_config'],
            not_hf_pretrained=self.hparams['not_hf_pretrained'],
            peft_config=self.hparams['peft_config'],
        ).train()

        if self.hparams['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
            print_fn('Gradient checkpointing enabled.')
        if self.hparams['use_ema']:
            self.models_parameters_for_ema_update['default'] = list(self.model.parameters())
            # https://github.com/fadel/pytorch_ema
            self.ema_models['default'] = EMAModel(
                self.model.parameters(),
                decay=self.hparams['ema_decay'],
            )
            print_fn('EMA enabled.')
        if self.hparams['torch_compile']:
            print_fn('Compiling the model... (takes a ~minute)')
            self.model = torch.compile(model=self.model)
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = instantiate_from_config(self.hparams['loss_config'])

    def forward(self, batch: dict[str, Any], batch_idx: int, **kwargs: dict[str, Any]) -> dict:
        """Forward pass.

        Args:
            batch: The batch.
            batch_idx: The batch index.
            **kwargs: Additional keyword arguments.

        """
        return self.loss_fn(
            lightning_module=self,
            model=self.model,
            batch=batch,
            **kwargs,
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]] | list[torch.optim.Optimizer]:
        """Configure optimizers."""
        # instantiate optimizer
        optimizer_config = self.hparams['optimizer_config']
        loss_fn = self.loss_fn if isinstance(self.loss_fn, nn.Module) else None
        ignore_params = optimizer_config.pop('ignore_parameters', None)
        params = get_optimizer_params(self.model, loss_fn, ignore_params)
        optimizer = instantiate_optimizer_from_config(optimizer_config, params)
        # instantiate scheduler
        scheduler_config = self.hparams.get('scheduler_config')
        num_warmup_steps = scheduler_config.pop('num_warmup_steps', None)
        warmup_ratio = scheduler_config.pop('warmup_ratio', None)
        if num_warmup_steps is None and warmup_ratio is not None:
            num_warmup_steps = int(
                self.trainer.estimated_stepping_batches * warmup_ratio,
            )
        scheduler = get_scheduler(
            optimizer=optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=num_warmup_steps,
            **scheduler_config,
        )
        self.print(
            f'Setting up scheduler (estimated_stepping_batches: {self.trainer.estimated_stepping_batches})...',
        )
        scheduler = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'monitor': 'val/loss',
            },
        ]
        return [optimizer], scheduler

    def sample(
        self,
        batch: dict[str, Any],
        max_batch_size: int | None = None,
        skip_special_tokens: bool = False,
        **kwargs: dict,
    ) -> SampleOutput:
        """Generate samples from the model.

        Ema weights will be used if enabled.
        """
        raise NotImplementedError
