from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel, get_scheduler

from ..modules.diffusionmodules.masked_loss import MaskedDiffusionLoss
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


class DiffusionLM(BaseLM):
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
        torch_compile: str | None = None,
        not_hf_pretrained: bool = False,
        sampler_config: dict | None = None,
        conditional: bool = False,
        encoder_config: dict | None = None,
        custom_encoder: bool = False,
        use_k_first_input_ids_as_prompt: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # save local args to self.hparams
        self.save_hyperparameters()

        if sampler_config is None:
            sampler_config = {
                'target': 'dlm.modules.diffusionmodules.sampling.AbsorbingSampler',
            }
        self.sampler = instantiate_from_config(sampler_config)
        self.conditional = conditional
        self.encoder_config = encoder_config
        self.custom_encoder = custom_encoder

    def configure_model(self):
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
        if self.hparams['torch_compile'] is not None:
            print_fn('Compiling the model... (takes a ~minute)')
            self.model.compile(mode=self.hparams['torch_compile'])
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = instantiate_from_config(self.hparams['loss_config'])

        if self.conditional:
            if self.custom_encoder:
                self.encoder: nn.Module = instantiate_model_from_config(
                    self.hparams['encoder_config'],
                    not_hf_pretrained=self.hparams.get('not_hf_pretrained', False),
                )
            else:
                self.encoder: nn.Module = instantiate_from_config(self.encoder_config)

            if self.hparams['gradient_checkpointing']:
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False},
                )
            if self.hparams['use_ema']:
                # https://github.com/fadel/pytorch_ema
                self.models_parameters_for_ema_update['encoder'] = list(self.encoder.parameters())
                self.ema_models['encoder'] = EMAModel(
                    self.encoder.parameters(),
                    decay=self.hparams['ema_decay'],
                )
            if self.hparams['torch_compile'] is not None:
                print('Compiling the encoder... (takes a ~minute)')
                self.encoder.compile(mode=self.hparams['torch_compile'])

        # Set the mask token id for the model
        if isinstance(self.loss_fn, MaskedDiffusionLoss):
            if getattr(self.loss_fn, 'mask_token_id', None) is not None:
                self.model.config.mask_token_id = self.loss_fn.mask_token_id
            else:
                assert hasattr(self.model.config, 'mask_token_id'), 'mask_token_id must be provided'

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]] | list[torch.optim.Optimizer]:
        """Configure optimizers."""
        optimizers = []
        schedulers = []

        # retrieve some parameters
        scheduler_config = self.hparams.get('scheduler_config')
        num_warmup_steps = scheduler_config.pop('num_warmup_steps', None)
        warmup_ratio = scheduler_config.pop('warmup_ratio', None)
        if num_warmup_steps is None and warmup_ratio is not None:
            num_warmup_steps = int(
                self.trainer.estimated_stepping_batches * warmup_ratio,
            )
        optimizer_config = self.hparams['optimizer_config']
        loss_fn = self.loss_fn if isinstance(self.loss_fn, nn.Module) else None
        ignore_params = optimizer_config.pop('ignore_parameters', None)

        # instantiate model optimizer
        params_model = get_optimizer_params(self.model, loss_fn, ignore_params)
        if self.conditional:
            params_encoder = get_optimizer_params(self.encoder, loss_fn, ignore_params)
            params_model = [{'params': params_model}, {'params': params_encoder}]
        optimizer_model = instantiate_optimizer_from_config(optimizer_config, params_model)
        # instantiate model scheduler
        scheduler_model = get_scheduler(
            optimizer=optimizer_model,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=num_warmup_steps,
            **scheduler_config,
        )
        optimizers.append(optimizer_model)
        schedulers.append(
            {
                'scheduler': scheduler_model,
                'interval': 'step',
                'frequency': 1,
                'monitor': 'val/loss',
            },
        )

        # if self.conditional:
        #     params_encoder = get_optimizer_params(self.encoder, loss_fn, ignore_params)
        # optimizer_encoder = instantiate_optimizer_from_config(optimizer_config, params_encoder)
        # scheduler_encoder = get_scheduler(
        #     optimizer=optimizer_encoder,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        #     num_warmup_steps=num_warmup_steps,
        #     **scheduler_config,
        # )
        # optimizers.append(optimizer_encoder)
        # schedulers.append(
        #     {
        #         'scheduler': scheduler_encoder,
        #         'interval': 'step',
        #         'frequency': 1,
        #         'monitor': 'val/loss',
        #     },
        # )

        print_rank_zero(
            f'Setting up scheduler (estimated_stepping_batches: {self.trainer.estimated_stepping_batches})...',
        )
        return optimizers, schedulers

    def on_train_start(self) -> None:
        """Call when the training starts."""
        super().on_train_start()
        # Only initialize weights if we're not loading from a checkpoint
        if hasattr(self.model, 'init_weights') and (not hasattr(self.hparams, 'weights_only_path')):
            print_rank_zero('Initializing model weights...')
            self.model.init_weights()

    def forward(self, batch: dict[str, Any], batch_idx: int, **kwargs: dict[str, Any]) -> dict:
        """Forward pass.

        Args:
            batch: The batch.
            batch_idx: The batch index.
            **kwargs: Additional keyword arguments.

        """
        if 'attention_mask' not in batch and 'input_ids' in batch:
            pad_token_id = None
            if self.tokenizer is not None:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = getattr(self.model.config, 'pad_token_id', None)
            if pad_token_id is not None:
                batch = dict(batch)
                batch['attention_mask'] = (batch['input_ids'] != pad_token_id).long()

        if self.conditional:
            if self.custom_encoder:
                kwargs['latent'] = self.encoder(batch['input_ids']).y_pred
            else:
                if 'latent' not in batch:
                    raise ValueError(
                        f'Missing latent variable in batch for conditional model. Keys in batch: {batch.keys()}',
                    )
                kwargs['latent'] = self.encoder(batch['latent']).y_pred

        return self.loss_fn(
            lightning_module=self,
            model=self.model,
            batch=batch,
            **kwargs,
        )

    def sample(
        self,
        batch: dict[str, Any],
        max_batch_size: int | None = None,
        skip_special_tokens: bool = False,
        **kwargs: dict[str, Any],
    ) -> SampleOutput | None:
        if self.hparams.get('use_k_first_input_ids_as_prompt', None) is not None:
            k = self.hparams['use_k_first_input_ids_as_prompt']
            if 'input_ids' in batch and batch['input_ids'] is not None:
                prompt_ids = batch['input_ids'][:, :k]
                batch['prompt_ids'] = prompt_ids
                print_rank_zero()
                print_rank_zero(f'Using first {k} tokens as prompt for conditional generation.')
            else:
                print_rank_zero('Warning: input_ids not found in batch, cannot use first k tokens as prompt.')

        if 'prompt_ids' in batch:
            prompt_ids = batch['prompt_ids']
            if max_batch_size is not None:
                prompt_ids = prompt_ids[:max_batch_size]
            prompt_len = prompt_ids.size(1)
            if self.tokenizer is not None:
                prompts = self.tokenizer.batch_decode(
                    prompt_ids,
                    skip_special_tokens=skip_special_tokens,
                )
            else:
                prompts = [''] * prompt_ids.size(0)  # Fallback if no tokenizer
        else:
            prompt_ids = None
            prompt_len = 0
            num_samples = kwargs.get('num_samples', max_batch_size or 1)
            kwargs['num_samples'] = num_samples
            prompts = None

        # set pad_token_id in self.sampler
        self.sampler.pad_token_id = (
            self.tokenizer.pad_token_id
            if (self.tokenizer is not None) and (hasattr(self.tokenizer, 'pad_token_id'))
            else None
        )

        # kwargs['all_input_ids'] = batch.get('input_ids', None)

        with torch.no_grad():  # self.ema_scope()
            if self.conditional:
                if self.custom_encoder:
                    kwargs['latent'] = self.encoder(batch['input_ids']).y_pred
                else:
                    if 'latent' not in batch:
                        raise ValueError('Missing latent variable in batch for conditional model')
                    kwargs['latent'] = self.encoder(batch['latent']).y_pred

            outputs = self.sampler(
                lightning_module=self,
                model=self.model,
                input_ids=prompt_ids,
                **kwargs,
            )

        output_ids = outputs  # outputs[:, prompt_len:]
        if self.tokenizer is not None:
            completions = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            completions = [''] * output_ids.size(0)  # Fallback if no tokenizer

        return SampleOutput(
            output_ids=output_ids,
            prompts=prompts,
            completions=completions,
        )
