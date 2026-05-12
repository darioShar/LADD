from typing import Any

import torch
from torch import nn
from transformers.optimization import get_scheduler

from dlm.modules.ema import EMAModel
from dlm.utils import get_optimizer_params, instantiate_from_config, instantiate_model_from_config, print_rank_zero

from ..metrics.statistics import MaxMetric, MeanMetric, MinMetric, StandardDeviationMetric
from ..metrics.token_metrics import (
    CodebookUtilizationMetric,
    EntropyMetric,
    MeanCodebookPairwiseDistanceMetric,
    MinCodebookPairwiseDistanceMetric,
    SelfTransitionRateMeanMetric,
    SelfTransitionRateVarMetric,
    TokenDistributionKLMetric,
    UniqueCodeRatioMeanMetric,
    UniqueCodeRatioVarMetric,
)
from ..metrics.wasserstein import SlicedWassersteinMetric
from .baselm import BaseLM
from .output import LatentDDMSampleOutput, SampleOutput


class LatentDDM(BaseLM):
    """Latent Discrete Diffusion Model (LatentDDM).

    This model consists of two main components:
    1. An encoder that maps an input sequence x_0 to a latent representation y_0.
    2. A denoiser (the main diffusion model) that learns to denoise a noisy
       version of x_0, conditioned on a noisy version of y_0.
    """

    def __init__(
        self,
        encoder_config: dict,
        joint_denoiser_config: dict,
        latent_denoiser_config: dict | None,
        sampler_config: dict | None,
        loss_config: dict,
        optimizer_config: dict,
        scheduler_config: dict | None = None,
        gradient_checkpointing: bool = False,
        peft_config: dict | None = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        torch_compile: str | None = None,
        not_hf_pretrained: bool = False,
        freeze_encoder: bool = False,  # Whether to freeze the encoder during training
        save_encoder: bool = True,  # Whether to save the encoder in checkpoints
        training_stage: int | None = None,
        core_model: str = 'JOINT',  # JOINT, SEQ
        use_k_first_input_ids_as_prompt: int | None = None,
        vector_quantizer_config: dict | None = None,
        discrete_metrics: bool = False,
        codebook_size: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the LatentDDM.

        Args:
            encoder_config: The encoder configuration.
            joint_denoiser_config: The joint denoiser configuration.
            sampler_config: The sampler configuration.
            loss_config: The loss configuration.
            optimizer_config: The optimizer configuration.
            scheduler_config: The scheduler configuration.
            gradient_checkpointing: Whether to use gradient checkpointing.
            peft_config: The PEFT configuration.
            use_ema: Whether to use EMA.
            ema_decay: The EMA decay.
            torch_compile: Whether to use torch.compile.
            not_hf_pretrained: Whether to use non-HF pretrained models.
            core_model: The core model.
            freeze_encoder: Whether to freeze the encoder during training.
            save_encoder: Whether to save the encoder in checkpoints.
            latent_denoiser_config: The latent denoiser configuration.

        """
        super().__init__(core_model=core_model, **kwargs)
        # The denoiser_config will be treated as the main model_config

        # save local args to self.hparams
        self.save_hyperparameters()
        # assume this will be passed from datamodule
        self.tokenizer = None

        if sampler_config is None:
            msg = 'sampler_config is required'
            raise ValueError(msg)
        self.sampler = instantiate_from_config(sampler_config)

        # Initialize latent-specific evaluation metrics as ModuleDicts
        self.eval_y_metrics = {
            'generated': {
                'y_min': MinMetric(),
                'y_max': MaxMetric(),
                'y_mean': MeanMetric(),
                'y_std': StandardDeviationMetric(),
                # 'y_entropy': EntropyMetric(),
                # 'y_unique_code_ratio_mean': UniqueCodeRatioMeanMetric(),
                # 'y_unique_code_ratio_var': UniqueCodeRatioVarMetric(),
                # 'y_self_transition_rate_mean': SelfTransitionRateMeanMetric(),
                # 'y_self_transition_rate_var': SelfTransitionRateVarMetric(),
            },
            'real': {
                'y_min': MinMetric(),
                'y_max': MaxMetric(),
                'y_mean': MeanMetric(),
                'y_std': StandardDeviationMetric(),
                # 'y_entropy': EntropyMetric(),
                # 'y_unique_code_ratio_mean': UniqueCodeRatioMeanMetric(),
                # 'y_unique_code_ratio_var': UniqueCodeRatioVarMetric(),
                # 'y_self_transition_rate_mean': SelfTransitionRateMeanMetric(),
                # 'y_self_transition_rate_var': SelfTransitionRateVarMetric(),
            },
            'distance_discrete': {
                'y_token_distribution_kl': TokenDistributionKLMetric(self.hparams.get('codebook_size', None)),
            },
            'distance_continuous': {},  # start empty
        }

        if self.hparams.get('discrete_metrics', False):
            codebook_size = self.hparams.get('codebook_size', None)

            self.eval_y_metrics['generated']['y_entropy'] = EntropyMetric()
            self.eval_y_metrics['generated']['y_unique_code_ratio_mean'] = UniqueCodeRatioMeanMetric()
            self.eval_y_metrics['generated']['y_unique_code_ratio_var'] = UniqueCodeRatioVarMetric()
            self.eval_y_metrics['generated']['y_self_transition_rate_mean'] = (
                SelfTransitionRateMeanMetric()
            )
            self.eval_y_metrics['generated']['y_self_transition_rate_var'] = (
                SelfTransitionRateVarMetric()
            )
            if codebook_size is not None:
                self.eval_y_metrics['generated']['y_codebook_utilization'] = (
                    CodebookUtilizationMetric(codebook_size)
                )

            self.eval_y_metrics['real']['y_entropy'] = EntropyMetric()
            self.eval_y_metrics['real']['y_unique_code_ratio_mean'] = UniqueCodeRatioMeanMetric()
            self.eval_y_metrics['real']['y_unique_code_ratio_var'] = UniqueCodeRatioVarMetric()
            self.eval_y_metrics['real']['y_self_transition_rate_mean'] = SelfTransitionRateMeanMetric()
            self.eval_y_metrics['real']['y_self_transition_rate_var'] = SelfTransitionRateVarMetric()
            if codebook_size is not None:
                self.eval_y_metrics['real']['y_codebook_utilization'] = (
                    CodebookUtilizationMetric(codebook_size)
                )

            # Codebook embedding space metrics (require access to vector_quantizer)
            self.eval_y_metrics['codebook'] = {
                'y_min_pairwise_dist': MinCodebookPairwiseDistanceMetric(),
                'y_mean_pairwise_dist': MeanCodebookPairwiseDistanceMetric(),
            }

        if self.hparams.get('enable_sliced_wasserstein', False) and not self.hparams.get('discrete_metrics', False):
            self.eval_y_metrics['distance_continuous']['y_sliced_wasserstein'] = SlicedWassersteinMetric()

        # create module dict
        for k, v in self.eval_y_metrics.items():
            self.eval_y_metrics[k] = nn.ModuleDict(v)
        self.eval_y_metrics = nn.ModuleDict(self.eval_y_metrics)

    def _get_metric_attribute_name(self, metric):
        """Resolve dynamic metric paths for latent-specific nested metrics."""
        parent_name = super()._get_metric_attribute_name(metric)
        if parent_name is not None:
            return parent_name

        for group_key, group_container in self.eval_y_metrics.items():
            if not isinstance(group_container, nn.ModuleDict):
                continue
            for metric_key, candidate in group_container.items():
                if candidate is metric:
                    return f'eval_y_metrics.{group_key}.{metric_key}'
        return None

    def state_dict(self, *args, **kwargs):
        """Override state_dict to optionally exclude encoder from saving.

        This is useful when the encoder is frozen and loaded from a pretrained checkpoint,
        as it can significantly reduce checkpoint size (e.g., when using large models like Qwen).
        """
        state = super().state_dict(*args, **kwargs)

        # Remove encoder parameters if save_encoder is False
        if not self.hparams.get('save_encoder', True):
            keys_to_remove = [k for k in state if k.startswith('encoder.')]
            for key in keys_to_remove:
                del state[key]
            if keys_to_remove:
                print_rank_zero(f'Excluding {len(keys_to_remove)} encoder parameters from checkpoint.')

        return state

    def configure_model(self) -> None:
        """Configure the model.

        This instantiates the denoiser, encoder, denoiser_, and vector quantizer.
        """

        # instantiate the vector quantizer
        if self.hparams['vector_quantizer_config'] is not None:
            print_rank_zero('Instantiating vector quantizer...')
            self.vector_quantizer: nn.Module = instantiate_model_from_config(
                self.hparams['vector_quantizer_config'],
                not_hf_pretrained=self.hparams.get('not_hf_pretrained', False),
            ).train()
        else:
            self.vector_quantizer = None

        # Instantiate the denoiser
        # check if trainer attribute exists
        print_rank_zero('Instantiating joint denoiser...')
        if self.hparams['joint_denoiser_config'] is None:
            print_rank_zero('joint denoiser config is None')
            self.joint_denoiser = None
        else:
            self.joint_denoiser: nn.Module = instantiate_model_from_config(
                self.hparams['joint_denoiser_config'],
                not_hf_pretrained=self.hparams.get('not_hf_pretrained', False),
            ).train()

        # Instantiate the encoder
        print_rank_zero('Instantiating encoder...')
        encoder_config = self.hparams['encoder_config']
        # qwen_encoder_config in coladd_base.yaml includes pretrained_model_name_or_path. instantiate_model_from_config treats that as “HF pretrained” and calls QwenEmbeddingModel.from_pretrained(...), which will likely fail because QwenEmbeddingModel is a wrapper, not a published HF checkpoint. So a conditional keeps Qwen working.
        if encoder_config is None:
            print_rank_zero('encoder config is None')
            self.encoder = None
        elif isinstance(encoder_config, dict) and 'params' in encoder_config and 'config' in encoder_config['params']:
            self.encoder = instantiate_model_from_config(
                encoder_config,
                not_hf_pretrained=self.hparams.get('not_hf_pretrained', False),
            )
        else:
            self.encoder = instantiate_from_config(encoder_config)

        # Freeze encoder if requested
        if self.hparams.get('freeze_encoder', False):
            for param in self.encoder.parameters():
                param.requires_grad = False
            print_rank_zero('Encoder parameters frozen.')
            self.encoder.eval()
        else:
            self.encoder.train()

        # Instantiate the latent denoiser if core model is not JOINT
        if (self.hparams['core_model'] != 'JOINT') and (self.hparams['latent_denoiser_config'] is not None):
            print_rank_zero(f'Core model = {self.hparams["core_model"]} -> Instantiating latent denoiser...')
            self.latent_denoiser: nn.Module = instantiate_model_from_config(
                self.hparams['latent_denoiser_config'],
                not_hf_pretrained=self.hparams.get('not_hf_pretrained', False),
            )
            if self.hparams['training_stage'] == 1:
                for param in self.latent_denoiser.parameters():
                    param.requires_grad = False
                print_rank_zero('Latent denoiser parameters frozen during training stage 1.')
                self.latent_denoiser.eval()
            else:
                self.latent_denoiser.train()
        else:
            self.latent_denoiser = None

        instantiated_models = {
            'joint_denoiser': self.joint_denoiser,
            'encoder': self.encoder,
            'latent_denoiser': self.latent_denoiser,
        }
        for key, model in instantiated_models.items():
            if model is not None:
                # Skip EMA tracking for frozen encoder to avoid redundant updates.
                if not (key == 'encoder' and self.hparams.get('freeze_encoder', False)):
                    # Store parameters as lists (not iterators) for EMA
                    self.models_parameters_for_ema_update[key] = list(model.parameters())
                # Handle gradient checkpointing for both models
                if self.hparams.get('gradient_checkpointing', False):
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={'use_reentrant': False},
                    )
                    print_rank_zero(f'{key} gradient checkpointing enabled.')
                # Handle torch.compile for our models
                if self.hparams.get('torch_compile', None) is not None:
                    print_rank_zero(f'Compiling {key}... (takes a ~minute)')
                    # Avoid fullgraph strict mode to allow graph breaks around FlashAttention kernels.
                    model.compile(mode=self.hparams['torch_compile'])

        # Setup EMA models
        if self.hparams.get('use_ema', False):
            for model_name, params in self.models_parameters_for_ema_update.items():
                self.ema_models[model_name] = EMAModel(
                    params,
                    decay=self.hparams.get('ema_decay', 0.9999),
                )
                print_rank_zero(f'EMA enabled for {model_name}.')

        if not hasattr(self, 'loss_fn'):
            self.loss_fn = instantiate_from_config(self.hparams['loss_config'])

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log grad norms for y-channel params in the joint denoiser."""
        super().on_before_optimizer_step(optimizer)
        if self.joint_denoiser is None:
            return

        y_param_groups = {
            'qkv_y': [],
            'mlp_y': [],
        }
        for name, param in self.joint_denoiser.named_parameters():
            if not param.requires_grad:
                continue
            if 'qkv_y' in name:
                y_param_groups['qkv_y'].append(param)
            elif 'mlp_y' in name:
                y_param_groups['mlp_y'].append(param)

        logs: dict[str, torch.Tensor] = {}
        for key, params in y_param_groups.items():
            if not params:
                continue
            logs[f'train/grad_norm_y/{key}'] = self._grad_l2_norm(params).to(self.device)

        if logs:
            self.log_dict(
                logs,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]] | list[torch.optim.Optimizer]:
        """Configure optimizers and learning rate schedulers.

        Working on both the encoder and the denoiser.
        """
        optimizer_config = self.hparams.get('optimizer_config', {})
        # Make a copy to avoid modifying the original config
        optimizer_config = optimizer_config.copy()
        ignore_params = optimizer_config.pop('ignore_parameters', None)

        instantiated_models = {
            'joint_denoiser': self.joint_denoiser,
            'encoder': self.encoder,
            'latent_denoiser': self.latent_denoiser,
            'vector_quantizer': self.vector_quantizer,
        }

        # Get parameters from available models
        params = []
        for key, model in instantiated_models.items():
            # Only include encoder parameters in optimizer if not frozen
            if (key == 'encoder') and self.hparams.get('freeze_encoder', False):
                continue
            # Only include latent denoiser if not in training stage 1:
            if (key == 'latent_denoiser') and (self.hparams.get('training_stage') == 1):
                continue
            # Include non-null models
            if model is not None:
                params += get_optimizer_params(model, None, ignore_params)

        # Instantiate optimizer
        optimizer = instantiate_from_config(optimizer_config, params=params)

        # Instantiate scheduler
        scheduler_config = self.hparams.get('scheduler_config')
        if scheduler_config:
            scheduler_config = scheduler_config.copy()
            num_warmup_steps = scheduler_config.pop('num_warmup_steps', None)
            warmup_ratio = scheduler_config.pop('warmup_ratio', None)
            if num_warmup_steps is None and warmup_ratio is not None:
                num_warmup_steps = int(
                    self.trainer.estimated_stepping_batches * warmup_ratio,
                )

            scheduler = get_scheduler(
                optimizer=optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=num_warmup_steps,
                **scheduler_config,
            )
            print_rank_zero(
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

        return [optimizer]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Calls parent to load checkpoint with multiple EMA models, and checks optimizer compatibility.

        Args:
            checkpoint: The checkpoint dictionary.

        """
        super().on_load_checkpoint(checkpoint)
        # Handle optimizer state dict compatibility when changing training stages
        self._check_optimizer_compatibility(checkpoint)

    def _check_optimizer_compatibility(self, checkpoint: dict) -> None:
        """Check if the optimizer state dict is compatible with current parameter groups.

        This handles cases where training stages change the set of trainable parameters,
        which would cause optimizer state dict mismatches when resuming.

        Args:
            checkpoint: The checkpoint dictionary.

        """
        if 'optimizer_states' not in checkpoint:
            return

        # Get current parameter groups that would be used by configure_optimizers
        current_params = self._get_current_optimizer_params()

        # Check if we have optimizer state to load
        optimizer_states = checkpoint['optimizer_states']
        if not optimizer_states or len(optimizer_states) == 0:
            return

        # Get the first optimizer state (assuming single optimizer)
        optimizer_state = optimizer_states[0]
        if 'param_groups' not in optimizer_state:
            return

        # Count parameters in saved state
        saved_param_count = sum(len(group['params']) for group in optimizer_state['param_groups'])
        current_param_count = len(current_params)

        if saved_param_count != current_param_count:
            print_rank_zero('WARNING: Optimizer parameter count mismatch detected!')
            print_rank_zero(f'  Saved optimizer had {saved_param_count} parameters')
            print_rank_zero(f'  Current configuration has {current_param_count} parameters')
            print_rank_zero('  This often occurs when changing training_stage or freeze_encoder settings.')
            print_rank_zero('  Skipping optimizer state loading - optimizer will restart from scratch.')

            # Clear optimizer states to prevent loading incompatible state
            checkpoint['optimizer_states'] = []

            # Also clear LR scheduler states since they depend on optimizer state
            if 'lr_schedulers' in checkpoint:
                checkpoint['lr_schedulers'] = []
                print_rank_zero('  Also cleared LR scheduler state.')

    def _get_current_optimizer_params(self) -> list:
        """Get the current parameters that would be included in the optimizer.

        This replicates the logic from configure_optimizers to determine
        which parameters are currently trainable.

        Returns:
            List of parameters that would be included in the optimizer.

        """
        optimizer_config = self.hparams.get('optimizer_config', {})
        ignore_params = optimizer_config.get('ignore_parameters', None)

        params = []

        # Get parameters from vector quantizer
        if self.vector_quantizer is not None:
            params += get_optimizer_params(self.vector_quantizer, None, ignore_params)

        # Get parameters from joint denoiser
        if self.joint_denoiser is not None:
            params += get_optimizer_params(self.joint_denoiser, None, ignore_params)

        # Only include encoder parameters if not frozen
        if not self.hparams.get('freeze_encoder', False):
            params += get_optimizer_params(self.encoder, None, ignore_params)

        # Include latent denoiser parameters if available and not in stage 1
        if self.latent_denoiser is not None and self.hparams.get('training_stage') != 1:
            params += get_optimizer_params(self.latent_denoiser, None, ignore_params)

        return params

    def forward(self, batch: dict[str, Any], batch_idx: int, **kwargs: dict[str, Any]) -> dict:
        """Compute the loss.

        It passes both the encoder and denoiser to the loss function.
        """

        return self.loss_fn(
            lightning_module=self,
            encoder=self.encoder,
            joint_denoiser=self.joint_denoiser,
            latent_denoiser=self.latent_denoiser,
            vector_quantizer=self.vector_quantizer,
            batch=batch,
            batch_idx=batch_idx,
            **kwargs,
        )

    def sample(
        self,
        batch: dict[str, Any],
        max_batch_size: int | None = None,
        skip_special_tokens: bool = False,
        **kwargs: dict[str, Any],
    ) -> SampleOutput | None:
        """Generate samples from the LatentDDM.

        This requires a custom sampler that can handle the joint denoising of
        the token sequence `x` and the latent vector `y`.

        Args:
            batch: The batch.
            max_batch_size: The maximum batch size.
            skip_special_tokens: Whether to skip special tokens.
            **kwargs: Additional keyword arguments.

        """
        if self.hparams.get('use_k_first_input_ids_as_prompt', None) is not None:
            k = self.hparams['use_k_first_input_ids_as_prompt']
            if 'input_ids' in batch and batch['input_ids'] is not None:
                prompt_ids = batch['input_ids'][:, :k]
                batch['prompt_ids'] = prompt_ids
                print_rank_zero()
                print_rank_zero(f'Using first {k} tokens as prompt for conditional generation.')
            else:
                print_rank_zero('Warning: input_ids not found in batch, cannot use first k tokens as prompt.')

        # The sampler will need both encoder and denoiser.
        # The encoder might be used to get an initial y_0 from a prompt.
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
            kwargs['num_samples'] = num_samples  # type: ignore[assignment]
            prompts = None

        # set pad_token_id in self.sampler
        self.sampler.pad_token_id = (
            self.tokenizer.pad_token_id
            if (self.tokenizer is not None) and (hasattr(self.tokenizer, 'pad_token_id'))
            else None
        )

        # kwargs['all_input_ids'] = batch.get('input_ids', None)

        # identify how many tokens are being generated
        # if prompts is not None:
        #     token_proportion_to_generate = self.sampler.max_new_tokens / (prompt_len + self.sampler.max_new_tokens)
        #     kwargs['token_proportion_to_generate'] = token_proportion_to_generate # type: ignore[assignment]
        #     print_rank_zero(f'Generating {self.sampler.max_new_tokens} new tokens after prompt of length {prompt_len} ({token_proportion_to_generate:.2%} of total).')

        with torch.no_grad():  # self.ema_scope()
            outputs, y_outputs = self.sampler(
                lightning_module=self,
                joint_denoiser=self.joint_denoiser,
                encoder=self.encoder,
                latent_denoiser=self.latent_denoiser,
                vector_quantizer=self.vector_quantizer,
                input_ids=prompt_ids,
                batch=batch,
                **kwargs,
            )

        # Handle case where sampler failed
        if outputs is None or y_outputs is None:
            print_rank_zero('WARNING: Sampler returned None, cannot generate samples')
            return None

        output_ids = outputs  # outputs[:, prompt_len:]
        if self.tokenizer is not None:
            completions = self.tokenizer.batch_decode(
                # output_ids[:, prompt_len:],
                output_ids,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            completions = [''] * output_ids.size(0)  # Fallback if no tokenizer

        return LatentDDMSampleOutput(
            output_ids=output_ids,
            prompts=prompts,
            completions=completions,
            y_outputs=y_outputs,
        )

    def _compute_generative_metrics(
        self,
        loss_dict: dict[str, Any],
        batch: dict[str, Any],
        generated_samples: LatentDDMSampleOutput,
        step: str,
        suffix: str = '',
    ) -> dict[str, torch.Tensor]:
        """Compute latent-specific generative metrics.

        This method adds metrics specific to latent diffusion models,
        including statistics on y_outputs and sliced Wasserstein between
        real and generated latents.
        """
        super()._compute_generative_metrics(loss_dict, batch, generated_samples, step, suffix)
        # Skip during sanity check
        if self.trainer.sanity_checking:
            return loss_dict

        # Process y_outputs if available
        if hasattr(generated_samples, 'y_outputs') and generated_samples.y_outputs is not None:
            y_outputs = generated_samples.y_outputs

            # get real y_0
            y_0_type = None
            is_discrete = y_outputs.dtype in (torch.int16, torch.int32, torch.int64, torch.long)
            if is_discrete:
                # if discrete latents
                x_0 = batch['input_ids']
                _, y_0, _, _ = self.sampler.encode_x_0_to_y_0(
                    x_0=x_0,
                    encoder=self.encoder,
                    vector_quantizer=self.vector_quantizer,
                )
                y_0_type = 'discrete'
            else:
                # if continuous latents
                encoder_output = self.encoder(batch['input_ids'])
                y_0 = encoder_output.y_pred
                y_0_type = 'continuous'

            # Update y_outputs statistics
            for key, metric in self.eval_y_metrics['generated'].items():
                if metric is not None:
                    metric.update(y_outputs)
                    loss_dict[f'eval/{key}'] = metric

            for key, metric in self.eval_y_metrics['real'].items():
                if metric is not None:
                    metric.update(y_0)
                    loss_dict[f'eval/real_{key}'] = metric

            for key, metric in self.eval_y_metrics[f'distance_{y_0_type}'].items():
                if metric is not None:
                    metric.update(y_0, y_outputs)
                    loss_dict[f'eval/distance_{key}'] = metric


            # Update codebook embedding space metrics if available
            if 'codebook' in self.eval_y_metrics and self.vector_quantizer is not None:
                codebook_embeddings = self.vector_quantizer.codebook()
                for key, metric in self.eval_y_metrics['codebook'].items():
                    if metric is not None:
                        metric.update(codebook_embeddings)
                        loss_dict[f'eval/codebook_{key}'] = metric

        return loss_dict

    def on_predict_epoch_end(self) -> None:
        """Compute and log latent-specific evaluation metrics at the end of prediction."""
        # First call parent method to handle base metrics
        super().on_predict_epoch_end()

        latent_metrics_to_log: dict[str, torch.Tensor] = {}

        # Generated-y metrics: eval/{key}
        for key, metric in self.eval_y_metrics.get('generated', {}).items():
            if metric is not None:
                latent_metrics_to_log[f'eval/{key}'] = metric.compute()
                metric.reset()

        # Real-y metrics: eval/real_{key}
        for key, metric in self.eval_y_metrics.get('real', {}).items():
            if metric is not None:
                latent_metrics_to_log[f'eval/real_{key}'] = metric.compute()
                metric.reset()

        # Distance metrics (discrete / continuous): eval/distance_{key}
        for dist_type in ('distance_discrete', 'distance_continuous'):
            for key, metric in self.eval_y_metrics.get(dist_type, {}).items():
                if metric is not None:
                    latent_metrics_to_log[f'eval/distance_{key}'] = metric.compute()
                    metric.reset()

        # Codebook embedding space metrics: eval/codebook_{key}
        for key, metric in self.eval_y_metrics.get('codebook', {}).items():
            if metric is not None:
                latent_metrics_to_log[f'eval/codebook_{key}'] = metric.compute()
                metric.reset()

        if latent_metrics_to_log:
            print_rank_zero(
                f'Computed latent evaluation metrics for prediction: {latent_metrics_to_log}',
            )
            # self.log_dict(latent_metrics_to_log, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def _log_samples(self) -> None:
        """Log samples to the logger (rank 0 only), including discrete latent tokens.

        Extends the parent method to also log latent token sequences when they are discrete.
        """
        if not self.generated_samples_for_logging or not hasattr(self, 'tokenizer') or self.tokenizer is None:
            print_rank_zero('Skipping _log_samples - no samples or tokenizer')
            return

        samples = self.generated_samples_for_logging[0]
        real_samples = self.real_samples_for_logging[0]

        real_decoded, decoded_prompts, decoded_completions = self._prepare_samples_for_logging(
            samples,
            real_samples,
        )

        # Check if we have discrete latent outputs
        latent_tokens = None
        real_latent_tokens = None
        if hasattr(samples, 'y_outputs') and samples.y_outputs is not None:
            y_outputs = samples.y_outputs
            is_discrete = y_outputs.dtype in (torch.int16, torch.int32, torch.int64, torch.long)
            if is_discrete:
                # Convert to list of strings for logging
                latent_tokens = [str(y.tolist()) for y in y_outputs]
                # Get real latent tokens
                if self.encoder is not None and hasattr(self.sampler, 'encode_x_0_to_y_0'):
                    # Use autocast to match trainer precision (needed for flash attention)
                    device_type = real_samples.device.type
                    autocast_dtype = torch.bfloat16 if device_type == 'cuda' else None
                    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                        _, real_y_0, _, _ = self.sampler.encode_x_0_to_y_0(
                            x_0=real_samples,
                            encoder=self.encoder,
                            vector_quantizer=self.vector_quantizer,
                        )
                        real_latent_tokens = [str(y.tolist()) for y in real_y_0]

        if latent_tokens is not None:
            # Include latent tokens in the log
            if decoded_prompts is None:
                log_data = [
                    [real, completion, real_lat, gen_lat]
                    for real, completion, real_lat, gen_lat in zip(
                        real_decoded,
                        decoded_completions,
                        real_latent_tokens or [''] * len(real_decoded),
                        latent_tokens,
                        strict=False,
                    )
                ]
                self._log_text_to_logger(
                    columns=['real', 'completion', 'real_latents', 'gen_latents'],
                    data=log_data,
                )
            else:
                log_data = [
                    [real, prompt, completion, real_lat, gen_lat]
                    for real, prompt, completion, real_lat, gen_lat in zip(
                        real_decoded,
                        decoded_prompts,
                        decoded_completions,
                        real_latent_tokens or [''] * len(real_decoded),
                        latent_tokens,
                        strict=False,
                    )
                ]
                self._log_text_to_logger(
                    columns=['real', 'prompt', 'completion', 'real_latents', 'gen_latents'],
                    data=log_data,
                )
        else:
            # Fallback to parent behavior (no latent tokens)
            super()._log_samples()
