import traceback
from contextlib import contextmanager
from typing import Any

import lightning as L
import torch
import torchmetrics

from ..metrics import BitsPerCharacterMetric, PerplexityMetric
from ..metrics.generative_perplexity import (
    GenerativePerplexityMetric,
    GradientMomentMetric,
    load_teacher_model,
)
from ..metrics.sensitivity import SensitivityMetric
from ..metrics.token_metrics import EntropyMetric, TokenDistributionKLMetric
from ..metrics.wasserstein import SlicedWassersteinMetric
from ..modules.ema import EMAModel
from ..utils import print_rank_zero
from .output import SampleOutput
from .restore_state_dict import (
    _get_all_param_groups,
    _ReentryLRController,
    load_weights_agnostic,
    restore_optim_from_checkpoint,
)


class BaseLM(L.LightningModule):
    """Base class for all models."""

    def __init__(
        self,
        eval_and_log_first_n_batches=0,
        enable_entropy=False,
        enable_generative_perplexity=False,
        teacher_model_name_or_path='gpt2-large',
        enable_gradient_moment_metric=False,
        gradient_moment_include_parameter_regex=None,
        enable_sliced_wasserstein=False,
        enable_token_distribution_kl=False,
        enable_sensitivity_metric=False,
        enable_grad_norms_logging=False,
        sensitivity_positions_per_sample=5,
        sensitivity_perturbation_trials=10,
        sensitivity_temperature=1.0,
        core_model='base',
        strict_loading=True,
        sequence_length=1024,
        limit_eval_batch_size=None,
        on_validation_only_eval=False,
        weights_only_path=None,
        vocab_size=None,
    ) -> None:
        """Initialize the BaseLM."""
        super().__init__()
        self.eval_and_log_first_n_batches = eval_and_log_first_n_batches
        self.enable_entropy = enable_entropy
        self.enable_generative_perplexity = enable_generative_perplexity
        self.teacher_model_name_or_path = teacher_model_name_or_path
        self.enable_gradient_moment_metric = enable_gradient_moment_metric
        self.gradient_moment_include_parameter_regex = gradient_moment_include_parameter_regex
        self.enable_sliced_wasserstein = enable_sliced_wasserstein
        self.enable_token_distribution_kl = enable_token_distribution_kl
        self.enable_sensitivity_metric = enable_sensitivity_metric
        self.enable_grad_norms_logging = enable_grad_norms_logging
        self.sensitivity_positions_per_sample = int(sensitivity_positions_per_sample)
        self.sensitivity_perturbation_trials = int(sensitivity_perturbation_trials)
        self.sensitivity_temperature = float(sensitivity_temperature)
        self.core_model = core_model
        self.strict_loading = strict_loading  # Whether to strictly enforce model loading, see https://github.com/Lightning-AI/pytorch-lightning/pull/19404
        self.sequence_length = sequence_length
        self.limit_eval_batch_size = limit_eval_batch_size
        self.on_validation_only_eval = on_validation_only_eval
        self.weights_only_path = weights_only_path
        self.vocab_size = vocab_size

        # Subclasses should populate this with models and models for EMA
        self.models_parameters_for_ema_update = {}
        self.ema_models: dict[str, EMAModel] = {}

        # Track validation state for EMA optimization
        self._ema_activation_nested_level = 0

        # Initialize metrics for proper distributed aggregation
        self.train_perplexity = PerplexityMetric()
        self.val_perplexity = PerplexityMetric()
        self.train_bpc = BitsPerCharacterMetric()
        self.val_bpc = BitsPerCharacterMetric()
        self.val_num_generated_samples = torchmetrics.SumMetric()
        self.val_sensitivity = SensitivityMetric() if self.enable_sensitivity_metric else torch.nn.Identity()

        # Evaluation metrics - use ModuleDict to support multiple inference step configurations.
        # Note: ModuleDict keys cannot be empty strings or start with numbers.
        self.inference_step_suffixes = {
            'default': '',
        }  # Maps metric key -> logging suffix (updated at validation/predict start).
        self._inference_step_list: list[int] = []
        self._sampling_step_configs: list[dict[str, Any]] = []

        self.eval_entropy = torch.nn.ModuleDict(
            {
                key: EntropyMetric() if self.enable_entropy else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.real_entropy = torch.nn.ModuleDict(
            {
                key: EntropyMetric() if self.enable_entropy else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.eval_generative_perplexity = torch.nn.ModuleDict(
            {
                key: GenerativePerplexityMetric() if self.enable_generative_perplexity else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.real_generative_perplexity = torch.nn.ModuleDict(
            {
                key: GenerativePerplexityMetric() if self.enable_generative_perplexity else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.eval_gradient_moment = torch.nn.ModuleDict(
            {
                key: (
                    GradientMomentMetric(
                        include_parameter_regex=self.gradient_moment_include_parameter_regex,
                    )
                    if self.enable_gradient_moment_metric
                    else torch.nn.Identity()
                )
                for key in self.inference_step_suffixes
            },
        )
        self.eval_sliced_wasserstein = torch.nn.ModuleDict(
            {
                key: SlicedWassersteinMetric() if self.enable_sliced_wasserstein else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.eval_token_distribution_kl = torch.nn.ModuleDict(
            {
                key: (
                    TokenDistributionKLMetric(vocab_size=vocab_size)
                    if self.enable_token_distribution_kl
                    else torch.nn.Identity()
                )
                for key in self.inference_step_suffixes
            },
        )
        self._gradient_moment_prev_batches: dict[str, dict[str, torch.Tensor | None] | None] = dict.fromkeys(self.inference_step_suffixes)
        self._teacher_model: torch.nn.Module | None = None
        self._teacher_tokenizer = None

        # Validation sampling control
        self.real_samples_for_logging = []
        self.generated_samples_for_logging = []
        self._printed_dataset_info = False
        self._printed_first_train_batch = False
        self._printed_first_val_samples = False

        # assume this will be passed from datamodule
        self.tokenizer = None

        # Store the current learning rate before checkpoint loading can overwrite it
        self._current_config_lr = None

        self.grad_norms: dict[str, float] = {}
        self._max_grad_norm_logs = 32  # avoid spamming the logger if you have many submodules

    def _iter_trainable_children(self):
        """Yield (name, module, param_list) for each top-level nn.Module child that has trainable params."""
        for name, module in self.named_children():
            # Skip metrics containers
            if isinstance(module, torchmetrics.Metric):
                continue
            params = [p for p in module.parameters(recurse=True) if p.requires_grad]
            if len(params) > 0:
                yield name, module, params

    def _get_metric_attribute_name(self, metric: torchmetrics.Metric) -> str | None:
        """Return the LightningModule attribute path for a metric instance.

        Lightning requires this when logging Metric objects that live inside ModuleDicts whose
        contents can change during runtime (e.g. inference-step-specific metrics).
        """
        top_level_metric_attrs = (
            'train_perplexity',
            'val_perplexity',
            'train_bpc',
            'val_bpc',
            'val_num_generated_samples',
            'val_sensitivity',
        )
        for attr in top_level_metric_attrs:
            if getattr(self, attr, None) is metric:
                return attr

        moduledict_metric_attrs = (
            'eval_entropy',
            'real_entropy',
            'eval_generative_perplexity',
            'real_generative_perplexity',
            'eval_gradient_moment',
            'eval_sliced_wasserstein',
            'eval_token_distribution_kl',
        )
        for container_attr in moduledict_metric_attrs:
            container = getattr(self, container_attr, None)
            if not isinstance(container, torch.nn.ModuleDict):
                continue
            for key, candidate in container.items():
                if candidate is metric:
                    return f'{container_attr}.{key}'
        return None

    def _log_dict_with_metric_attributes(
        self,
        values: dict[str, Any],
        *,
        batch_size: int,
        prog_bar: bool,
        logger: bool,
        sync_dist: bool,
        on_epoch: bool,
        on_step: bool = False,
    ) -> None:
        """Log a dictionary while providing explicit metric_attribute for Metric values."""
        for name, value in values.items():
            if isinstance(value, torchmetrics.Metric):
                metric_attribute = self._get_metric_attribute_name(value)
                log_kwargs = {
                    'batch_size': batch_size,
                    'prog_bar': prog_bar,
                    'logger': logger,
                    'sync_dist': sync_dist,
                    'on_epoch': on_epoch,
                    'on_step': on_step,
                }
                if metric_attribute is not None:
                    self.log(name, value, metric_attribute=metric_attribute, **log_kwargs)
                else:
                    # Fallback: avoid crashing if Lightning cannot resolve a dynamic metric attribute.
                    self.log(name, value.compute(), **log_kwargs)
            else:
                self.log(
                    name,
                    value,
                    batch_size=batch_size,
                    prog_bar=prog_bar,
                    logger=logger,
                    sync_dist=sync_dist,
                    on_epoch=on_epoch,
                    on_step=on_step,
                )

    @staticmethod
    def _grad_l2_norm(params: list[torch.nn.Parameter]) -> torch.Tensor:
        """Compute global L2 grad norm over a set of params (safely handles None grads)."""
        # Use float32 for stability even under mixed precision
        sq_sum = None
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            # handle fp16/bf16 grads robustly
            gn = g.float().norm(2)
            sq = gn * gn
            if sq_sum is None:
                sq_sum = sq
            else:
                sq_sum = sq_sum + sq
        if sq_sum is None:
            dev = params[0].device if len(params) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.zeros((), dtype=torch.float32, device=dev)
        return torch.sqrt(sq_sum).to(params[0].device)

    def _maybe_mark_cudagraph_step_begin(self) -> None:
        if not self.hparams.get('torch_compile'):
            return
        if self.device.type != 'cuda':
            return
        torch.compiler.cudagraph_mark_step_begin()

    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude evaluation metrics from saving."""
        state = super().state_dict(*args, **kwargs)

        # Remove evaluation metrics that should not be saved
        keys_to_remove = [
            k
            for k in state
            if any(
                pattern in k
                for pattern in [
                    '_teacher_model',
                    'eval_generative_perplexity',
                    'real_generative_perplexity',
                    'eval_gradient_moment',
                    'eval_entropy',
                    'real_entropy',
                    'eval_sliced_wasserstein',
                    'eval_token_distribution_kl',
                ]
            )
        ]

        for key in keys_to_remove:
            del state[key]

        return state

    def configure_model(self) -> None:
        """Configure the model.

        Subclasses should override this to instantiate their specific models
        e.g., self.model = ..., self.encoder = ...
        """
        raise NotImplementedError

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]] | list[torch.optim.Optimizer]:
        """Configure optimizers.

        Subclasses should override this method to configure their optimizers.
        """
        msg = 'Subclasses must implement configure_optimizers.'
        raise NotImplementedError(msg)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """On save checkpoint.

        Args:
            checkpoint: The checkpoint.

        """
        if self.hparams['use_ema']:
            ema_models_to_save = self.ema_models
            if not self.hparams.get('save_encoder', True):
                ema_models_to_save = {name: model for name, model in self.ema_models.items() if name != 'encoder'}
            if ema_models_to_save:
                if len(ema_models_to_save) == 1:
                    name, model_ema = next(iter(ema_models_to_save.items()))
                    key = 'ema_state_dict' if name == 'default' else f'{name}_ema_state_dict'
                    checkpoint[key] = model_ema.state_dict()
                else:
                    for name, model_ema in ema_models_to_save.items():
                        key = 'ema_state_dict' if name == 'default' else f'{name}_ema_state_dict'
                        checkpoint[key] = model_ema.state_dict()

        # Don't save evaluation metrics or perplexity models in checkpoint
        # They will be reloaded on demand

    def _maybe_remap_compiled_state_dict_keys(self, checkpoint: dict) -> None:
        """Remap checkpoint keys between compiled and eager module naming schemes.

        torch.compile wraps modules under ``_orig_mod``. A checkpoint saved from a compiled
        model therefore uses keys like ``model._orig_mod.*`` while an eager model expects
        ``model.*``. Conversely, a compiled model may need the reverse mapping when loading
        an eager checkpoint. If the normalized key spaces match, rewrite the checkpoint
        state_dict to the current module naming scheme before Lightning loads it.
        """
        checkpoint_state_dict = checkpoint.get('state_dict')
        if not isinstance(checkpoint_state_dict, dict) or not checkpoint_state_dict:
            return

        current_state_dict = super().state_dict()
        if not current_state_dict:
            return

        def _normalize_key(key: str) -> str:
            return key.replace('._orig_mod.', '.')

        checkpoint_keys = list(checkpoint_state_dict.keys())
        current_keys = list(current_state_dict.keys())
        if set(checkpoint_keys) == set(current_keys):
            return

        current_key_by_normalized = {_normalize_key(key): key for key in current_keys}
        normalized_checkpoint_keys = {_normalize_key(key) for key in checkpoint_keys}
        normalized_current_keys = set(current_key_by_normalized.keys())
        if normalized_checkpoint_keys != normalized_current_keys:
            return

        remapped_state_dict: dict[str, Any] = {}
        remapped_count = 0
        for key, value in checkpoint_state_dict.items():
            normalized_key = _normalize_key(key)
            target_key = current_key_by_normalized.get(normalized_key)
            if target_key is None:
                remapped_state_dict[key] = value
                continue
            remapped_state_dict[target_key] = value
            if target_key != key:
                remapped_count += 1

        checkpoint['state_dict'] = remapped_state_dict
        print_rank_zero(
            f'Remapped {remapped_count} checkpoint parameter keys between compiled and eager naming schemes.',
        )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """On load checkpoint.

        Args:
            checkpoint: The checkpoint.

        """
        self._maybe_remap_compiled_state_dict_keys(checkpoint)
        # reset ema nested level.
        if hasattr(self, '_ema_activation_nested_level'):
            self._ema_activation_nested_level = 0
        if self.hparams['use_ema']:
            print_rank_zero('Restoring EMA state from checkpoint...')
            for name, model_ema in self.ema_models.items():
                ckpt_key_name = 'ema_state_dict' if name == 'default' else f'{name}_ema_state_dict'
                try:
                    model_ema.load_state_dict(checkpoint[ckpt_key_name], strict=True)
                except Exception as e:
                    print_rank_zero(f'⚠️ Error loading ema key in checkpoint for {ckpt_key_name}: {e}')
                model_ema.to(self.device)
                print_rank_zero('Successfully loaded EMA state for', name)
                # Fix EMA step synchronization issue when resuming
                if 'global_step' in checkpoint:
                    model_ema.optimization_step = checkpoint['global_step']

    def _load_weights_only(self, restore_optimizer: bool = True) -> None:
        path = self.hparams.get('weights_only_path')
        if path:
            print_rank_zero(f'⚡ Restoring state from checkpoint: {path}')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

            # 1. Restore the global_step (must be done on the fit_loop)
            # should modify trainer.lr_schedulers on_train_start if you want to.
            # I cannot implement it correctly, lightning is such a horrible framework to work with
            # if 'global_step' in checkpoint:
            #     step = checkpoint['global_step']
            #     self.trainer.fit_loop._global_step = step
            #     print_rank_zero(f'✅ Restored global_step to {step}')
            #     # should modify trainer.lr_schedulers on_train_start if you want to.

            # 2. Restore main model weights
            # Get checksum before loading
            model_params_before = {name: param.data.clone() for name, param in self.named_parameters()}
            param_sum_before = sum(p.sum().item() for p in model_params_before.values())
            print_rank_zero(f'Model parameter sum BEFORE loading: {param_sum_before:.6f}')

            load_weights_agnostic(self, checkpoint)

            # Get checksum after loading
            param_sum_after = sum(p.sum().item() for name, p in self.named_parameters())
            print_rank_zero(f'Model parameter sum AFTER loading: {param_sum_after:.6f}')
            print_rank_zero(f'Parameters changed: {param_sum_before != param_sum_after}')

            # --- RE-INITIALIZE EMA INSTEAD OF LOADING ---
            # if self.hparams.get('use_ema', False):
            # print_rank_zero('✅ Weights loaded. Re-initializing EMA from the new model state for fine-tuning.')
            # This assumes your EMA models are already instantiated but need to be reset.
            # A common way to do this is to re-create them or load the main model's state.
            # for name in self.ema_models:
            #     # Create a new EMAModel instance based on the *current* model state
            #     # Note: You might need to pass decay and other params from your original EMA setup
            #     # For simplicity, let's assume you can access the original decay rate.
            #     original_decay = self.ema_models[name].decay
            #     self.ema_models[name] = EMAModel(
            #         self.models_parameters_for_ema_update[name],
            #         decay=original_decay,
            #     ).to(self.device)

            # --- LOAD EMA WITH STRICT=FALSE ---
            # A dict representation would make strict=False handling easier here.
            print_rank_zero('Restoring EMA state by calling `on_load_checkpoint`...')
            try:
                self.on_load_checkpoint(checkpoint)
                print_rank_zero('restored EMA state from checkpoint')
            except Exception as e:
                print_rank_zero(f'⚡ Failed to load EMA state: {e}')
                print_rank_zero(traceback.format_exc())

            # restore optimizer:
            optim_all_passed = True
            if restore_optimizer:
                print_rank_zero('⚡ Restoring optimizer state from checkpoint...')
                # Attempt to restore optimizer states, allowing for partial mismatches
                optim_all_passed = restore_optim_from_checkpoint(self.trainer, checkpoint)

            # restore lr
            # restore_lr_from_checkpoint(self.trainer, checkpoint)

            if not optim_all_passed:
                print_rank_zero('⚠️ Some optimizers failed to load. Enabling LR re-entry mechanism.')
                # If optimizer states didn't load (e.g., param group mismatch), enable re-entry:
                self._reentry = _ReentryLRController(zero_steps=2000, warm_steps=2000, target_scale=1.0)
                # Snapshot base LRs so we know what to aim for
                pgs = _get_all_param_groups(self.trainer)
                self._reentry.base_lrs = [pg['lr'] for pg in pgs]

            # Clean up to prevent re-loading
            self.hparams.weights_only_path = None

            # Mark that we loaded weights for later verification
            # self._weights_loaded_checksum = param_sum_after

    def on_fit_start(self) -> None:
        """
        Called at the very beginning of fit, after model has been moved to device
        and wrapped by strategies like FSDP. This is the correct place to
        load weights-only checkpoints.
        """
        # self._load_weights_only()
        self._print_dataset_sizes()

    def _activate_ema(self) -> bool:
        """Activate EMA weights; move to a higher nested level."""
        print_rank_zero(f'Activating EMA: level={self._ema_activation_nested_level}, step={self.global_step}')
        if (self._ema_activation_nested_level == 0) and self.hparams.get('use_ema', False) and hasattr(self, 'ema_models'):
            # Check if EMA has been stepped at least once
            for model_name, ema_model in self.ema_models.items():
                if (
                    ema_model is not None
                    and ema_model.optimization_step > 0
                    and hasattr(ema_model, 'temp_stored_params')
                    and ema_model.temp_stored_params is None
                ):
                    ema_model.store(self.models_parameters_for_ema_update[model_name])
                    ema_model.copy_to(self.models_parameters_for_ema_update[model_name])

            self._ema_activation_nested_level += 1
        return self._ema_activation_nested_level > 0

    def _deactivate_ema(self) -> bool:
        """Deactivate EMA weights; move to a lower nested level."""
        print_rank_zero(f'Deactivating EMA: level={self._ema_activation_nested_level}, step={self.global_step}')
        try:
            if self.hparams['use_ema']:
                if self._ema_activation_nested_level == 1:
                    for model_name, model_ema in self.ema_models.items():
                        if model_ema.temp_stored_params is not None:
                            model_ema.restore(self.models_parameters_for_ema_update[model_name])
                            self._ema_activation_nested_level = 0
                elif self._ema_activation_nested_level > 0:
                    self._ema_activation_nested_level -= 1
                else:
                    raise ValueError('EMA deactivation called without matching activation')
        except Exception as e:
            print_rank_zero(f'Error during EMA deactivation: {e}')
            # print the traceback
            traceback.print_exc()
        return self._ema_activation_nested_level == 0

    @contextmanager
    def ema_scope(self, context: str | None = None):
        """EMA scope.

        Args:
            context: The context.

        """
        if self._activate_ema() and (context is not None):
            print_rank_zero(f'{context}: Using EMA weights in ema_scope')
        try:
            yield None
        finally:
            ema_deactivated = self._deactivate_ema()
            if ema_deactivated and (context is not None):
                print_rank_zero(f'{context}: Restored training weights')

    def ema_step(self) -> None:
        """EMA step for one or multiple models."""
        for name, model_ema in self.ema_models.items():
            model_ema.step(self.models_parameters_for_ema_update[name])

    def get_batch_size(self, batch: dict[str, Any]) -> int:
        """Get the batch size.

        Args:
            batch: The batch.

        """
        return batch['input_ids'].size(0)

    def _print_dataset_sizes(self) -> None:
        if self._printed_dataset_info:
            return
        datamodule = getattr(self.trainer, 'datamodule', None)
        if datamodule is None:
            return
        datasets: dict[str, Any] = {}
        if isinstance(getattr(datamodule, 'datasets', None), dict):
            datasets.update({k: v for k, v in datamodule.datasets.items() if v is not None})
        for key in ('train_dataset', 'val_dataset', 'test_dataset', 'predict_dataset'):
            if hasattr(datamodule, key):
                dataset = getattr(datamodule, key)
                if dataset is not None:
                    datasets.setdefault(key.replace('_dataset', ''), dataset)
        if not datasets:
            return
        parts = []
        for name in sorted(datasets):
            dataset = datasets[name]
            size = None
            if hasattr(dataset, 'num_samples'):
                try:
                    size = dataset.num_samples
                except Exception:
                    size = None
            if size is None and hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__'):
                size = 'unknown'
            if size is None:
                try:
                    size = len(dataset)
                except Exception:
                    size = 'unknown'
            parts.append(f'{name}={size}')
        print_rank_zero(f'Dataset sizes: {", ".join(parts)}')
        self._printed_dataset_info = True

    def _decode_batch_samples(
        self,
        input_ids: torch.Tensor | list | None,
        attention_mask: torch.Tensor | list | None = None,
        max_samples: int = 2,
    ) -> list[str]:
        if self.tokenizer is None or input_ids is None:
            return []
        decoded: list[str] = []
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            sample_count = min(max_samples, input_ids.size(0))
            for idx in range(sample_count):
                seq = input_ids[idx]
                if attention_mask is not None:
                    mask = attention_mask[idx] if isinstance(attention_mask, torch.Tensor) else attention_mask[idx]
                    # Use the mask to filter out padding tokens (where mask == 0)
                    if isinstance(mask, torch.Tensor):
                        seq = seq[mask.bool()]
                    else:
                        # For list-based masks, extract non-padded positions
                        seq = seq[[i for i, m in enumerate(mask) if m == 1]]
                decoded.append(self.tokenizer.decode(seq.tolist(), skip_special_tokens=False))
            return decoded

        sample_count = min(max_samples, len(input_ids))
        for idx in range(sample_count):
            seq = input_ids[idx]
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            decoded.append(self.tokenizer.decode(seq, skip_special_tokens=False))
        return decoded

    def _decode_generated_samples(
        self,
        samples: SampleOutput | torch.Tensor | None,
        max_samples: int = 2,
    ) -> list[str]:
        if samples is None:
            return []
        if hasattr(samples, 'completions') and samples.completions:
            return list(samples.completions)[:max_samples]
        output_ids = samples.output_ids if hasattr(samples, 'output_ids') else samples
        if output_ids is None:
            return []
        if self.tokenizer is None:
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids[:max_samples].tolist()
            else:
                output_ids = output_ids[:max_samples]
            return [str(seq) for seq in output_ids]
        return self._decode_batch_samples(output_ids, attention_mask=None, max_samples=max_samples)

    def _print_samples(self, header: str, samples: list[str]) -> None:
        if not samples:
            return
        print_rank_zero(f'{header}')
        for idx, sample in enumerate(samples):
            print_rank_zero(f'=== Sample {idx + 1} ===\n' + sample + '\n')

    def forward(self, batch: dict[str, Any], batch_idx: int, **kwargs: dict[str, Any]) -> dict:
        """Forward pass.

        Args:
            batch: The batch.
            batch_idx: The batch index.
            **kwargs: Additional keyword arguments.

        """
        raise NotImplementedError

    def shared_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        step: str = 'train',
        prefix: str = '',
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Shared step.

        Args:
            batch: The batch.
            batch_idx: The batch index.
            step: The step.
            prefix: The prefix.
            **kwargs: Additional keyword arguments.

        """
        outputs = self(batch, batch_idx, **kwargs)
        loss = outputs['loss']
        loss_dict = {f'{step}/{prefix}{k}': v for k, v in outputs.items() if v is not None}
        if self.trainer.sanity_checking:
            return loss, loss_dict
        loss_dict = self._compute_metrics(loss, loss_dict, batch, step)
        loss_dict = self._compute_sensitivity_metrics(loss_dict, batch, step)
        return loss, loss_dict

    def on_train_start(self):
        self._load_weights_only()
        self._ensure_ema_states_on_correct_device()

    def _ensure_ema_states_on_correct_device(self) -> None:
        """Move EMA shadow parameters onto the same device as the tracked model parameters."""
        if not self.hparams.get('use_ema', False):
            return
        for name, ema_model in self.ema_models.items():
            if ema_model is None:
                continue
            params = self.models_parameters_for_ema_update.get(name)
            if not params:
                continue
            reference_param = next((p for p in params if p is not None), None)
            if reference_param is None:
                continue
            target_device = reference_param.device
            if len(ema_model.shadow_params) == 0:
                continue
            current_device = ema_model.shadow_params[0].device
            if current_device == target_device:
                continue
            ema_model.to(device=target_device)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: The batch.
            batch_idx: The batch index.

        """
        self._maybe_mark_cudagraph_step_begin()

        # print the first batch text, decode the input_ids with the tokenizer
        # if (not self._printed_first_train_batch) and batch_idx == 0 and self.tokenizer is not None:
        if batch_idx == 0 and (self.tokenizer is not None):
            decoded = self._decode_batch_samples(
                batch.get('input_ids'),
                batch.get('attention_mask'),
                max_samples=2,
            )
            self._print_samples('First training batch samples:', decoded)
            # also print the latents if present. We will just print their tensor values (item -> str)
            if 'latent' in batch:
                latents = batch['latent']
                if isinstance(latents, torch.Tensor):
                    latents = latents[:2].tolist()
                latent_strs = [str(latent) for latent in latents]
                self._print_samples('First training batch latents:', latent_strs)
            self._printed_first_train_batch = True

        batch_size = self.get_batch_size(batch)
        loss, loss_dict = self.shared_step(batch, batch_idx)
        
        should_log_train = self.trainer.is_global_zero and (
            self.trainer.log_every_n_steps > 0
            and (self.global_step + 1) % self.trainer.log_every_n_steps == 0
        )
        if should_log_train:
            # Add global step
            self.log(
                'global_step',
                int(self.global_step),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
            self.log_dict(
                loss_dict,
                batch_size=batch_size,
                prog_bar=False, # don't put these in the prog bar to avoid clutter, but still log them to other loggers
                logger=True,
                on_step=True,  # Explicitly log on step
                on_epoch=False,  # on epoch too
                sync_dist=False,  # Sync or not across distributed processes (mean reduction)
                # It is fine not to sync training metrics and we gain a lot of time. We want low variance at validation only.
            )
        return loss

    # def on_before_optimizer_step(self, optimizer):
    #     """Compute the gradient norm of all trainable attribute"""
    #     for name, param in self.named_parameters():
    #         # check that param is in train mode
    #         if param.requires_grad:
    #             grad_norm = param.grad.norm(2).item()
    #             self.log(
    #                 f'train/grad_norm_{name}',
    #                 grad_norm,
    #                 logger=True,
    #                 on_step=True,
    #                 sync_dist=True,
    #             )

    def on_before_optimizer_step(self, optimizer) -> None:
        """Compute L2 gradient norms per available submodule and store/log them.

        Called by Lightning after backward (and after unscaling in AMP), just before the optimizer step.
        """
        
        if not self.enable_grad_norms_logging:
            return
        
        # Collect per-module grad norms
        grad_norms: dict[str, torch.Tensor] = {}
        device = self.device
        total_sq = torch.zeros((), dtype=torch.float32, device=self.device)

        # Top-level children that have trainable params
        for name, _module, params in self._iter_trainable_children():
            n = self._grad_l2_norm(params).to(device)
            grad_norms[name] = n
            total_sq = total_sq + (n * n)

        total_norm = torch.sqrt(total_sq)
        grad_norms['total'] = total_norm

        # Store a python-float snapshot (rank-local; logging below uses sync_dist=True)
        self.grad_norms = {k: float(v.item()) for k, v in grad_norms.items()}

        # ---- Logging (DDP-safe) ----
        # Always log the total
        self.log(
            'train/grad_norm_total',
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False, # fine during training, we can handle some noise here to save latency
        )

        # Log a bounded number of per-module norms to avoid metric spam
        to_log = list(grad_norms.items())
        # Put "total" first, then others alphabetically (skip 'total' in the loop below)
        to_log = [(k, v) for k, v in to_log if k != 'total']
        to_log.sort(key=lambda kv: kv[0])
        to_log = to_log[: self._max_grad_norm_logs]

        per_module_logs = {f'train/grad_norm/{k}': v.to(device) for k, v in to_log}
        if per_module_logs:
            self.log_dict(
                per_module_logs,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

    def optimizer_step(self, *args: tuple[Any], **kwargs: dict[str, Any]) -> None:
        """Optimizer step.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        """
        # --- RE-ENTRY LR OVERRIDE ---
        if hasattr(self, '_reentry') and self._reentry is not None:
            step = int(self.global_step)
            pgs = _get_all_param_groups(self.trainer)
            for pg, base_lr in zip(pgs, self._reentry.base_lrs, strict=False):
                pg['lr'] = self._reentry.lr_at(step, base_lr)

        super().optimizer_step(*args, **kwargs)

        # This is now handled in training_step to ensure sync with optimizer
        if self.hparams['use_ema']:
            self.ema_step()

    def _lazy_initialize_generative_perplexity(self):
        # LAZY INITIALIZATION: Create the metric only when validation starts.
        # self.tokenizer should have been set by now.
        if self.enable_generative_perplexity and self.eval_and_log_first_n_batches > 0:
            # Check if any of the metrics need initialization
            first_key = next(iter(self.inference_step_suffixes.keys()))
            if (
                isinstance(self.eval_generative_perplexity[first_key], GenerativePerplexityMetric)
                and self.eval_generative_perplexity[first_key].perplexity_model is None
            ):
                print_rank_zero('Lazily initializing GenerativePerplexityMetric for all inference step configs...')
                perplexity_model, perplexity_tokenizer = self._lazy_initialize_teacher_model()

                # Initialize all metric instances
                for key in self.inference_step_suffixes:
                    self.eval_generative_perplexity[key].lazy_initialize(
                        perplexity_model=perplexity_model,
                        perplexity_tokenizer=perplexity_tokenizer,
                        source_tokenizer=self.tokenizer,
                        max_length=self.sequence_length,
                    )
                    self.real_generative_perplexity[key].lazy_initialize(
                        perplexity_model=perplexity_model,
                        perplexity_tokenizer=perplexity_tokenizer,
                        source_tokenizer=self.tokenizer,
                        max_length=self.sequence_length,
                    )

    def _lazy_initialize_teacher_model(self) -> tuple[torch.nn.Module, Any]:
        if self._teacher_model is None or self._teacher_tokenizer is None:
            print_rank_zero(f'Loading teacher model/tokenizer from {self.teacher_model_name_or_path}...')
            with torch.inference_mode(False):
                teacher_model, teacher_tokenizer = load_teacher_model(self.teacher_model_name_or_path)
                # Keep teacher model out of Lightning/nn.Module registration to avoid
                # checkpoint bloat and optimizer/strategy side effects.
                object.__setattr__(self, '_teacher_model', teacher_model.to(self.device))
            self._teacher_tokenizer = teacher_tokenizer
        else:
            with torch.inference_mode(False):
                object.__setattr__(self, '_teacher_model', self._teacher_model.to(self.device))
        return self._teacher_model, self._teacher_tokenizer

    def _lazy_initialize_gradient_moment_metric(self) -> None:
        # LAZY INITIALIZATION: Create the metric only when validation/prediction starts.
        # self.tokenizer should have been set by now.
        if self.enable_gradient_moment_metric and self.eval_and_log_first_n_batches > 0:
            first_key = next(iter(self.inference_step_suffixes.keys()))
            if (
                isinstance(self.eval_gradient_moment[first_key], GradientMomentMetric)
                and self.eval_gradient_moment[first_key].reference_model is None
            ):
                print_rank_zero('Lazily initializing GradientMomentMetric for all inference step configs...')
                reference_model, reference_tokenizer = self._lazy_initialize_teacher_model()

                for key in self.inference_step_suffixes:
                    self.eval_gradient_moment[key].lazy_initialize(
                        reference_model=reference_model,
                        reference_tokenizer=reference_tokenizer,
                        source_tokenizer=self.tokenizer,
                        max_length=self.sequence_length,
                    )

    def _reset_gradient_moment_buffer(self) -> None:
        self._gradient_moment_prev_batches = dict.fromkeys(self.inference_step_suffixes)

    def _warn_if_gradient_moment_has_too_few_batches(self) -> None:
        if not self.enable_gradient_moment_metric:
            return
        if int(self.eval_and_log_first_n_batches) < 2:
            print_rank_zero(
                'GradientMomentMetric is enabled but eval_and_log_first_n_batches < 2. '
                'At least 2 batches are required to form one unbiased estimator sample.',
            )

    def _normalize_inference_steps(self, num_inference_steps: Any) -> list[int]:
        if isinstance(num_inference_steps, (list, tuple)):
            raw_steps = list(num_inference_steps)
        elif num_inference_steps is None:
            raw_steps = []
        else:
            raw_steps = [num_inference_steps]

        steps: list[int] = []
        seen: set[int] = set()
        for step in raw_steps:
            if step is None:
                continue
            try:
                step_int = int(step)
            except (TypeError, ValueError):
                continue
            if step_int < 1 or step_int in seen:
                continue
            steps.append(step_int)
            seen.add(step_int)
        return steps

    def _build_sampling_step_configs(
        self,
        num_inference_steps: Any,
        num_inference_steps_latent: Any = None,
    ) -> list[dict[str, Any]]:
        x_steps = self._normalize_inference_steps(num_inference_steps)
        y_steps = self._normalize_inference_steps(num_inference_steps_latent)
        if not x_steps:
            return []

        if not y_steps:
            return [
                {
                    'metric_key': f'steps_{step}',
                    'suffix': f'_{step}',
                    'num_inference_steps': step,
                    'num_inference_steps_latent': None,
                    'total_num_inference_steps': step,
                }
                for step in x_steps
            ]

        if len(y_steps) == 1:
            paired_steps = list(zip(x_steps, y_steps * len(x_steps), strict=False))
        elif len(x_steps) == 1:
            paired_steps = list(zip(x_steps * len(y_steps), y_steps, strict=False))
        elif len(x_steps) == len(y_steps):
            paired_steps = list(zip(x_steps, y_steps, strict=False))
        else:
            raise ValueError(
                'num_inference_steps and num_inference_steps_latent must either have the same length, '
                'or one of them must be a scalar.',
            )

        use_explicit_pair_suffix = len(y_steps) > 1
        configs: list[dict[str, Any]] = []
        seen_suffixes: set[str] = set()
        for x_step, y_step in paired_steps:
            suffix = f'_x{x_step}_y{y_step}' if use_explicit_pair_suffix else f'_{x_step}'
            if suffix in seen_suffixes:
                continue
            seen_suffixes.add(suffix)
            configs.append(
                {
                    'metric_key': f'steps_{suffix.lstrip("_")}',
                    'suffix': suffix,
                    'num_inference_steps': x_step,
                    'num_inference_steps_latent': y_step,
                    'total_num_inference_steps': x_step + y_step,
                },
            )
        return configs

    def _reset_inference_step_metrics(self) -> None:
        self.eval_entropy = torch.nn.ModuleDict(
            {
                key: EntropyMetric() if self.enable_entropy else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.real_entropy = torch.nn.ModuleDict(
            {
                key: EntropyMetric() if self.enable_entropy else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.eval_generative_perplexity = torch.nn.ModuleDict(
            {
                key: GenerativePerplexityMetric() if self.enable_generative_perplexity else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.real_generative_perplexity = torch.nn.ModuleDict(
            {
                key: GenerativePerplexityMetric() if self.enable_generative_perplexity else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.eval_gradient_moment = torch.nn.ModuleDict(
            {
                key: (
                    GradientMomentMetric(
                        include_parameter_regex=self.gradient_moment_include_parameter_regex,
                    )
                    if self.enable_gradient_moment_metric
                    else torch.nn.Identity()
                )
                for key in self.inference_step_suffixes
            },
        )
        self.eval_sliced_wasserstein = torch.nn.ModuleDict(
            {
                key: SlicedWassersteinMetric() if self.enable_sliced_wasserstein else torch.nn.Identity()
                for key in self.inference_step_suffixes
            },
        )
        self.eval_token_distribution_kl = torch.nn.ModuleDict(
            {
                key: (
                    TokenDistributionKLMetric(vocab_size=self.vocab_size)
                    if self.enable_token_distribution_kl
                    else torch.nn.Identity()
                )
                for key in self.inference_step_suffixes
            },
        )
        self.eval_entropy.to(self.device)
        self.real_entropy.to(self.device)
        self.eval_generative_perplexity.to(self.device)
        self.real_generative_perplexity.to(self.device)
        self.eval_gradient_moment.to(self.device)
        self.eval_sliced_wasserstein.to(self.device)
        self.eval_token_distribution_kl.to(self.device)
        self._reset_gradient_moment_buffer()

    def _configure_inference_steps(
        self,
        num_inference_steps: Any,
        num_inference_steps_latent: Any = None,
    ) -> list[int]:
        sampling_configs = self._build_sampling_step_configs(
            num_inference_steps,
            num_inference_steps_latent=num_inference_steps_latent,
        )
        if not sampling_configs:
            self._inference_step_list = []
            self._sampling_step_configs = []
            return []
        suffixes = {config['metric_key']: config['suffix'] for config in sampling_configs}
        if suffixes != self.inference_step_suffixes:
            self.inference_step_suffixes = suffixes
            self._reset_inference_step_metrics()
        self._sampling_step_configs = sampling_configs
        self._inference_step_list = [int(config['num_inference_steps']) for config in sampling_configs]
        return self._inference_step_list

    def on_validation_start(self) -> None:
        if self.trainer.sanity_checking:
            return
        # Load model weights if necessary
        self._load_weights_only()
        # Switch to EMA weights before validation.
        self._activate_ema()
        self._configure_inference_steps(
            getattr(self.sampler, 'num_inference_steps', None),
            getattr(self.sampler, 'num_inference_steps_latent', None),
        )
        self.real_samples_for_logging = []
        self.generated_samples_for_logging = []
        self._printed_first_val_samples = False
        # LAZY INITIALIZATION: Create the metric only when validation starts.
        self._lazy_initialize_generative_perplexity()
        self._lazy_initialize_gradient_moment_metric()
        self._warn_if_gradient_moment_has_too_few_batches()
        self._reset_gradient_moment_buffer()
        if (self.tokenizer is not None) and hasattr(self.tokenizer, 'pad_token_id'):
            if self.enable_entropy:
                for key in self.inference_step_suffixes:
                    self.eval_entropy[key].pad_token_id = self.tokenizer.pad_token_id
                    self.real_entropy[key].pad_token_id = self.tokenizer.pad_token_id
            if self.enable_sliced_wasserstein:
                for key in self.inference_step_suffixes:
                    self.eval_sliced_wasserstein[key].pad_token_id = self.tokenizer.pad_token_id
            if self.enable_token_distribution_kl:
                for key in self.inference_step_suffixes:
                    self.eval_token_distribution_kl[key].pad_token_id = self.tokenizer.pad_token_id

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validate step.

        Args:
            batch: The batch.
            batch_idx: The batch index.

        """
        self._maybe_mark_cudagraph_step_begin()
        # print the batch text, decode the input_ids with the tokenizer
        if batch_idx == 0 and (self.tokenizer is not None):
            decoded = self._decode_batch_samples(
                batch.get('input_ids'),
                batch.get('attention_mask'),
                max_samples=2,
            )
            self._print_samples('Validation batch samples:', decoded)
            if 'latent' in batch:
                latents = batch['latent']
                if isinstance(latents, torch.Tensor):
                    latents = latents[:2].tolist()
                latent_strs = [str(latent) for latent in latents]
                self._print_samples('Validation batch latents:', latent_strs)

        batch_size = self.get_batch_size(batch)
        loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        if not self.on_validation_only_eval:
            # Skip computation if we are only evaluating on a few batches
            # EMA weights are already active from on_validation_start
            # No need for torch.no_grad as validation_step is automatically no_grad
            loss, loss_dict = self.shared_step(batch, batch_idx=batch_idx, step='val')

        # Skip logging and sampling during sanity check
        if self.trainer.sanity_checking:
            return loss

        # Generate samples during the first few validation batches
        if batch_idx < self.eval_and_log_first_n_batches and hasattr(self, 'sample'):
            if self.limit_eval_batch_size is not None:
                batch = {k: v[: self.limit_eval_batch_size] for k, v in batch.items()}
                batch_size = self.get_batch_size(batch)

            sampling_step_configs = self._sampling_step_configs
            if not sampling_step_configs:
                self._configure_inference_steps(
                    getattr(self.sampler, 'num_inference_steps', None),
                    getattr(self.sampler, 'num_inference_steps_latent', None),
                )
                sampling_step_configs = self._sampling_step_configs

            for idx, sampling_config in enumerate(sampling_step_configs):
                num_steps = sampling_config['num_inference_steps']
                suffix = sampling_config['suffix']
                if num_steps is None or num_steps < 1:
                    continue

                sample_kwargs = {
                    'num_inference_steps': num_steps,
                }
                num_inference_steps_latent = sampling_config.get('num_inference_steps_latent')
                if num_inference_steps_latent is not None:
                    sample_kwargs['num_inference_steps_latent'] = num_inference_steps_latent

                samples = self.sample(
                    batch,
                    max_batch_size=batch_size,
                    skip_special_tokens=False,
                    **sample_kwargs,
                )

                if samples is not None:
                    # Extract token sequences
                    loss_dict = self._compute_generative_metrics(
                        loss_dict,
                        batch,
                        generated_samples=samples,
                        step='val',
                        suffix=suffix,
                    )

                    # Only update total count once (for the first configured steps)
                    if idx == 0:
                        if not self._printed_first_val_samples:
                            decoded = self._decode_generated_samples(samples, max_samples=2)
                            self._print_samples('Generated samples:', decoded)
                            self._printed_first_val_samples = True
                        num_generated = samples.output_ids.shape[0]
                        self.val_num_generated_samples.update(num_generated)
                        loss_dict['val/val_num_generated_samples'] = self.val_num_generated_samples

                        # Store samples for logging (rank 0 only)
                        if self.trainer.global_rank == 0 and len(self.generated_samples_for_logging) < 3:
                            self.generated_samples_for_logging.append(samples)
                            self.real_samples_for_logging.append(batch['input_ids'])

        # 4. Log everything in a single call
        if loss_dict != {}:
            self.log(
                'step',
                int(self.global_step),
                batch_size=batch_size,
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_epoch=True,
                on_step=False,
            )
            self._log_dict_with_metric_attributes(
                loss_dict,
                batch_size=batch_size,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                on_epoch=True,
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute evaluation metrics, log samples, and reset metrics."""
        # Skip during sanity check
        if self.trainer.sanity_checking:
            return

        # metrics_to_log: dict[str, torchmetrics.Metric] = {
        #     'eval/generative_perplexity': self.eval_generative_perplexity,
        #     'eval/real_generative_perplexity': self.real_generative_perplexity,
        #     'eval/entropy': self.eval_entropy,
        #     'eval/real_entropy': self.real_entropy,
        #     'eval/sliced_wasserstein': self.eval_sliced_wasserstein,
        #     'eval/token_distribution_kl': self.eval_token_distribution_kl,
        # }
        # for key, metric in metrics_to_log.items():
        #     if metric is not None:
        #         try:
        #             metrics_to_log[key] = metric.compute()
        #             metric.reset()
        #         except Exception as e:
        #             print_rank_zero(f'Warning: Failed to compute {key} in validation: {e}')

        # metrics_to_log = {k: v for k, v in metrics_to_log.items() if v is not None}
        # # Log evaluation metrics
        # if metrics_to_log:
        #     print_rank_zero(f'Logging evaluation metrics for validation: {metrics_to_log}')
        #     self.log_dict(metrics_to_log, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        # Log sample text (rank 0 only)
        if self.trainer.global_rank == 0 and self.generated_samples_for_logging:
            self._log_samples()

        # Clear generated samples
        self.real_samples_for_logging = []
        self.generated_samples_for_logging = []

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        """Run prediction, and then restore training weights after validation."""
        self._reset_gradient_moment_buffer()
        self._deactivate_ema()

    def on_predict_start(self) -> None:
        # Load model weights if necessary
        self._load_weights_only()
        # Switch to EMA weights before prediction.
        self._activate_ema()
        self._configure_inference_steps(
            getattr(self.sampler, 'num_inference_steps', None),
            getattr(self.sampler, 'num_inference_steps_latent', None),
        )
        # LAZY INITIALIZATION: Create the metric only when prediction starts.
        self._lazy_initialize_generative_perplexity()
        self._lazy_initialize_gradient_moment_metric()
        self._warn_if_gradient_moment_has_too_few_batches()
        self._reset_gradient_moment_buffer()
        if (self.tokenizer is not None) and hasattr(self.tokenizer, 'pad_token_id'):
            if self.enable_entropy:
                for key in self.inference_step_suffixes:
                    self.eval_entropy[key].pad_token_id = self.tokenizer.pad_token_id
                    self.real_entropy[key].pad_token_id = self.tokenizer.pad_token_id
            if self.enable_sliced_wasserstein:
                for key in self.inference_step_suffixes:
                    self.eval_sliced_wasserstein[key].pad_token_id = self.tokenizer.pad_token_id
            if self.enable_token_distribution_kl:
                for key in self.inference_step_suffixes:
                    self.eval_token_distribution_kl[key].pad_token_id = self.tokenizer.pad_token_id

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> SampleOutput | None:
        """Prediction step for distributed sampling.

        Args:
            batch: Batch containing placeholder data for sampling
            batch_idx: Batch index
            dataloader_idx: Dataloader index

        Returns:
            SampleOutput with generated samples
        """
        self._maybe_mark_cudagraph_step_begin()
        if self.limit_eval_batch_size is not None:
            batch = {k: v[: self.limit_eval_batch_size] for k, v in batch.items()}

        # Extract sampling parameters from batch
        batch_size = self.get_batch_size(batch)

        if batch_idx < self.eval_and_log_first_n_batches:
            sampling_step_configs = self._sampling_step_configs
            if not sampling_step_configs:
                self._configure_inference_steps(
                    getattr(self.sampler, 'num_inference_steps', None),
                    getattr(self.sampler, 'num_inference_steps_latent', None),
                )
                sampling_step_configs = self._sampling_step_configs

            samples_default = None
            for idx, sampling_config in enumerate(sampling_step_configs):
                num_steps = sampling_config['num_inference_steps']
                suffix = sampling_config['suffix']
                if num_steps is None or num_steps < 1:
                    continue

                sample_kwargs = {
                    'num_inference_steps': num_steps,
                }
                num_inference_steps_latent = sampling_config.get('num_inference_steps_latent')
                if num_inference_steps_latent is not None:
                    sample_kwargs['num_inference_steps_latent'] = num_inference_steps_latent

                samples: SampleOutput = self.sample(
                    batch=batch,
                    max_batch_size=batch_size,
                    skip_special_tokens=False,
                    **sample_kwargs,
                )

                # Update evaluation metrics with generated samples
                if samples is not None:
                    # Extract token sequences
                    loss_dict = {}
                    loss_dict = self._compute_generative_metrics(
                        loss_dict=loss_dict,
                        batch=batch,
                        generated_samples=samples,
                        step='pred',
                        suffix=suffix,
                    )

                    # Store the first configured samples to return
                    if idx == 0:
                        samples_default = samples

            return samples_default
        return None

    def on_predict_epoch_end(self) -> None:
        """Compute and log evaluation metrics at the end of prediction."""
        metrics_to_log: dict[str, torch.Tensor] = {}

        # Iterate over all metric instances for all inference step configurations
        for metric_key, log_suffix in self.inference_step_suffixes.items():
            if self.enable_generative_perplexity:
                metrics_to_log[f'eval/generative_perplexity{log_suffix}'] = self.eval_generative_perplexity[
                    metric_key
                ].compute()
                self.eval_generative_perplexity[metric_key].reset()
                metrics_to_log[f'eval/real_generative_perplexity{log_suffix}'] = self.real_generative_perplexity[
                    metric_key
                ].compute()
                self.real_generative_perplexity[metric_key].reset()

            if self.enable_gradient_moment_metric:
                metrics_to_log[f'eval/gradient_moment{log_suffix}'] = self.eval_gradient_moment[metric_key].compute()
                self.eval_gradient_moment[metric_key].reset()

            if self.enable_entropy:
                metrics_to_log[f'eval/entropy{log_suffix}'] = self.eval_entropy[metric_key].compute()
                self.eval_entropy[metric_key].reset()
                metrics_to_log[f'eval/real_entropy{log_suffix}'] = self.real_entropy[metric_key].compute()
                self.real_entropy[metric_key].reset()

            if self.enable_sliced_wasserstein:
                metrics_to_log[f'eval/sliced_wasserstein{log_suffix}'] = self.eval_sliced_wasserstein[
                    metric_key
                ].compute()
                self.eval_sliced_wasserstein[metric_key].reset()

            if self.enable_token_distribution_kl:
                metrics_to_log[f'eval/token_distribution_kl{log_suffix}'] = self.eval_token_distribution_kl[
                    metric_key
                ].compute()
                self.eval_token_distribution_kl[metric_key].reset()

        # Log evaluation metrics
        if metrics_to_log:
            print_rank_zero(f'Computed evaluation metrics for prediction: {metrics_to_log}')
            # self.log_dict(metrics_to_log, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self._reset_gradient_moment_buffer()

    def on_predict_end(self) -> None:
        """Run prediction, and then restore training weights after prediction."""
        self._reset_gradient_moment_buffer()
        self._deactivate_ema()

    def sample(
        self,
        batch: dict[str, Any],
        max_batch_size: int | None = None,
        skip_special_tokens: bool = False,
        **kwargs: dict[str, Any],
    ) -> SampleOutput:
        """Generate samples from the model.

        Ema weights will be used if enabled.
        """
        raise NotImplementedError

    def _compute_metrics(
        self,
        loss: torch.tensor,
        loss_dict: dict,
        batch: dict[str, Any],
        step: str,
    ) -> dict[str, torch.Tensor]:
        """Compute metrics against generated samples.

        Ema weights will be used if enabled.
        """
        # Skip all metrics computation during sanity check
        if self.trainer.sanity_checking:
            return None

        batch_size = self.get_batch_size(batch)

        if step == 'train':
            # Update and prepare token metrics
            attention_mask = batch.get('attention_mask', torch.ones_like(batch['input_ids']))
            num_tokens = attention_mask.sum().item()

            # Update perplexity and BPC metrics
            self.train_perplexity.update(loss_dict[f'{step}/elbo_x'] * batch_size, num_tokens)
            self.train_bpc.update(loss_dict[f'{step}/elbo_x'] * batch_size, num_tokens)

            # Add perplexity and BPC to logged metrics

            loss_dict.update(
                {
                    'train/perplexity': self.train_perplexity,
                    'train/bpc': self.train_bpc,
                },
            )
            # self.log('train/perplexity', self.train_perplexity, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log('train/bpc', self.train_bpc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        elif step == 'val':
            # Get number of tokens from attention mask if available
            attention_mask = batch.get('attention_mask', torch.ones_like(batch['input_ids']))
            num_tokens = attention_mask.sum().item()
            self.val_perplexity.update(loss_dict[f'{step}/elbo_x'] * batch_size, num_tokens)
            self.val_bpc.update(loss_dict[f'{step}/elbo_x'] * batch_size, num_tokens)

            # Add perplexity and BPC to validation metrics
            loss_dict.update(
                {
                    'val/perplexity': self.val_perplexity,
                    'val/bpc': self.val_bpc,
                },
            )
        return loss_dict

    def _compute_sensitivity_metrics(
        self,
        loss_dict: dict[str, Any],
        batch: dict[str, Any],
        step: str,
    ) -> dict[str, torch.Tensor]:
        """Compute sensitivity metrics against generated samples.

        Ema weights will be used if enabled.
        """
        # Skip all metrics computation during sanity check
        if self.trainer.sanity_checking:
            return None

        if self.enable_sensitivity_metric and step == 'val':
            # check if sensitivity distance has been computed by the loss function already
            key = f'{step}/sensitivity'
            if key not in loss_dict:
                raise ValueError(
                    f'Sensitivity metric enabled but {key} not found in loss_dict. Ensure that the loss function is computing and returning this metric when sensitivity metric is enabled.',
                )
            # update the metric with the new value so it can be properly averaged across validation steps
            self.val_sensitivity.update(loss_dict[key])
            # replace the raw value in loss_dict with the metric instance so it gets logged/printed properly
            loss_dict[key] = self.val_sensitivity

        return loss_dict

    def _update_gradient_moment_metric_with_previous_batch(
        self,
        metric_key: str,
        generated_token_ids: torch.Tensor,
        reference_token_ids: torch.Tensor,
        reference_attention_mask: torch.Tensor | None = None,
    ) -> None:
        if not self.enable_gradient_moment_metric:
            return

        previous_batch = self._gradient_moment_prev_batches.get(metric_key)
        current_batch = {
            'generated_token_ids': generated_token_ids.detach().cpu().clone(),
            'reference_token_ids': reference_token_ids.detach().cpu().clone(),
            'reference_attention_mask': (
                reference_attention_mask.detach().cpu().clone()
                if isinstance(reference_attention_mask, torch.Tensor)
                else None
            ),
        }

        if previous_batch is None:
            self._gradient_moment_prev_batches[metric_key] = current_batch
            return

        metric = self.eval_gradient_moment[metric_key]
        if isinstance(metric, GradientMomentMetric):
            metric.update(
                generated_token_ids_a=previous_batch['generated_token_ids'],
                reference_token_ids_a=previous_batch['reference_token_ids'],
                generated_token_ids_b=current_batch['generated_token_ids'],
                reference_token_ids_b=current_batch['reference_token_ids'],
                reference_attention_mask_a=previous_batch['reference_attention_mask'],
                reference_attention_mask_b=current_batch['reference_attention_mask'],
            )

        # Use disjoint pairs: (batch1, batch2), (batch3, batch4), ...
        self._gradient_moment_prev_batches[metric_key] = None

    def _compute_generative_metrics(
        self,
        loss_dict: dict[str, Any],
        batch: dict[str, Any],
        generated_samples: SampleOutput,
        step: str,
        suffix: str = '',
    ) -> dict[str, torch.Tensor]:
        """Compute generative metrics against generated samples.

        Ema weights will be used if enabled.
        """
        # Skip all metrics computation during sanity check
        if self.trainer.sanity_checking:
            return None

        if hasattr(generated_samples, 'output_ids') and generated_samples.output_ids is not None:
            sequences = generated_samples.output_ids
        else:
            sequences = generated_samples if isinstance(generated_samples, torch.Tensor) else None

        if sequences is not None:
            # Map suffix to metric key based on current inference step configuration.
            metric_key = {v: k for k, v in self.inference_step_suffixes.items()}.get(suffix, 'default')
            attention_mask = batch.get('attention_mask')

            # Update evaluation metrics
            if self.enable_generative_perplexity:
                self.eval_generative_perplexity[metric_key].update(sequences)
                # Pass attention_mask to properly handle padding tokens (especially for left-padded sequences)
                self.real_generative_perplexity[metric_key].update(batch['input_ids'], attention_mask=attention_mask)
                loss_dict[f'eval/generative_perplexity{suffix}'] = self.eval_generative_perplexity[metric_key]
                loss_dict[f'eval/real_generative_perplexity{suffix}'] = self.real_generative_perplexity[metric_key]
            if self.enable_gradient_moment_metric:
                self._update_gradient_moment_metric_with_previous_batch(
                    metric_key=metric_key,
                    generated_token_ids=sequences,
                    reference_token_ids=batch['input_ids'],
                    reference_attention_mask=attention_mask,
                )
                loss_dict[f'eval/gradient_moment{suffix}'] = self.eval_gradient_moment[metric_key]
            if self.enable_entropy:
                self.eval_entropy[metric_key].update(sequences)
                self.real_entropy[metric_key].update(batch['input_ids'])
                loss_dict[f'eval/entropy{suffix}'] = self.eval_entropy[metric_key]
                loss_dict[f'eval/real_entropy{suffix}'] = self.real_entropy[metric_key]
            if self.enable_sliced_wasserstein:
                # Extract real data from batch
                real_data = batch['input_ids']
                self.eval_sliced_wasserstein[metric_key].update(real_data, sequences)
                loss_dict[f'eval/sliced_wasserstein{suffix}'] = self.eval_sliced_wasserstein[metric_key]
            if self.enable_token_distribution_kl:
                # Extract real data from batch
                real_data = batch['input_ids']
                self.eval_token_distribution_kl[metric_key].update(real_data, sequences)
                loss_dict[f'eval/token_distribution_kl{suffix}'] = self.eval_token_distribution_kl[metric_key]
        return loss_dict

    def _prepare_samples_for_logging(
        self,
        samples: SampleOutput,
        real_samples: torch.Tensor,
    ) -> tuple[list[str], list[str] | None, list[str]]:
        """Prepare sample data for logging.

        Args:
            samples: Generated samples.
            real_samples: Real token sequences.

        Returns:
            Tuple of (real_decoded, decoded_prompts, decoded_completions).
        """
        if hasattr(samples, 'prompts') and (samples.prompts is not None):
            decoded_prompts = samples.prompts
        else:
            decoded_prompts = None

        if hasattr(samples, 'completions'):
            decoded_completions = samples.completions
        elif hasattr(samples, 'output_ids') and samples.output_ids is not None:
            decoded_completions = [
                self.tokenizer.decode(output, skip_special_tokens=False) for output in samples.output_ids
            ]
        else:
            decoded_completions = ['<no_completions>']

        real_decoded = [self.tokenizer.decode(real, skip_special_tokens=False) for real in real_samples]

        return real_decoded, decoded_prompts, decoded_completions

    def _log_text_to_logger(
        self,
        columns: list[str],
        data: list[list[str]],
    ) -> None:
        """Log text data to the logger.

        Args:
            columns: Column names for the logged table.
            data: List of rows to log.
        """
        if hasattr(self.logger, 'log_text'):
            self.logger.log_text(
                'samples',
                columns=columns,
                data=data,
                step=self.global_step,
            )
        else:
            print_rank_zero(f'Logger {type(self.logger)} does not have log_text method')

    def _log_samples(self) -> None:
        """Log samples to the logger (rank 0 only)."""
        if not self.generated_samples_for_logging or not hasattr(self, 'tokenizer') or self.tokenizer is None:
            print_rank_zero('Skipping _log_samples - no samples or tokenizer')
            return

        samples = self.generated_samples_for_logging[0]
        real_samples = self.real_samples_for_logging[0]

        real_decoded, decoded_prompts, decoded_completions = self._prepare_samples_for_logging(
            samples,
            real_samples,
        )

        if decoded_prompts is None:
            log_data = [
                [real, completion] for real, completion in zip(real_decoded, decoded_completions, strict=False)
            ]
            self._log_text_to_logger(columns=['real', 'completion'], data=log_data)
        else:
            log_data = [
                [real, prompt, completion]
                for real, prompt, completion in zip(
                    real_decoded,
                    decoded_prompts,
                    decoded_completions,
                    strict=False,
                )
            ]
            self._log_text_to_logger(columns=['real', 'prompt', 'completion'], data=log_data)
