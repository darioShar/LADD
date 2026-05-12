import contextlib
import copy
import inspect
import warnings
from collections.abc import Iterable
from typing import Any

import torch
import transformers
from packaging import version

if transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
    import deepspeed


def deprecate(
    *args,
    take_from: dict | Any | None = None,
    standard_warn=True,
    stacklevel=2,
):
    from .. import __version__

    deprecated_kwargs = take_from
    values = ()
    if not isinstance(args[0], tuple):
        args = (args,)

    for attribute, version_name, message in args:
        if version.parse(version.parse(__version__).base_version) >= version.parse(
            version_name,
        ):
            raise ValueError(
                f"The deprecation tuple {(attribute, version_name, message)} should be removed since diffusers'"
                f' version {__version__} is >= {version_name}',
            )

        warning = None
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            values += (deprecated_kwargs.pop(attribute),)
            warning = f'The `{attribute}` argument is deprecated and will be removed in version {version_name}.'
        elif hasattr(deprecated_kwargs, attribute):
            values += (getattr(deprecated_kwargs, attribute),)
            warning = f'The `{attribute}` attribute is deprecated and will be removed in version {version_name}.'
        elif deprecated_kwargs is None:
            warning = f'`{attribute}` is deprecated and will be removed in version {version_name}.'

        if warning is not None:
            warning = warning + ' ' if standard_warn else ''
            warnings.warn(warning + message, FutureWarning, stacklevel=stacklevel)

    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        key, value = next(iter(deprecated_kwargs.items()))
        raise TypeError(
            f'{function} in {filename} line {line_number - 1} got an unexpected keyword argument `{key}`',
        )

    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return values


# copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L305
class EMAModel:
    """Exponential Moving Average of models weights"""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        foreach: bool = True,
        model_cls: Any | None = None,
        model_config: dict[str, Any] = None,
        **kwargs,
    ):
        """Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).

        """
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                'Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. '
                'Please pass the parameters of the module instead.'
            )
            deprecate(
                'passing a `torch.nn.Module` to `ExponentialMovingAverage`',
                '1.0.0',
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            use_ema_warmup = True

        if kwargs.get('max_value') is not None:
            deprecation_message = 'The `max_value` argument is deprecated. Please use `decay` instead.'
            deprecate('max_value', '1.0.0', deprecation_message, standard_warn=False)
            decay = kwargs['max_value']

        if kwargs.get('min_value') is not None:
            deprecation_message = 'The `min_value` argument is deprecated. Please use `min_decay` instead.'
            deprecate('min_value', '1.0.0', deprecation_message, standard_warn=False)
            min_decay = kwargs['min_value']

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get('device') is not None:
            deprecation_message = 'The `device` argument is deprecated. Please use `to` instead.'
            deprecate('device', '1.0.0', deprecation_message, standard_warn=False)
            self.to(device=kwargs['device'])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, path, model_cls, foreach=False) -> 'EMAModel':
        _, ema_kwargs = model_cls.from_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(
            model.parameters(),
            model_cls=model_cls,
            model_config=model.config,
            foreach=foreach,
        )

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError(
                '`save_pretrained` can only be used if `model_cls` was defined at __init__.',
            )

        if self.model_config is None:
            raise ValueError(
                '`save_pretrained` can only be used if `model_config` was defined at __init__.',
            )

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop('shadow_params', None)

        model.register_to_config(**state_dict)
        self.copy_to(model.parameters())
        model.save_pretrained(path)

    def get_decay(self, optimization_step: int) -> float:
        """Compute the decay factor for the exponential moving average."""
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext()

        if self.foreach:
            if transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                context_manager = deepspeed.zero.GatheredParameters(
                    parameters,
                    modifier_rank=None,
                )

            with context_manager:
                params_grad = [param for param in parameters if param.requires_grad]
                s_params_grad = [
                    s_param
                    for s_param, param in zip(
                        self.shadow_params,
                        parameters,
                        strict=True,  # strict=False,
                    )
                    if param.requires_grad
                ]

                if len(params_grad) < len(parameters):
                    torch._foreach_copy_(
                        [
                            s_param
                            for s_param, param in zip(
                                self.shadow_params,
                                parameters,
                                strict=True,
                            )
                            if not param.requires_grad
                        ],
                        [param for param in parameters if not param.requires_grad],
                        non_blocking=True,
                    )

                torch._foreach_sub_(
                    s_params_grad,
                    torch._foreach_sub(s_params_grad, params_grad),
                    alpha=one_minus_decay,
                )

        else:
            for s_param, param in zip(self.shadow_params, parameters, strict=True):
                if transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                    context_manager = deepspeed.zero.GatheredParameters(
                        param,
                        modifier_rank=None,
                    )

                with context_manager:
                    if param.requires_grad:
                        s_param.sub_(one_minus_decay * (s_param - param))
                    else:
                        s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.

        """
        parameters = list(parameters)
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters],
                [
                    s_param.to(param.device).data
                    for s_param, param in zip(
                        self.shadow_params,
                        parameters,
                        strict=True,
                    )
                ],
            )
        else:
            for s_param, param in zip(self.shadow_params, parameters, strict=True):
                param.data.copy_(s_param.to(param.device).data)

    def pin_memory(self) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """
        self.shadow_params = [p.pin_memory() for p in self.shadow_params]

    def to(self, device=None, dtype=None, non_blocking=False) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`

        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if p.is_floating_point()
            else p.to(device=device, non_blocking=non_blocking)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            'decay': self.decay,
            'min_decay': self.min_decay,
            'optimization_step': self.optimization_step,
            'update_after_step': self.update_after_step,
            'use_ema_warmup': self.use_ema_warmup,
            'inv_gamma': self.inv_gamma,
            'power': self.power,
            'shadow_params': self.shadow_params,
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""Saves the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`. The parameters to be temporarily stored.

        """
        self.temp_stored_params = [param.detach().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters
        without: affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.

        """
        if self.temp_stored_params is None:
            raise RuntimeError(
                'This ExponentialMovingAverage has no `store()`ed weights to `restore()`',
            )
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters],
                [c_param.data for c_param in self.temp_stored_params],
            )
        else:
            for c_param, param in zip(
                self.temp_stored_params,
                parameters,
                strict=True,
            ):
                param.data.copy_(c_param.data)

        # Better memory-wise.
        # self.temp_stored_params = None
        # --- START of fix ---
        # Explicitly delete the large list of tensors
        # del self.temp_stored_params
        self.temp_stored_params = None

        # Run Python's garbage collector
        # gc.collect()

        # Clear PyTorch's CUDA memory cache to combat fragmentation
        # if torch.cuda.is_available():
        # torch.cuda.empty_cache()
        # --- END of fix ---

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        r"""Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.

        """
        assert strict, (
            'Non-strict loading has not been completely implemented yet, \
                should use named parameters rather than a list for shadow params'
        )
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get('decay', self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.min_decay = state_dict.get('min_decay', self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError('Invalid min_decay')

        self.optimization_step = state_dict.get(
            'optimization_step',
            self.optimization_step,
        )
        if not isinstance(self.optimization_step, int):
            raise ValueError('Invalid optimization_step')

        self.update_after_step = state_dict.get(
            'update_after_step',
            self.update_after_step,
        )
        if not isinstance(self.update_after_step, int):
            raise ValueError('Invalid update_after_step')

        self.use_ema_warmup = state_dict.get('use_ema_warmup', self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError('Invalid use_ema_warmup')

        self.inv_gamma = state_dict.get('inv_gamma', self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError('Invalid inv_gamma')

        self.power = state_dict.get('power', self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError('Invalid power')

        # 2. Intelligent, non-strict loading for shadow_params
        source_params = state_dict.get('shadow_params', [])
        target_params = self.shadow_params

        missing_keys = []
        unexpected_keys = []
        mismatched_shape_keys = []

        # Check for parameters in the current model not in the checkpoint
        if len(target_params) > len(source_params):
            missing_keys = list(range(len(source_params), len(target_params)))

        # Check for parameters in the checkpoint not in the current model
        if len(source_params) > len(target_params):
            unexpected_keys = list(range(len(target_params), len(source_params)))

        # Iterate through the parameters that exist in both and check shapes
        for i, (target_p, source_p) in enumerate(zip(target_params, source_params, strict=False)):
            if target_p.shape != source_p.shape:
                mismatched_shape_keys.append(i)
                continue  # Skip loading this parameter

            # If shapes match, copy the data
            target_p.data.copy_(source_p.to(target_p.device).data)

        # 3. Report or raise errors based on strict mode
        if strict:
            error_msgs = []
            if missing_keys:
                error_msgs.append(f'Missing keys (parameter indices) in state_dict: {missing_keys}')
            if unexpected_keys:
                error_msgs.append(f'Unexpected keys (parameter indices) in state_dict: {unexpected_keys}')
            if mismatched_shape_keys:
                shapes = [f'index {i} ({target_params[i].shape} vs {source_params[i].shape})' for i in mismatched_shape_keys]
                error_msgs.append(f'Size Mismatches for parameter indices: {", ".join(shapes)}')

            if error_msgs:
                raise RuntimeError('Error(s) in loading state_dict for EMAModel:\n\t' + '\n\t'.join(error_msgs))
        # In non-strict mode, just print a warning for informational purposes
        elif missing_keys or unexpected_keys or mismatched_shape_keys:
            print('Non-strict loading for EMAModel:')
            if missing_keys:
                print(f'  - Skipped loading for {len(missing_keys)} new parameters.')
            if unexpected_keys:
                print(f'  - Ignored {len(unexpected_keys)} parameters from checkpoint not in current model.')
            if mismatched_shape_keys:
                print(f'  - Skipped loading for {len(mismatched_shape_keys)} parameters due to shape mismatch.')

        # Simple check after loading
        if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
            raise ValueError('shadow_params must all be Tensors')
