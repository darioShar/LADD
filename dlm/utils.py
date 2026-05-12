import glob
import importlib
import os
import re

import torch
from natsort import natsorted
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch import nn
from transformers import PreTrainedModel

def _cpu_count() -> int:
    if hasattr(os, 'sched_getaffinity'):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _auto_num_devices() -> int:
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    if torch.backends.mps.is_available():
        return 1
    return 1


OmegaConf.register_new_resolver('eval', eval, replace=True)
OmegaConf.register_new_resolver('cpu_count', _cpu_count, replace=True)
OmegaConf.register_new_resolver('auto_num_devices', _auto_num_devices, replace=True)

CHECKPOINT_STEP_PATTERN = re.compile(r'(?:^|-)step=(?P<step>\d+)(?:-v\d+)?\.ckpt$')
CHECKPOINT_EPOCH_PATTERN = re.compile(r'^epoch=(?P<epoch>\d+)-step=\d+(?:-v\d+)?\.ckpt$')


def get_rank() -> int:
    """Get the current rank, works both before and after dist initialization."""
    return int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))


def is_rank_zero() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0


def print_rank_zero(*args, **kwargs):
    """Print only on rank 0, works both before and after dist initialization."""
    if is_rank_zero():
        print(*args, **kwargs)


def hf_offline_enabled() -> bool:
    return os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1'


def resolve_local_files_only(local_files_only: bool | None = None) -> bool:
    if local_files_only is None:
        return hf_offline_enabled()
    return bool(local_files_only) or hf_offline_enabled()


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print_rank_zero(f'{model.__class__.__name__} has {total_params * 1.0e-6:.2f} M params.')
    return total_params


def load_state_dict(ckpt):
    def get_state_dict_from_lightning(path):
        pl_sd = torch.load(path, map_location='cpu', weights_only=True)
        if 'global_step' in pl_sd:
            print_rank_zero(f'Global Step: {pl_sd["global_step"]}')
        if 'ema_state_dict' in pl_sd:
            print_rank_zero('Loading EMA state dict')
            sd = pl_sd['ema_state_dict']['shadow_params']
        else:
            sd = pl_sd['state_dict']
        return sd

    print_rank_zero(f'Loading model from {ckpt}')
    if ckpt.endswith('ckpt'):
        if os.path.isdir(ckpt) and os.path.exists(
            os.path.join(ckpt, 'pytorch_model.bin'),
        ):
            sd = torch.load(os.path.join(ckpt, 'pytorch_model.bin'), map_location='cpu')
        elif os.path.isdir(ckpt):
            # convert deepspeed checkpoint to fp32 state dict
            import tempfile

            from lightning.pytorch.utilities.deepspeed import (
                convert_zero_checkpoint_to_fp32_state_dict,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                fp32_ckpt = os.path.join(tmpdir, 'pytorch_model.bin')
                convert_zero_checkpoint_to_fp32_state_dict(ckpt, fp32_ckpt)
                sd = get_state_dict_from_lightning(fp32_ckpt)
        else:
            sd = get_state_dict_from_lightning(ckpt)
    elif ckpt.endswith('safetensors'):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError
    return sd


# taken from https://github.com/Stability-AI/generative-models/blob/main/sgm/util.py
def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit('.', 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    assert 'target' in config, 'Expected key `target` to instantiate.'
    return get_obj_from_str(config['target'])(**config.get('params', dict()), **kwargs)


def instantiate_from_config_hf_pretrained(config, **kwargs):
    assert 'target' in config, 'Expected key `target` to instantiate.'
    params = dict(config.get('params', dict()))
    params['local_files_only'] = resolve_local_files_only(params.get('local_files_only'))
    return get_obj_from_str(config['target']).from_pretrained(
        **params,
        output_loading_info=True,
        **kwargs,
    )


def instantiate_model_from_config(
    config,
    not_hf_pretrained=False,
    ckpt=None,
    model_name='model',
    state_dict=None,
    peft_config=None,
    **kwargs,
) -> PreTrainedModel:
    assert 'params' in config, 'Expected key `params` in config.'
    if not_hf_pretrained:
        return instantiate_from_config(config, **kwargs)
    if 'config' in config['params']:
        # Check if config["params"]["config"] is already an instantiated object
        config_obj = config['params']['config']
        if hasattr(config_obj, '__class__') and not isinstance(config_obj, dict):
            # It's already an instantiated config object, use it directly
            model_config = config_obj
            has_pretrained_path = hasattr(config_obj, 'pretrained_model_name_or_path')
        else:
            # It's a dictionary, need to instantiate it
            if isinstance(config_obj, dict) and 'params' in config_obj:
                # Check if it has pretrained_model_name_or_path in nested params
                config_params = config_obj['params']
                if isinstance(config_params, dict):
                    has_pretrained_path = 'pretrained_model_name_or_path' in config_params
                else:
                    has_pretrained_path = hasattr(config_params, 'pretrained_model_name_or_path')
            else:
                has_pretrained_path = False

            if has_pretrained_path:
                model_config = instantiate_from_config_hf_pretrained(config_obj)
            else:
                model_config = instantiate_from_config(config_obj)

        config['params']['config'] = model_config
    if ckpt is None and 'params' in config:
        if OmegaConf.is_config(config):
            ckpt = OmegaConf.select(config, 'params.ckpt', default=None)
            if ckpt is not None:
                # Create a copy without the ckpt field to avoid modifying the original
                config_params = OmegaConf.to_container(config['params'], resolve=True)
                if 'ckpt' in config_params:
                    del config_params['ckpt']
                config['params'] = OmegaConf.create(config_params)
        else:
            ckpt = config['params'].pop('ckpt', None)

    if 'pretrained_model_name_or_path' in config['params']:
        if 'torch_dtype' in config['params']:
            if config['params']['torch_dtype'] != 'auto':
                config['params']['torch_dtype'] = get_obj_from_str(
                    config['params']['torch_dtype'],
                )
        model: PreTrainedModel = instantiate_from_config_hf_pretrained(config, **kwargs)
    else:
        model: PreTrainedModel = instantiate_from_config(config, **kwargs)

    if peft_config is not None:
        model.add_adapter(instantiate_from_config(peft_config))
    if ckpt is not None:
        state_dict = load_state_dict(ckpt)
        state_dict = {k.replace(f'{model_name}.', '', 1): v for k, v in state_dict.items()}
        # config["params"]["state_dict"] = sd
    if state_dict is not None:
        model.load_state_dict(state_dict)
        print_rank_zero('loaded state dict')
    return model


def instantiate_optimizer_from_config(config, parameters):
    assert 'target' in config, 'Expected key `target` to instantiate.'
    return get_obj_from_str(config['target'])(
        parameters,
        **config.get('params', dict()),
    )


def load_model_from_config(config, ckpt):
    print_rank_zero(f'Loading model from {ckpt}')
    model = get_obj_from_str(config['target']).load_from_checkpoint(
        ckpt,
        **config['params'],
    )
    model.eval()
    return model


# copied from https://github.com/Stability-AI/generative-models/blob/main/main.py#L203
def _parse_checkpoint_step(ckpt_path: str) -> int | None:
    match = CHECKPOINT_STEP_PATTERN.search(os.path.basename(ckpt_path))
    if match is None:
        return None
    return int(match.group('step'))


def _parse_checkpoint_epoch(ckpt_path: str) -> int | None:
    match = CHECKPOINT_EPOCH_PATTERN.search(os.path.basename(ckpt_path))
    if match is None:
        return None
    return int(match.group('epoch'))


def _select_checkpoint_by_field(
    logdir: str,
    *,
    field_name: str,
    field_value: int,
    parser,
):
    ckpt = os.path.join(logdir, 'checkpoints', 'epoch*.ckpt')
    print_rank_zero(f'Searching for {field_name}-specific checkpoints in {ckpt}')
    epoch_ckpts = natsorted(glob.glob(ckpt))
    matching_ckpts = [path for path in epoch_ckpts if parser(path) == field_value]

    if not matching_ckpts:
        available_values = [parsed for path in epoch_ckpts if (parsed := parser(path)) is not None]
        available_values_str = ', '.join(str(value) for value in available_values[:20])
        if len(available_values) > 20:
            available_values_str += ', ...'
        raise ValueError(
            f'Could not find a checkpoint with {field_name}={field_value} in {os.path.join(logdir, "checkpoints")}. '
            f'Available saved {field_name}s: {available_values_str or "none"}.',
        )

    if len(matching_ckpts) > 1:
        print_rank_zero(
            f'Found {len(matching_ckpts)} checkpoints for {field_name}={field_value}; picking the newest by mtime.',
        )
        ckpt = sorted(matching_ckpts, key=lambda path: os.path.getmtime(path))[-1]
    else:
        ckpt = matching_ckpts[0]

    melk_ckpt_name = 'last-v1.ckpt'
    print_rank_zero(f'Selected checkpoint for requested {field_name}={field_value}: {ckpt}')
    print_rank_zero(f'Current melk ckpt name: {melk_ckpt_name}')
    return ckpt, melk_ckpt_name


def get_checkpoint_name(logdir, step: int | None = None, epoch: int | None = None):
    if step is not None and epoch is not None:
        raise ValueError('Specify at most one of step or epoch when selecting a checkpoint.')

    if step is not None:
        return _select_checkpoint_by_field(logdir, field_name='step', field_value=step, parser=_parse_checkpoint_step)

    if epoch is not None:
        return _select_checkpoint_by_field(logdir, field_name='epoch', field_value=epoch, parser=_parse_checkpoint_epoch)

    ckpt = os.path.join(logdir, 'checkpoints', 'last**.ckpt')
    print_rank_zero(f'Searching for checkpoints in {ckpt}')
    ckpt = natsorted(glob.glob(ckpt))
    print_rank_zero('available "last" checkpoints:')
    print_rank_zero(ckpt)
    if len(ckpt) > 1:
        print_rank_zero('got most recent checkpoint')
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print_rank_zero(f'Most recent ckpt is {ckpt}')
        with open(os.path.join(logdir, 'most_recent_ckpt.txt'), 'w') as f:
            f.write(ckpt + '\n')
        try:
            version = int(ckpt.split('/')[-1].split('-v')[-1].split('.')[0])
        except Exception as e:
            print_rank_zero('version confusion but not bad')
            print_rank_zero(e)
            version = 1
        # version = last_version + 1
    elif len(ckpt) == 0:
        ckpt = os.path.join(logdir, 'checkpoints', 'epoch**.ckpt')
        ckpt = natsorted(glob.glob(ckpt))
        ckpt = ckpt[-1]
        version = 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f'last-v{version}.ckpt'
    print_rank_zero(f'Current melk ckpt name: {melk_ckpt_name}')
    return ckpt, melk_ckpt_name


def get_optimizer_params(
    model: nn.Module,
    loss_fn: nn.Module | None = None,
    ignore_parameters: list[str] | None = None,
) -> list[torch.nn.Parameter]:
    """Get the optimizer parameters for a model.

    Args:
        model: The model to get the parameters for.
        loss_fn: The loss function to get the parameters for.
        ignore_parameters: The parameters to ignore.

    """

    def not_ignore_parameters(name: str) -> bool:
        if ignore_parameters is None:
            return True
        return any(ignore_n not in name for ignore_n in ignore_parameters)

    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    params = [p for n, p in model.named_parameters() if p.requires_grad and not_ignore_parameters(n)]
    if loss_fn is not None:
        params += [p for n, p in loss_fn.named_parameters() if p.requires_grad and not_ignore_parameters(n)]
    return params


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: The PyTorch model to count parameters for.

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
    }


def format_param_count(count: int) -> str:
    """Format parameter count in human-readable form.

    Args:
        count: Number of parameters.

    Returns:
        Formatted string (e.g., '12.3M', '1.5B').
    """
    if count >= 1e9:
        return f'{count / 1e9:.2f}B'
    elif count >= 1e6:
        return f'{count / 1e6:.2f}M'
    elif count >= 1e3:
        return f'{count / 1e3:.2f}K'
    else:
        return str(count)


def log_model_info(model: nn.Module, name: str = 'Model', print_fn=print) -> None:
    """Log comprehensive information about a model.

    Args:
        model: The PyTorch model.
        name: Name of the model for logging.
        print_fn: Function to use for printing (default: print).
    """
    params = count_model_parameters(model)
    mode = 'Training' if model.training else 'Eval'

    # Get device and dtype if model has parameters
    try:
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype
    except StopIteration:
        device = 'N/A'
        dtype = 'N/A'

    print_fn('-' * 80)
    print_fn(f'{name} Information:')
    print_fn(f'  Total parameters:      {format_param_count(params["total"]):>10} ({params["total"]:,})')
    print_fn(f'  Trainable parameters:  {format_param_count(params["trainable"]):>10} ({params["trainable"]:,})')
    if params['frozen'] > 0:
        print_fn(f'  Frozen parameters:     {format_param_count(params["frozen"]):>10} ({params["frozen"]:,})')
    print_fn(f'  Mode: {mode}')
    print_fn(f'  Device: {device}')
    print_fn(f'  Dtype: {dtype}')
    print_fn('-' * 80)


def log_lightning_module_info(pl_module, print_fn=print) -> None:
    """Log information about all models in a Lightning module.

    Args:
        pl_module: Lightning module containing models.
        print_fn: Function to use for printing (default: print).
    """
    print_fn('')
    print_fn('=' * 80)
    print_fn('MODEL INFORMATION')
    print_fn('=' * 80)

    # Track if we found any models
    found_models = False

    # Log main model if it exists
    if hasattr(pl_module, 'model'):
        log_model_info(pl_module.model, 'Main Model', print_fn)
        found_models = True

    # Log encoder if it exists
    if hasattr(pl_module, 'encoder'):
        log_model_info(pl_module.encoder, 'Encoder', print_fn)
        found_models = True

    # Log decoder if it exists
    if hasattr(pl_module, 'decoder'):
        log_model_info(pl_module.decoder, 'Decoder', print_fn)
        found_models = True

    # Log denoiser if it exists
    if hasattr(pl_module, 'denoiser'):
        log_model_info(pl_module.denoiser, 'Denoiser', print_fn)
        found_models = True

    # Log loss function if it's a module with parameters
    if hasattr(pl_module, 'loss_fn') and isinstance(pl_module.loss_fn, nn.Module):
        loss_params = count_model_parameters(pl_module.loss_fn)
        if loss_params['total'] > 0:
            log_model_info(pl_module.loss_fn, 'Loss Module', print_fn)
            found_models = True

    # Log EMA information if enabled
    if hasattr(pl_module, 'hparams') and pl_module.hparams.get('use_ema', False):
        print_fn('-' * 80)
        print_fn('EMA (Exponential Moving Average):')
        print_fn(f'  Enabled: Yes')
        print_fn(f'  Decay: {pl_module.hparams.get("ema_decay", "N/A")}')
        if hasattr(pl_module, 'ema_models'):
            print_fn(f'  EMA models: {list(pl_module.ema_models.keys())}')
        print_fn('-' * 80)

    # Log total module parameters
    total_params = count_model_parameters(pl_module)
    print_fn('')
    print_fn('Total Lightning Module:')
    print_fn(f'  Total parameters:      {format_param_count(total_params["total"]):>10} ({total_params["total"]:,})')
    print_fn(f'  Trainable parameters:  {format_param_count(total_params["trainable"]):>10} ({total_params["trainable"]:,})')
    if total_params['frozen'] > 0:
        print_fn(f'  Frozen parameters:     {format_param_count(total_params["frozen"]):>10} ({total_params["frozen"]:,})')

    print_fn('=' * 80)
    print_fn('')

    if not found_models:
        print_fn('Warning: No models found in Lightning module')
