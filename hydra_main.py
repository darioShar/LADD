# main.py
import datetime
import errno
import multiprocessing.util as mp_util
import os
import re
import sys
import tempfile
from pathlib import Path


def _set_hf_cache_env(cache_dir: str, override: bool) -> str:
    cache_root = os.path.abspath(os.path.expanduser(cache_dir))
    hub_cache = os.path.join(cache_root, 'hub')
    datasets_cache = os.path.join(cache_root, 'datasets')
    env_vars = {
        'HF_HOME': cache_root,
        'HF_HUB_CACHE': hub_cache,
        'HF_DATASETS_CACHE': datasets_cache,
        'TRANSFORMERS_CACHE': hub_cache,
    }
    for key, value in env_vars.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return cache_root

# Configure HF cache env before importing libraries that read these variables at import time.
_initial_hf_cache_dir = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE_DIR') or './.hf_cache'
_set_hf_cache_env(_initial_hf_cache_dir, override=False)

import hydra
import lightning as L
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.distributed import _init_dist_connection
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from dlm.lightning.fsdp import fsdp_huggingface
from dlm.utils import get_checkpoint_name, instantiate_from_config

if torch.backends.mps.is_available():
    torch.multiprocessing.set_sharing_strategy('file_system')

# Use an absolute temp root so torch.compile/inductor subprocesses do not depend on cwd.
tmp_root = os.path.abspath('.tmp')
local_tmp = os.path.join(
    tmp_root,
    f'j{os.environ.get("SLURM_JOB_ID", "0")}_r{os.environ.get("RANK", "0")}',
)
Path(local_tmp).mkdir(parents=True, exist_ok=True)

for key in ('TMPDIR', 'TEMP', 'TMP'):
    os.environ[key] = local_tmp
tempfile.tempdir = local_tmp

compile_cache_root = os.path.join(local_tmp, 'compile_cache')
for key, dirname in (
    ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
    ('TRITON_CACHE_DIR', 'triton'),
):
    os.environ.setdefault(key, os.path.abspath(os.path.join(compile_cache_root, dirname)))
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

def _patch_multiprocessing_temp_cleanup() -> None:
    original_remove = mp_util._remove_temp_dir

    def _remove_temp_dir(rmtree, tempdir: str) -> None:
        try:
            original_remove(rmtree, tempdir)
        except OSError as exc:
            if exc.errno == errno.EBUSY:
                return
            raise

    mp_util._remove_temp_dir = _remove_temp_dir  # type: ignore[assignment]

_patch_multiprocessing_temp_cleanup()

# env — same as before
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid deadlock
os.environ['HYDRA_FULL_ERROR'] = '1'
# HF_HUB_DISABLE_XET=1
os.environ['HF_HUB_DISABLE_XET'] = '1'

def configure_hf_cache(cache_dir: str | None) -> str | None:
    if cache_dir is None:
        return None
    cache_root = _set_hf_cache_env(cache_dir, override=True)
    hub_cache = os.environ['HF_HUB_CACHE']
    datasets_cache = os.environ['HF_DATASETS_CACHE']
    # Create directories if they don't exist.
    Path(hub_cache).mkdir(parents=True, exist_ok=True)
    Path(datasets_cache).mkdir(parents=True, exist_ok=True)
    return cache_root


def configure_hf_offline(enabled: bool) -> None:
    if not enabled:
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'


def resolve_hf_cache_dir(config_cache_dir: str | None) -> str | None:
    env_cache_dir = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE_DIR')
    if config_cache_dir is None:
        return env_cache_dir
    if env_cache_dir and config_cache_dir == './.hf_cache':
        return env_cache_dir
    return config_cache_dir


cache_dir = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE_DIR') or './.hf_cache'

def print_rank_zero(*args, **kwargs):
    """Print only on rank 0, works both before and after dist initialization."""
    # Check env var first (works before dist.init), then check dist if initialized
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    if rank == 0:
        print(*args, **kwargs)


def _maybe_get_slurm_environment() -> SLURMEnvironment | None:
    required_keys = ('SLURM_PROCID', 'SLURM_NTASKS')
    if not all(key in os.environ for key in required_keys):
        return None
    try:
        return SLURMEnvironment()
    except Exception as exc:  # pragma: no cover - best effort logging
        print_rank_zero(f'Warning: failed to initialize SLURM environment: {exc}')
        return None


def _infer_local_rank() -> int:
    for env_key in ('LOCAL_RANK', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        value = os.environ.get(env_key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def _running_under_torchrun() -> bool:
    if os.environ.get('TORCHELASTIC_RUN_ID'):
        return True
    required_keys = ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')
    return all(key in os.environ for key in required_keys)


def _auto_accelerator() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _num_trainer_devices(cfg: DictConfig) -> int:
    devices = cfg.lightning.trainer.get('devices', None)
    if devices is None or devices == 'auto':
        if cfg.get('num_gpu_devices') is not None:
            return int(cfg.num_gpu_devices)
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
        return 1
    if isinstance(devices, (list, tuple)):
        return len(devices)
    return int(devices)


def _configure_runtime_hardware(cfg: DictConfig) -> tuple[str, int, int, bool]:
    trainer_cfg = cfg.lightning.trainer

    requested_accelerator = trainer_cfg.get('accelerator', 'auto')
    accelerator = _auto_accelerator() if requested_accelerator in (None, 'auto') else str(requested_accelerator)
    trainer_cfg.accelerator = accelerator

    if cfg.get('num_gpu_devices') is None:
        cfg.num_gpu_devices = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    num_devices = _num_trainer_devices(cfg)
    num_nodes = int(cfg.get('num_nodes', trainer_cfg.get('num_nodes', 1)))
    distributed = accelerator != 'mps' and (num_devices * num_nodes) > 1


    return accelerator, num_devices, num_nodes, distributed


def remap_checkpoint_paths(checkpoint_path: str):
    """
    Load checkpoint and remap old module paths to new ones.
    Returns (path_to_remapped_ckpt, did_remap: bool).
    """
    path_mappings = {
        'dlm.modules.transformers': 'dlm.transformers',
    }

    print_rank_zero(f'Loading checkpoint for path remapping: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    needs_remapping = False

    def remap_recursive(obj):
        nonlocal needs_remapping
        if isinstance(obj, dict):
            return {k: remap_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [remap_recursive(i) for i in obj]
        if isinstance(obj, str):
            for old_path, new_path in path_mappings.items():
                if old_path in obj:
                    needs_remapping = True
                    obj = obj.replace(old_path, new_path)
                    print_rank_zero(f'Remapped: {old_path} -> {new_path}')
            return obj
        return obj

    remapped = remap_recursive(checkpoint)

    if not needs_remapping:
        print_rank_zero('No path remapping needed')
        return checkpoint_path, False

    with tempfile.NamedTemporaryFile(
        prefix='.tmp_remapped_ckpt_',
        suffix='.ckpt',
        delete=False,
        dir='.',  # current working directory
    ) as tmp:
        torch.save(remapped, tmp.name)  # write via path to be safe on all OSes

    print_rank_zero(f'Created remapped checkpoint: {tmp.name}')
    return tmp.name, True


def _jst_now_string():
    # Keep the exact timestamp style you used before, in JST
    tz = datetime.timezone(datetime.timedelta(hours=9), name='JST')
    return datetime.datetime.now(tz=tz).strftime('%Y-%m-%dT%H-%M-%S')


def _build_nowname(cfg: DictConfig, now: str) -> str:
    """
    Reproduce your previous naming policy:
    now + "_" + tag + postfix, where tag is derived from config selection.
    We use Hydra choices as a surrogate for "last base config path".
    If you want exact legacy tags, set cfg.naming.override_tag explicitly.
    """
    choices = HydraConfig.get().runtime.choices if HydraConfig.initialized() else {}
    if cfg.get('naming') and cfg.naming.get('override_tag'):
        tag = cfg.naming.override_tag
    else:
        # Similar to taking dataset/model from your previous 'configs/...' path
        parts = []
        if 'data' in choices:
            parts.append(choices['data'].split('/')[-1])
        if 'model' in choices:
            parts.append(choices['model'].split('/')[-1])
        tag = '_'.join(parts) if parts else 'run'
    name = f'{now}_{tag}{cfg.postfix or ""}'
    return name.lstrip('_')


def _make_log_structure(cfg: DictConfig, now: str, keep_existing: bool, resume_logdir: Path | None):
    """
    Build logdir/ckptdir/cfgdir and run name, preserving your layout.
    - If keep_existing=True (resume without fork), reuse resume_logdir and its name.
    - Else, create a fresh nowname in cfg.paths.log_root.
    """
    base_root = Path(get_original_cwd()).resolve()
    if keep_existing and resume_logdir is not None:
        logdir = Path(resume_logdir).resolve()
        nowname = logdir.name
    else:
        nowname = _build_nowname(cfg, now)
        log_root = Path(cfg.paths.log_root)
        if not log_root.is_absolute():
            log_root = (base_root / log_root).resolve()
        else:
            log_root = log_root.resolve()
        logdir = (log_root / nowname).resolve()

    ckptdir = logdir / 'checkpoints'
    cfgdir = logdir / 'configs'
    return logdir, nowname, ckptdir, cfgdir


RUN_DIR_TIMESTAMP = re.compile(r'^(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})(?:_|$)')


def _logdir_has_checkpoints(logdir: Path) -> bool:
    ckpt_dir = logdir / 'checkpoints'
    if not ckpt_dir.is_dir():
        return False
    return any(ckpt_dir.glob('last*.ckpt')) or any(ckpt_dir.glob('epoch*.ckpt'))


def _select_latest_timestamped_logdir(parent: Path) -> Path | None:
    candidates: list[tuple[datetime.datetime, Path]] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        match = RUN_DIR_TIMESTAMP.match(child.name)
        if match is None:
            continue
        timestamp = datetime.datetime.strptime(match.group('stamp'), '%Y-%m-%dT%H-%M-%S')
        candidates.append((timestamp, child))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1].resolve()


def _resolve_resume_dir(path: Path) -> Path:
    if _logdir_has_checkpoints(path):
        return path.resolve()

    latest_logdir = _select_latest_timestamped_logdir(path)
    if latest_logdir is not None:
        print_rank_zero(f'Resolved resume.from_dir="{path}" to latest timestamped run "{latest_logdir}"')
        return latest_logdir

    raise ValueError(
        f'Cannot resolve resume.from_dir={path}. Expected either a run directory with checkpoints/ '
        f'or a parent directory containing timestamped run subdirectories.',
    )


def _resolve_resume(cfg: DictConfig):
    """
    Unify resume options. Supports:
      - cfg.resume.from_dir: lightning logdir OR a .ckpt file path
      - cfg.resume.ckpt_path: a single .ckpt file path
      - cfg.resume.ckpt_steps: exact training step to load from a resolved logdir
      - cfg.resume.ckpt_epochs: exact epoch to load from a resolved logdir
    Returns (resume_ckpt, resume_logdir, melk_ckpt_name).
    """
    resume_path = cfg.resume.from_dir
    single_ckpt = cfg.resume.ckpt_path
    resume_step = cfg.resume.get('ckpt_steps')
    resume_epoch = cfg.resume.get('ckpt_epochs')

    if resume_step is not None and resume_step < 0:
        raise ValueError(f'resume.ckpt_steps must be non-negative, got {resume_step}')
    if resume_epoch is not None and resume_epoch < 0:
        raise ValueError(f'resume.ckpt_epochs must be non-negative, got {resume_epoch}')
    if resume_step is not None and resume_epoch is not None:
        raise ValueError('Use at most one of resume.ckpt_steps or resume.ckpt_epochs.')

    if single_ckpt:
        if resume_step is not None or resume_epoch is not None:
            raise ValueError(
                'resume.ckpt_steps and resume.ckpt_epochs cannot be combined with resume.ckpt_path. '
                'Use only one explicit selector.',
            )
        # explicit single checkpoint
        ckpt_path = Path(single_ckpt).expanduser().resolve()
        if not ckpt_path.exists():
            raise ValueError(f'Cannot find checkpoint: {single_ckpt}')
        return str(ckpt_path), None, None

    if resume_path:
        p = Path(resume_path).expanduser().resolve()
        if not p.exists():
            raise ValueError(f'Cannot find {resume_path}')
        if p.is_file() or str(p).rstrip('/').endswith('.ckpt'):
            if resume_step is not None or resume_epoch is not None:
                raise ValueError(
                    'resume.ckpt_steps and resume.ckpt_epochs require resume.from_dir to point to a run directory '
                    'or parent directory, '
                    'not a single checkpoint file.',
                )
            # path is a checkpoint file; logdir is its parent.parent
            parts = str(p).split('/')
            logdir = Path('/'.join(parts[:-2])).resolve()
            ckpt = str(p)
            _, melk_name = get_checkpoint_name(str(logdir))
        else:
            # path is either a logdir or a parent containing timestamped logdirs
            logdir = _resolve_resume_dir(p)
            ckpt, melk_name = get_checkpoint_name(str(logdir), step=resume_step, epoch=resume_epoch)
        print_rank_zero('#' * 100)
        print_rank_zero(f'Resuming from checkpoint "{ckpt}"')
        print_rank_zero('#' * 100)
        return ckpt, logdir, melk_name

    # No resume
    return None, None, None


@hydra.main(version_base='1.3', config_path='conf', config_name='config')
def main(cfg: DictConfig):
    # ============================================================================
    # INITIALIZATION
    # ============================================================================

    hf_cache_dir = resolve_hf_cache_dir(cfg.get('hf_cache_dir', cache_dir))
    configure_hf_cache(hf_cache_dir)
    configure_hf_offline(bool(cfg.get('hf_offline', False)))

    # Torch settings & reproducibility
    if cfg.get('seed') is not None:
        L.seed_everything(cfg.seed, workers=True)
        print_rank_zero(f'🎲 Random seed set to: {cfg.seed}')
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)

    if cfg.get('torch_compile') is not None:
        try:
            import torch._dynamo as dynamo

            dynamo.config.optimize_ddp = False
            print_rank_zero('torch.compile enabled: disabling Dynamo DDP optimizer to avoid higher-order op errors.')
            try:
                import torch._inductor as inductor

                if hasattr(inductor, 'config'):
                    if hasattr(inductor.config, 'triton') and hasattr(inductor.config.triton, 'cudagraphs'):
                        inductor.config.triton.cudagraphs = False
                    if hasattr(inductor.config, 'use_cuda_graphs'):
                        inductor.config.use_cuda_graphs = False
                    if hasattr(inductor.config, 'force_disable_cudagraphs'):
                        inductor.config.force_disable_cudagraphs = True
                print_rank_zero('torch.compile enabled: disabling Inductor cudagraphs to avoid overwritten outputs.')
            except Exception as exc:  # pragma: no cover - best effort logging
                print_rank_zero(f'Warning: failed to update torch._inductor config: {exc}')
        except Exception as exc:  # pragma: no cover - best effort logging
            print_rank_zero(f'Warning: failed to update torch._dynamo config: {exc}')

    # ============================================================================
    # DISTRIBUTED SETUP
    # ============================================================================

    slurm_env = _maybe_get_slurm_environment()
    is_torchrun = _running_under_torchrun()
    use_slurm_env = slurm_env is not None and not is_torchrun
    if use_slurm_env:
        os.environ.setdefault('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0'))
        os.environ.setdefault('RANK', os.environ.get('SLURM_PROCID', '0'))
        os.environ.setdefault('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1'))

    # Only manually init distributed for SLURM (not for torchrun)
    # torchrun handles its own distributed initialization via rendezvous
    if use_slurm_env and not dist.is_initialized() and slurm_env.world_size() > 1:
        _init_dist_connection(slurm_env, 'nccl')
    # Note: Removed manual dist.init_process_group() for torchrun case
    # Lightning's DDPStrategy will handle initialization when using torchrun

    # === Add this near the top of main(), before init_process_group ===
    local_rank = _infer_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    accelerator, num_devices, num_nodes, distributed = _configure_runtime_hardware(cfg)
    if not distributed:
        print_rank_zero(
            f'Using accelerator={accelerator}, devices={num_devices}, num_nodes={num_nodes}; '
            'Lightning strategy and VQ DDP sync are disabled.',
        )
    else:
        print_rank_zero(f'Using accelerator={accelerator}, devices={num_devices}, num_nodes={num_nodes}.')

    # Warn about MPS slow startup
    if torch.backends.mps.is_available() and accelerator == 'mps':
        print_rank_zero('')
        print_rank_zero('=' * 80)
        print_rank_zero('⚠️  MPS PERFORMANCE NOTE')
        print_rank_zero('=' * 80)
        print_rank_zero('You are using MPS (Metal Performance Shaders) on Mac.')
        print_rank_zero('The first few iterations will be SLOW due to:')
        print_rank_zero('  1. MPS kernel compilation and caching')
        print_rank_zero('  2. PyTorch graph optimization')
        print_rank_zero('  3. Metal shader compilation')
        print_rank_zero('')
        print_rank_zero('This is normal behavior. Performance will improve after ~5-10 iterations.')
        print_rank_zero('Subsequent runs will be faster as kernels are cached.')
        print_rank_zero('=' * 80)
        print_rank_zero('')


    # ============================================================================
    # RUN CONFIGURATION
    # ============================================================================

    # Generate consistent timestamp across all ranks
    # For torchrun, use RANK env var since dist may not be initialized yet
    local_rank_for_print = int(os.environ.get('RANK', '0'))
    now_list = [None]
    if local_rank_for_print == 0:
        now = _jst_now_string()
        now_list[0] = now

    # Only broadcast if dist is already initialized (SLURM case)
    # For torchrun, each rank generates its own timestamp which is fine for logging
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast_object_list(now_list, src=0)

    now = now_list[0] if now_list[0] is not None else _jst_now_string()

    # Print run header (only on rank 0)
    if local_rank_for_print == 0:
        print_rank_zero('')
        print_rank_zero('=' * 80)
        print_rank_zero(f'🚀 Starting {cfg.mode.upper()} run: {cfg.projectname or "Experiment"}')
        print_rank_zero('=' * 80)
        print_rank_zero(f'Timestamp: {now}')
        print_rank_zero(f'Mode: {cfg.mode}')
        print_rank_zero(f'Debug: {cfg.debug}')
        if cfg.get('seed') is not None:
            print_rank_zero(f'Seed: {cfg.seed}')
        print_rank_zero('=' * 80)
        # print_rank_zero('')
        # print_rank_zero('Resolved config (after interpolation):')
        # print_rank_zero(OmegaConf.to_yaml(cfg, resolve=True))
        # print_rank_zero('')

    # Resolve resume (folder or single ckpt)
    resume_ckpt, resume_logdir, melk_ckpt_name = _resolve_resume(cfg)

    # fork_log semantics:
    # - If resuming and fork_log=True: load ckpt from resume_logdir, BUT create a NEW run folder & new WandB run
    # - If resuming and fork_log=False: keep using the SAME run folder name (like your old script)
    keep_existing_folder = bool(resume_logdir and not cfg.fork_log)

    # Compute output folders to match your legacy structure
    logdir, nowname, ckptdir, cfgdir = _make_log_structure(
        cfg,
        now,
        keep_existing=keep_existing_folder,
        resume_logdir=resume_logdir,
    )

    if local_rank_for_print == 0:
        Path(logdir).mkdir(parents=True, exist_ok=True)
        print_rank_zero(f'📁 Log directory: {logdir}')

    # Optional backward-compat remap of checkpoint paths
    remapped_ckpt = None
    ckpt_needed_remapping = False
    if resume_ckpt is not None:
        remapped_ckpt, ckpt_needed_remapping = remap_checkpoint_paths(resume_ckpt)
        resume_ckpt = remapped_ckpt

    # ============================================================================
    # MODEL & DATA INSTANTIATION
    # ============================================================================

    if local_rank_for_print == 0:
        print_rank_zero('Instantiating model and data modules...')

    # Model
    model: L.LightningModule = instantiate_from_config(OmegaConf.to_container(cfg.model, resolve=True))

    # Data
    data: L.LightningDataModule = instantiate_from_config(OmegaConf.to_container(cfg.data, resolve=True))
    if hasattr(data, 'tokenizer'):
        model.tokenizer = data.tokenizer
    else:
        print_rank_zero('⚠️  Warning: Data module does not have a tokenizer.')

    # Note: Model summary can be added via RichModelSummary callback in config if desired.

    # ============================================================================
    # TRAINER SETUP (Strategy, Callbacks, Loggers)
    # ============================================================================

    # Strategy
    strategy = None
    if cfg.lightning.get('strategy'):
        strategy_cfg = OmegaConf.to_container(cfg.lightning.strategy, resolve=True)
        tgt = strategy_cfg.get('target') or strategy_cfg.get('_target_', '')
        if tgt == 'dlm.lightning.fsdp_huggingface':
            # Your wrapper needs the model instance
            strategy = fsdp_huggingface(model=model, **cfg.lightning.strategy.get('params', {}))
        else:
            if tgt == 'lightning.pytorch.strategies.DDPStrategy' and use_slurm_env:
                params = dict(strategy_cfg.get('params', {}) or {})
                if 'cluster_environment' not in params:
                    params['cluster_environment'] = slurm_env
                    strategy_cfg['params'] = params
                    print_rank_zero('🔧 Detected SLURM environment, attaching it to DDPStrategy.')
            # Anything else (e.g., DDPStrategy or native FSDPStrategy) can be instantiated normally
            strategy = instantiate_from_config(strategy_cfg)

    # Callbacks
    callbacks = []
    if cfg.lightning.get('callbacks'):
        callbacks.extend(
            [instantiate_from_config(OmegaConf.to_container(c, resolve=True)) for c in cfg.lightning.callbacks.values()],
        )
        if local_rank_for_print == 0:
            print_rank_zero(f'⚙️  Loaded {len(callbacks)} callbacks:')
            for cb in callbacks:
                print_rank_zero(f'   • {type(cb).__name__}')

    # Add RichModelSummary for proper model info logging
    from lightning.pytorch.callbacks import RichModelSummary

    callbacks.append(RichModelSummary(max_depth=2))
    
    # override ModelCheckpoint.dirpath to ckptdir
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.dirpath = str(ckptdir)  # <— send checkpoints to <logdir>/checkpoints
    setup_callbacks_cfg = {
        'target': 'dlm.callbacks.setup_callback.SetupCallback',
        'params': {
            'resume': resume_logdir is not None,
            'now': now,
            'logdir': str(logdir),
            'ckptdir': ckptdir,
            'cfgdir': cfgdir,
            'config': cfg,
            'lightning_config': cfg.lightning,
            'debug': cfg.debug,
            'ckpt_name': melk_ckpt_name,
        },
    }
    callbacks.append(instantiate_from_config(setup_callbacks_cfg))
    enable_wandb = bool(cfg.get('enable_wandb', True))
    if enable_wandb:
        from dlm.callbacks.wandb_callback import SaveWandbIDCallback

        callbacks.append(SaveWandbIDCallback())

    # Only add these when doing prediction
    if cfg.mode == 'pred' and cfg.lightning.get('predict_callbacks'):
        callbacks.extend(
            [
                instantiate_from_config(OmegaConf.to_container(c, resolve=True))
                for c in cfg.lightning.predict_callbacks.values()
            ],
        )

    # Loggers, with optional W&B fork/resume logic.
    csv_logger = CSVLogger(save_dir=str(logdir), name='csv', version=None)
    loggers = [csv_logger]

    if enable_wandb:
        from lightning.pytorch.loggers import WandbLogger

        wandb_name = str(cfg.paths.log_root).split('/')[-1] + '_' + nowname
        wandb_kwargs = {
            'name': wandb_name,  # nowname,
            'project': cfg.projectname,
            'offline': cfg.debug,
            'save_dir': str(logdir),
            'group': cfg.group if cfg.group is not None else HydraConfig.get().runtime.choices.get('data', 'default'),
        }
        if resume_logdir and not cfg.fork_log:
            wandb_id_path = resume_logdir / 'wandb_id.txt'
            if wandb_id_path.exists():
                with open(wandb_id_path) as f:
                    wandb_id = f.read().strip()
                print_rank_zero(f"📊 Resuming W&B run (ID: {wandb_id})")
                wandb_kwargs['id'] = wandb_id
                wandb_kwargs['resume'] = 'allow'
                wandb_kwargs['name'] = None
            else:
                print_rank_zero('⚠️  Could not find wandb_id.txt. Starting a new W&B run.')
                wandb_kwargs['id'] = None
                wandb_kwargs['resume'] = None
        else:
            # Fresh run when forking or starting new
            wandb_kwargs['id'] = None
            wandb_kwargs['resume'] = None

        loggers.insert(0, WandbLogger(**wandb_kwargs))

    if local_rank_for_print == 0:
        if enable_wandb:
            print_rank_zero(f'📊 Logging: W&B ({"offline" if cfg.debug else "online"}) + CSV')
        else:
            print_rank_zero('📊 Logging: CSV only (W&B disabled)')

    # Trainer
    trainer_kwargs = {'logger': loggers, 'callbacks': callbacks}
    if strategy is not None:
        trainer_kwargs['strategy'] = strategy
    
    # Convert OmegaConf to a mutable Python dict
    trainer_config = OmegaConf.to_container(cfg.lightning.trainer, resolve=True)

    # Use no_grad-style evaluation instead of inference_mode for robustness.
    # This avoids inference-tensor backward issues when metrics need autograd.
    # enable_gradient_moment_metric = bool(cfg.model.params.get('enable_gradient_moment_metric', False))
    # if trainer_config.get('inference_mode', True) and enable_gradient_moment_metric:
    #     trainer_config['inference_mode'] = False
    #     print_rank_zero(
    #         'Setting Trainer(inference_mode=False) so evaluation uses no_grad '
    #         'instead of inference_mode.',
    #     )

    # If we explicitly passed a strategy object in trainer_kwargs,
    # remove the 'strategy' key from the config dict to avoid the collision.
    if 'strategy' in trainer_kwargs and 'strategy' in trainer_config:
        print_rank_zero(f"Overrides: Replacing config strategy '{trainer_config['strategy']}' with manually instantiated strategy.")
        del trainer_config['strategy']

    # trainer = L.Trainer(**cfg.lightning.trainer, **trainer_kwargs)
    trainer = L.Trainer(**trainer_config, **trainer_kwargs)
    trainer.logdir = str(logdir)

    # ---------------- Weights-only vs full resume ----------------
    # - If load_weights_only is provided: do NOT pass ckpt_path to Trainer; pass path to model via hparams
    # - Otherwise, ckpt_path = resume_ckpt
    final_ckpt_for_trainer = resume_ckpt
    if cfg.load_weights_only:
        model.hparams.weights_only_path = cfg.load_weights_only
        final_ckpt_for_trainer = None

    # Save composed config into the run folder (like your SetupCallback used to)
    # You can still keep your SetupCallback; this is just a convenience snapshot.
    try:
        Path(cfgdir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, cfgdir / 'composed_config.yaml')
    except Exception as e:
        print_rank_zero(f'⚠️  Warning: failed to save composed config: {e}')

    # ============================================================================
    # RUN TRAINING/VALIDATION/PREDICTION
    # ============================================================================

    if local_rank_for_print == 0:
        print_rank_zero('')
        print_rank_zero('=' * 80)
        print_rank_zero(f'▶️  Starting {cfg.mode} phase...')
        print_rank_zero('=' * 80)
        print_rank_zero('')

    if local_rank_for_print == 0:
        data.prepare_data()
    try:
        if cfg.mode == 'train':
            trainer.fit(model=model, datamodule=data, ckpt_path=final_ckpt_for_trainer)
        elif cfg.mode == 'val':
            trainer.validate(model=model, datamodule=data, ckpt_path=final_ckpt_for_trainer)
        elif cfg.mode == 'pred':
            trainer.predict(model=model, datamodule=data, ckpt_path=final_ckpt_for_trainer)
    finally:
        # 1. Robust Rank Check: Use ENV variables instead of trainer.global_rank
        #    because trainer state might be broken if we crashed early.
        rank = int(os.environ.get("RANK", "0"))
        
        # 2. Robust Move: Check if source exists before moving
        if cfg.debug and rank == 0 and ('debug_runs' not in str(logdir)):
            dst = Path(logdir).parent / 'debug_runs' / Path(logdir).name
            src = Path(logdir)
            
            if src.exists():
                try:
                    Path(dst).parent.mkdir(parents=True, exist_ok=True)
                    src.rename(dst)
                    print_rank_zero(f"Moved debug run to {dst}")
                except OSError as e:
                    print_rank_zero(f"Could not move debug run: {e}")
            
        # cleanup temp remapped ckpt
        if ckpt_needed_remapping and remapped_ckpt and rank == 0:
            if os.path.exists(remapped_ckpt):
                print_rank_zero(f'🧹 Cleaning up temporary checkpoint: {remapped_ckpt}')
                os.remove(remapped_ckpt)

    # ============================================================================
    # COMPLETION
    # ============================================================================

    if local_rank_for_print == 0:
        print_rank_zero('')
        print_rank_zero('=' * 80)
        print_rank_zero('✅ Lightning run successfully completed!')
        print_rank_zero('=' * 80)
        if torch.cuda.is_available():
            print_rank_zero(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
        print_rank_zero(f'Results saved to: {logdir}')
        print_rank_zero('=' * 80)
        print_rank_zero('')


if __name__ == '__main__':
    # Keep sys.path behavior from your old script
    sys.path.append(os.getcwd())
    main()
