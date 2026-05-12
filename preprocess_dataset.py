#!/usr/bin/env python
"""Preprocess datasets using this codebase's datamodules and data configs.

This script intentionally reuses project code paths (datamodule instantiation,
`prepare_data()`, and `setup()`) so behavior stays aligned with training.

Examples:
    uv run python preprocess_dataset.py conf/data/owt/owt_bert_512.yaml
    uv run python preprocess_dataset.py conf/data/lm1b/lm1b_qwen_packed.yaml
    uv run python preprocess_dataset.py conf/data/text8/text8.yaml
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dlm.utils import instantiate_from_config

# Match environment variables from hydra_main.py
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'


def _ensure_sched_getaffinity_fallback() -> None:
    """Provide os.sched_getaffinity on platforms that do not implement it."""
    if hasattr(os, 'sched_getaffinity'):
        return

    def _sched_getaffinity(_pid: int) -> set[int]:
        return set(range(os.cpu_count() or 1))

    os.sched_getaffinity = _sched_getaffinity  # type: ignore[attr-defined]


def configure_hf_cache(cache_dir: str) -> str:
    """Configure HuggingFace cache directories.

    Matches the cache setup logic in hydra_main.py.
    """
    cache_root = os.path.abspath(os.path.expanduser(cache_dir))
    hub_cache = os.path.join(cache_root, 'hub')
    datasets_cache = os.path.join(cache_root, 'datasets')

    os.environ['HF_HOME'] = cache_root
    os.environ['HF_HUB_CACHE'] = hub_cache
    os.environ['HF_DATASETS_CACHE'] = datasets_cache
    os.environ['TRANSFORMERS_CACHE'] = hub_cache

    Path(hub_cache).mkdir(parents=True, exist_ok=True)
    Path(datasets_cache).mkdir(parents=True, exist_ok=True)
    return cache_root


def configure_hf_offline(enabled: bool) -> None:
    if not enabled:
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'


def _extract_data_config(config_path: Path) -> dict:
    """Load a YAML config and return the `data` config dict."""
    cfg = _load_config(config_path)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_resolved, dict):
        raise TypeError(f'Config must resolve to a dictionary: {config_path}')

    if isinstance(cfg_resolved.get('data'), dict):
        data_cfg = cfg_resolved['data']
    else:
        data_cfg = cfg_resolved

    if 'target' not in data_cfg:
        raise KeyError(
            f'Could not find a datamodule target in config: {config_path}. '
            'Expected either `data.target` or top-level `target`.',
        )
    return data_cfg


def _load_config(config_path: Path):
    repo_conf_dir = Path(__file__).resolve().parent / 'conf'
    resolved_path = config_path.expanduser().resolve()
    try:
        relative_path = resolved_path.relative_to(repo_conf_dir.resolve())
    except ValueError:
        return OmegaConf.load(str(config_path))

    if relative_path.suffix not in {'.yaml', '.yml'}:
        return OmegaConf.load(str(config_path))

    config_name = str(relative_path.with_suffix(''))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(repo_conf_dir.resolve()), version_base='1.3'):
        return compose(config_name=config_name)


def _ensure_datamodule_trainer(datamodule: object) -> None:
    """Attach a minimal trainer-like object when running outside Lightning Trainer."""
    trainer = getattr(datamodule, 'trainer', None)
    if trainer is not None and hasattr(trainer, 'is_global_zero'):
        return
    datamodule.trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        print=print,
    )


def _describe_outputs(datamodule: object) -> None:
    save_dir = getattr(datamodule, 'save_dir', None)
    raw_save_dir = getattr(datamodule, 'raw_save_dir', None)

    if save_dir is not None:
        save_path = Path(save_dir).expanduser().resolve()
        print(f'  save_dir: {save_path} (exists={save_path.exists()})')
    if raw_save_dir is not None:
        raw_path = Path(raw_save_dir).expanduser().resolve()
        print(f'  raw_save_dir: {raw_path} (exists={raw_path.exists()})')

    datasets = getattr(datamodule, 'datasets', None)
    if isinstance(datasets, dict) and datasets:
        split_lengths = {}
        for split_name, split_dataset in datasets.items():
            try:
                split_lengths[split_name] = len(split_dataset)
            except Exception:
                split_lengths[split_name] = 'unknown'
        print(f'  dataset splits: {split_lengths}')


def preprocess_from_config(config_path: Path) -> None:
    data_cfg = _extract_data_config(config_path)
    target = data_cfg['target']
    print(f'\n{"=" * 100}')
    print(f'Preprocessing from config: {config_path}')
    print(f'Data target: {target}')
    print(f'Working directory: {Path.cwd()}')
    print(f'{"=" * 100}')

    datamodule = instantiate_from_config(data_cfg)
    _ensure_datamodule_trainer(datamodule)

    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    _describe_outputs(datamodule)
    print('[OK] Preprocessing completed')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Preprocess dataset(s) using project datamodule configs.',
    )
    parser.add_argument(
        'configs',
        nargs='+',
        help='Path(s) to data config YAML files (e.g., conf/data/owt/owt_bert_512.yaml).',
    )
    parser.add_argument(
        '--cache-dir',
        default='./.hf_cache',
        help='HF cache root. Default: ./.hf_cache (matches hydra_main.py)',
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Enable HF/Transformers offline mode (local cache only).',
    )
    parser.add_argument(
        '--keep-cwd',
        action='store_true',
        help='Do not change to repository root before preprocessing.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_sched_getaffinity_fallback()

    if not args.keep_cwd:
        repo_root = Path(__file__).resolve().parent
        os.chdir(repo_root)

    cache_root = configure_hf_cache(args.cache_dir)
    configure_hf_offline(args.offline)

    print(f'Cache root: {cache_root}')
    print(f'Offline mode: {args.offline}')

    config_paths = [Path(config).expanduser().resolve() for config in args.configs]
    failed: list[Path] = []

    for config_path in config_paths:
        try:
            preprocess_from_config(config_path)
        except Exception as e:
            print(f'[FAIL] {config_path}: {e}')
            failed.append(config_path)

    print(f'\n{"=" * 100}')
    print('PREPROCESS SUMMARY')
    print(f'{"=" * 100}')
    print(f'Total configs: {len(config_paths)}')
    print(f'Successful: {len(config_paths) - len(failed)}')
    print(f'Failed: {len(failed)}')
    if failed:
        print('Failed configs:')
        for failed_path in failed:
            print(f'  - {failed_path}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
