#!/usr/bin/env python
"""Pre-download Hugging Face datasets for offline training.

Examples:
    uv run python download_hf_dataset.py --datasets text8 owt lm1b
    uv run python download_hf_dataset.py text8 owt lm1b
    uv run python download_hf_dataset.py --datasets all
    uv run python download_hf_dataset.py owt lm1b --cache-dir /path/to/.hf_cache
"""

import argparse
import os
from pathlib import Path

# Match environment variables from hydra_main.py
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'

HF_DATASET_MAP = {
    'owt': {'path': 'Skylion007/openwebtext'},
    'lm1b': {'path': 'lm1b'},
    'text8': {'path': 'afmck/text8'},
    'ptb': {'path': 'shenlong7/ptb_text_only', 'name': 'penn_treebank'},
    'wikitext103': {'path': 'Salesforce/wikitext', 'name': 'wikitext-103-raw-v1'},
    'lambada': {'path': 'EleutherAI/lambada_openai'},
    'ag_news': {'path': 'sh0416/ag_news'},
    'pubmed': {'path': 'armanc/scientific_papers', 'name': 'pubmed'},
    'arxiv': {'path': 'armanc/scientific_papers', 'name': 'arxiv'},
}

DEFAULT_DATASETS = ['owt', 'lm1b', 'text8', 'ptb', 'wikitext103', 'lambada', 'ag_news', 'pubmed', 'arxiv']
AVAILABLE_DATASETS = sorted(HF_DATASET_MAP)


def configure_hf_cache(cache_dir: str) -> str:
    """Configure HuggingFace cache directories.

    Matches the cache configuration in hydra_main.py exactly.
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


def download_hf_dataset(dataset_key: str) -> None:
    from datasets import load_dataset

    dataset_config = HF_DATASET_MAP[dataset_key]
    cache_dir = os.environ.get('HF_DATASETS_CACHE')
    dataset_path = dataset_config['path']
    dataset_name = dataset_config.get('name')

    print(f'\n{"=" * 80}')
    dataset_label = dataset_path if dataset_name is None else f'{dataset_path} [{dataset_name}]'
    print(f'Downloading Hugging Face dataset: {dataset_key} ({dataset_label})')
    print(f'{"=" * 80}')

    dataset = load_dataset(
        dataset_path,
        name=dataset_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    split_names = list(dataset.keys()) if hasattr(dataset, 'keys') else []
    print(f'[OK] Ready: {dataset_key}. Splits: {split_names}')


def resolve_datasets_arg(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    option_datasets = args.datasets or []
    positional_datasets = args.datasets_positional or []

    if option_datasets and positional_datasets:
        parser.error('Pass datasets either positionally or via --datasets, not both.')

    datasets = option_datasets or positional_datasets
    if not datasets:
        parser.error('No datasets provided. Use --datasets all or pass dataset names positionally.')

    if 'all' in datasets:
        return DEFAULT_DATASETS
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description='Pre-download datasets used by this project.')
    parser.add_argument(
        'datasets_positional',
        nargs='*',
        choices=[*AVAILABLE_DATASETS, 'all'],
        help='Datasets to pre-download. Positional alias for --datasets.',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=[*AVAILABLE_DATASETS, 'all'],
        help=f'Datasets to pre-download. Options: {", ".join(AVAILABLE_DATASETS)}, all',
    )
    parser.add_argument(
        '--cache-dir',
        default='./.hf_cache',
        help='HF cache root. Default: ./.hf_cache (matches hydra_main.py)',
    )
    args = parser.parse_args()

    cache_root = configure_hf_cache(args.cache_dir)
    datasets_to_download = resolve_datasets_arg(args, parser)

    print(f'Cache root: {cache_root}')
    print(f'Datasets to pre-download: {", ".join(datasets_to_download)}')

    success_count = 0
    failed_datasets = []

    for dataset_key in datasets_to_download:
        try:
            download_hf_dataset(dataset_key)
            success_count += 1
        except Exception as e:
            print(f'[FAIL] Failed to prepare {dataset_key}: {e}')
            failed_datasets.append(dataset_key)

    print(f'\n{"=" * 80}')
    print('DOWNLOAD SUMMARY')
    print(f'{"=" * 80}')
    print(f'Total datasets: {len(datasets_to_download)}')
    print(f'Successful: {success_count}')
    print(f'Failed: {len(failed_datasets)}')
    if failed_datasets:
        print(f'Failed datasets: {", ".join(failed_datasets)}')
    print('\n[OK] Dataset pre-download script completed')


if __name__ == '__main__':
    main()
