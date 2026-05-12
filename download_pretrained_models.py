#!/usr/bin/env python
"""Download pretrained models (Qwen, GPT-2, BERT, etc.).

This script should be run on login nodes where internet access is available,
before launching training jobs on compute nodes without internet access.

Usage:
    # Download all default models to ./.hf_cache
    python download_pretrained_models.py

    # Download specific models
    python download_pretrained_models.py --models qwen-0.6b qwen-8b gpt2 bert-base-uncased

    # Use custom cache directory
    python download_pretrained_models.py --cache-dir /path/to/cache

    # Download tokenizers only (faster, for testing)
    python download_pretrained_models.py --tokenizer-only
"""

import argparse
import os
from pathlib import Path
# Match environment variables from hydra_main.py
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'


# Map of model aliases to HuggingFace model identifiers
MODEL_MAP = {
    'qwen-0.6b': 'Qwen/Qwen3-Embedding-0.6B',
    'qwen-8b': 'Qwen/Qwen3-Embedding-8B',
    'gpt2': 'gpt2',
    'gpt2-large': 'gpt2-large',
    'bert-base-uncased': 'bert-base-uncased',
}
MODEL_CHOICES = list(MODEL_MAP.keys())

# Default models to download (used in configs)
DEFAULT_MODELS = ['qwen-0.6b', 'gpt2-large', 'bert-base-uncased']


def configure_hf_cache(cache_dir: str) -> str:
    """Configure HuggingFace cache directories via environment variables.

    This function matches the cache configuration in hydra_main.py exactly.

    Args:
        cache_dir: Root cache directory (e.g., './.hf_cache')

    Returns:
        Configured cache root path.
    """
    cache_root = os.path.abspath(os.path.expanduser(cache_dir))
    hub_cache = os.path.join(cache_root, 'hub')
    datasets_cache = os.path.join(cache_root, 'datasets')

    # Set all relevant HF environment variables (matching hydra_main.py)
    os.environ['HF_HOME'] = cache_root
    os.environ['HF_HUB_CACHE'] = hub_cache
    os.environ['HF_DATASETS_CACHE'] = datasets_cache
    os.environ['TRANSFORMERS_CACHE'] = hub_cache

    # Create directories if they don't exist
    Path(hub_cache).mkdir(parents=True, exist_ok=True)
    Path(datasets_cache).mkdir(parents=True, exist_ok=True)

    print(f'Configured HuggingFace cache root: {cache_root}')
    print(f'  - Hub cache: {hub_cache}')
    print(f'  - Datasets cache: {datasets_cache}')

    return cache_root


def download_model(model_id: str, tokenizer_only: bool = False, trust_remote_code: bool = True) -> None:
    """Download a pretrained model and/or tokenizer.

    Args:
        model_id: HuggingFace model identifier (e.g., 'Qwen/Qwen3-Embedding-0.6B')
        tokenizer_only: If True, only download tokenizer (faster)
        trust_remote_code: Whether to trust remote code in model repos
    """
    from transformers import AutoModel, AutoTokenizer

    print(f'\n{"=" * 80}')
    print(f'Downloading: {model_id}')
    print(f'{"=" * 80}')

    # Always download tokenizer
    print('  → Downloading tokenizer...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        print('  ✓ Tokenizer downloaded successfully')
        print(f'    Vocab size: {tokenizer.vocab_size}')
    except Exception as e:
        print(f'  ✗ Failed to download tokenizer: {e}')
        raise

    # Download full model if requested
    if not tokenizer_only:
        print('  → Downloading model weights...')
        try:
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
            print('  ✓ Model downloaded successfully')

            # Print model info
            if hasattr(model, 'config'):
                config = model.config
                if hasattr(config, 'hidden_size'):
                    print(f'    Hidden size: {config.hidden_size}')
                if hasattr(config, 'num_hidden_layers'):
                    print(f'    Layers: {config.num_hidden_layers}')
                if hasattr(config, 'num_attention_heads'):
                    print(f'    Attention heads: {config.num_attention_heads}')

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f'    Total parameters: {total_params:,}')

            del model  # Free memory

        except Exception as e:
            print(f'  ✗ Failed to download model: {e}')
            raise

    print(f'✓ Completed: {model_id}\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download pretrained models to HuggingFace cache.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=[*MODEL_CHOICES, 'all'],
        default=None,
        help=f'Models to download. Options: {", ".join(MODEL_CHOICES)}, all. Default: {", ".join(DEFAULT_MODELS)}',
    )
    parser.add_argument(
        '--cache-dir',
        default='./.hf_cache',
        help='HF cache root directory. Default: ./.hf_cache (matches hydra_main.py)',
    )
    parser.add_argument(
        '--tokenizer-only',
        action='store_true',
        help='Only download tokenizers (faster, for testing cache setup)',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=True,
        help='Trust remote code in model repos (default: True)',
    )

    args = parser.parse_args()

    # Configure cache
    cache_root = configure_hf_cache(args.cache_dir)

    # Determine which models to download
    if args.models is None:
        models_to_download = DEFAULT_MODELS
    elif 'all' in args.models:
        models_to_download = MODEL_CHOICES
    else:
        models_to_download = args.models

    print(f'\nModels to download: {", ".join(models_to_download)}')
    if args.tokenizer_only:
        print('Mode: Tokenizer only')
    else:
        print('Mode: Full model + tokenizer')
    print()

    # Download each model
    success_count = 0
    failed_models = []

    for model_alias in models_to_download:
        try:
            model_id = MODEL_MAP[model_alias]
            download_model(
                model_id,
                tokenizer_only=args.tokenizer_only,
                trust_remote_code=args.trust_remote_code,
            )
            success_count += 1
        except Exception as e:
            model_label = MODEL_MAP.get(model_alias, model_alias)
            print(f'✗ Failed to download {model_alias} ({model_label}): {e}\n')
            failed_models.append(model_alias)

    # Summary
    print(f'\n{"=" * 80}')
    print('DOWNLOAD SUMMARY')
    print(f'{"=" * 80}')
    print(f'Total models: {len(models_to_download)}')
    print(f'Successful: {success_count}')
    print(f'Failed: {len(failed_models)}')

    if failed_models:
        print(f'\nFailed models: {", ".join(failed_models)}')

    if cache_root:
        print(f'\nCache location: {cache_root}')
        print('\nTo use this cache in your training runs, set:')
        print(f'  export HF_HOME={cache_root}')
        print(f'  export TRANSFORMERS_CACHE={cache_root}/hub')
        print(f'  export HF_DATASETS_CACHE={cache_root}/datasets')

    print('\n✓ Download script completed')


if __name__ == '__main__':
    main()
