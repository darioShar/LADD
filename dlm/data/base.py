import logging
import os
from itertools import chain
from pathlib import Path
from typing import Any

import lightning as L
import torch
import transformers
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ..utils import instantiate_from_config, instantiate_from_config_hf_pretrained, print_rank_zero




class DatasetWithPrecomputedLatents(Dataset):
    """A wrapper that adds precomputed latents to an existing dataset."""

    def __init__(self, base_dataset, precomputed_latents, verify_latents_match):
        self.base_dataset = base_dataset
        self.precomputed_latents = precomputed_latents
        self.verify_latents_match = verify_latents_match
        self._latents_are_dataset = hasattr(precomputed_latents, 'column_names')

        if self.verify_latents_match:
            # Validate that the number of latents matches the dataset size
            latents_len = len(self.precomputed_latents)
            if not self._latents_are_dataset:
                latents_len = len(self.precomputed_latents['latent'])
            if latents_len != len(self.base_dataset):
                raise ValueError(
                    f'Number of precomputed latents ({latents_len}) '
                    f'does not match dataset size ({len(self.base_dataset)})',
                )
        if self._latents_are_dataset:
            shape_hint = None
            try:
                shape_hint = self.precomputed_latents[0]['latent'].shape
            except Exception:
                shape_hint = None
            print_rank_zero(f'Precomputed latents dataset attached (len={len(self.precomputed_latents)}, shape={shape_hint}).')
        else:
            print_rank_zero(f'Precomputed latents successfully added to dataset: {self.precomputed_latents["latent"].shape}')

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, dict):
            item = dict(item)  # Make a copy to avoid modifying the original
        else:
            # If item is not a dict, wrap it
            item = {'input_ids': item}

        # Add precomputed latents
        if self._latents_are_dataset:
            latent_row = self.precomputed_latents[int(idx)]
            item['latent'] = latent_row['latent']
        else:
            item['latent'] = self.precomputed_latents['latent'][idx]

        # Verify indices match (similar to text8 implementation)
        if self.verify_latents_match:
            expected_idx = None
            if self._latents_are_dataset:
                if 'idx' in latent_row:
                    expected_idx = latent_row['idx']
            else:
                assert ('idx' in self.precomputed_latents), 'Could not find "idx" in precomputed latents for verification'
                expected_idx = self.precomputed_latents['idx'][idx]
            if expected_idx is None:
                return item
            if not isinstance(expected_idx, torch.Tensor):
                expected_idx = torch.tensor(expected_idx)
            if not isinstance(idx, torch.Tensor):
                idx = torch.tensor(idx)
            if not torch.equal(idx, expected_idx):
                raise ValueError(f'Index mismatch: {idx} != {expected_idx}')

        return item

    # def __getattr__(self, name):
    #     # Delegate other attributes to the base dataset
    #     return getattr(self.base_dataset, name)


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_config: dict | None = None,
        data_path: str | None = None,
        data_name: str | None = None,
        data_kwargs: dict | None = None,
        tokenizer_config: dict | None = None,
        transform_config: dict | None = None,
        batch_size: int = 64,
        eval_batch_size: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 2,
        train_split: str = 'train',
        val_size: float | None = None,
        val_split: str = 'validation',
        test_split: str = 'test',
        transform_all: bool = False,  # If False, the transform will be applied on-the-fly pm batches when __getitem__ is called.
    ):
        super().__init__()
        if data_config is None:
            if data_kwargs is None:
                data_kwargs = {}
            data_config = {
                'target': 'datasets.load_dataset',
                'params': {'path': data_path, 'name': data_name, **data_kwargs},
            }
        self.data_config = data_config
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.train_split = train_split
        self.val_size = val_size
        self.val_split = val_split
        self.test_split = test_split
        self.transform_all = transform_all
        self.tokenizer = instantiate_from_config_hf_pretrained(tokenizer_config) if tokenizer_config is not None else None
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.transform = None
        if transform_config is not None:
            self.transform = instantiate_from_config(
                transform_config,
                tokenizer=self.tokenizer,
            )
        self.datasets = {}

    def prepare_data(self):
        target = self.data_config.get('target') if hasattr(self.data_config, 'get') else None
        params = self.data_config.get('params') if hasattr(self.data_config, 'get') else None
        data_path = params.get('path') if hasattr(params, 'get') else None
        print_rank_zero(
            f'BaseDataModule.prepare_data: cwd={os.getcwd()}, target={target}, data_path={data_path}',
        )
        instantiate_from_config(self.data_config)

    def preprocess(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        return dataset

    def setup(self, stage=None):
        dataset = instantiate_from_config(self.data_config)
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({self.train_split: dataset})
        dataset = self.preprocess(dataset)
        if self.transform is not None and self.transform_all:
            dataset = dataset.map(
                self.transform,
                num_proc=self.num_workers,
                desc='Applying transform',
                fn_kwargs={'batched': False},
            )
        elif self.transform is not None:
            dataset = dataset.with_transform(self.transform)
        self._assign_dataset_splits(dataset)

    def _assign_dataset_splits(self, dataset: DatasetDict) -> None:
        self.datasets = {}
        if self.val_size is not None:
            if self.train_split not in dataset:
                raise ValueError(
                    f'Cannot create validation split from dataset without train split "{self.train_split}". '
                    f'Available splits: {list(dataset.keys())}',
                )
            if isinstance(self.val_size, float):
                self.datasets['train'], self.datasets['val'] = (
                    dataset[self.train_split].train_test_split(test_size=self.val_size, seed=42).values()
                )
            else:
                # if val_size is an int, we take the last val_size samples as the validation set
                train_dataset = dataset[self.train_split].select(
                    range(len(dataset[self.train_split]) - self.val_size),
                )
                val_dataset = dataset[self.train_split].select(
                    range(len(dataset[self.train_split]) - self.val_size, len(dataset[self.train_split])),
                )
                self.datasets['train'] = train_dataset
                self.datasets['val'] = val_dataset
        else:
            if self.train_split in dataset:
                self.datasets['train'] = dataset[self.train_split]
            if self.val_split in dataset:
                self.datasets['val'] = dataset[self.val_split]
        if self.test_split in dataset:
            self.datasets['test'] = dataset[self.test_split]

    def _dataloader_kwargs(self, batch_size, shuffle, drop_last):
        kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers,
            'drop_last': drop_last,
            'pin_memory': self.pin_memory,
            'collate_fn': self._collate_batch,
        }
        if self.num_workers > 0:
            kwargs['persistent_workers'] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs['prefetch_factor'] = self.prefetch_factor
        return kwargs

    def _collate_batch(self, batch):
        batch = default_collate(batch)
        attention_mask = batch.get('attention_mask')
        if attention_mask is None:
            if getattr(self, 'pack_sequences', False):
                batch['attention_mask'] = torch.ones_like(batch['input_ids'])
            return batch
        if getattr(self, 'pack_sequences', False):
            return batch
        if attention_mask.dim() != 2:
            return batch
        if attention_mask.dtype == torch.bool:
            is_full = attention_mask.all()
        else:
            is_full = (attention_mask == 1).all()
        if bool(is_full):
            batch.pop('attention_mask', None)
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            **self._dataloader_kwargs(
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            **self._dataloader_kwargs(
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=True,
            ),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            **self._dataloader_kwargs(
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            **self._dataloader_kwargs(
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=True,
            ),
        )


class DataModuleForPT(BaseDataModule):
    """
    Data module for pretraining, where all texts are tokenized and grouped into chunks of max_length.
    """

    def __init__(
        self,
        max_length: int,
        transform_config=None,
        streaming_num_shards=None,
        num_proc=None,
        save_dir=None,
        raw_save_dir=None,
        precomputed_latents_path=None,
        verify_latents_match = True,
        pack_sequences: bool = True,
        text_column: str = 'text',
        text_fields: list[str] | None = None,
        text_field_separator: str = ' ',
        detokenizers: list[str] | None = None,
        use_eos_separation: bool = True,
        **kwargs,
    ):
        assert transform_config is None, 'transform_config is not supported for DataModuleForPT'
        super().__init__(**kwargs)
        self.max_length = max_length
        self.streaming_num_shards = streaming_num_shards
        self.num_proc = num_proc or self.num_workers
        self.save_dir = save_dir
        self.raw_save_dir = raw_save_dir
        self.precomputed_latents_path = precomputed_latents_path
        self.precomputed_latents = {'train': None, 'val': None}
        self.verify_latents_match = verify_latents_match
        self.pack_sequences = pack_sequences
        self.text_column = text_column
        self.text_fields = text_fields
        self.text_field_separator = text_field_separator
        self.detokenizers = list(detokenizers) if detokenizers is not None else None
        self.use_eos_separation = use_eos_separation
        self._built = False

    def _detokenizer_names(self) -> list[str]:
        if self.detokenizers is not None:
            return self.detokenizers
        params = self.data_config.get('params', {}) if hasattr(self.data_config, 'get') else {}
        dataset_path = params.get('path', '') if hasattr(params, 'get') else ''
        if isinstance(dataset_path, str) and 'lm1b' in dataset_path:
            return ['lm1b']
        return []

    def _normalize_text_column(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if not self.text_fields:
            return dataset

        text_fields = tuple(self.text_fields)
        separator = self.text_field_separator
        target_column = self.text_column

        def _build_text(example: dict[str, Any]) -> dict[str, str]:
            values: list[str] = []
            for field in text_fields:
                value = example[field]
                if value is None:
                    continue
                value_str = str(value).strip()
                if value_str:
                    values.append(value_str)
            return {target_column: separator.join(values)}

        return dataset.map(
            _build_text,
            num_proc=self.num_proc,
            desc=f'Building {target_column} column',
        )

    def _apply_detokenizers(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        from .utils import get_detokenizer

        for name in self._detokenizer_names():
            detokenizer = get_detokenizer(name)

            def _detokenize(example, detokenizer=detokenizer):
                return {self.text_column: detokenizer(example[self.text_column])}

            dataset = dataset.map(
                _detokenize,
                num_proc=self.num_proc,
                desc=f'Applying {name} detokenizer',
            )
        return dataset

    def preprocess(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if self.raw_save_dir is not None:
            if os.path.exists(self.raw_save_dir):
                dataset = load_from_disk(self.raw_save_dir)
            elif self.streaming_num_shards is None:
                if self.trainer.is_global_zero:
                    dataset.save_to_disk(self.raw_save_dir)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            else:
                print_rank_zero('Skipping raw_save_dir for streaming datasets.')
        if self.save_dir is not None and os.path.exists(self.save_dir):
            return load_from_disk(self.save_dir)
        if self.streaming_num_shards is not None:
            dataset = dataset.to_iterable_dataset(num_shards=self.streaming_num_shards)
        dataset = self._normalize_text_column(dataset)
        dataset = self._apply_detokenizers(dataset)
        # tokenize the dataset
        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger(
            'transformers.tokenization_utils_base',
        ).setLevel(logging.ERROR)

        if self.pack_sequences:
            def tokenize_function(examples):
                input_ids = self.tokenizer(
                    examples[self.text_column],
                    add_special_tokens=False,
                    return_attention_mask=False,
                ).input_ids
                if self.use_eos_separation:
                    input_ids = [i + [self.tokenizer.eos_token_id] for i in input_ids]
                return {'input_ids': input_ids}

            dataset = dataset.map(
                tokenize_function,
                num_proc=self.num_proc,
                batched=True,
                load_from_cache_file=True,
                desc='Tokenizing',
            )
            dataset = dataset.select_columns(['input_ids'])

            # group the dataset into chunks of block_size
            def group_texts(
                examples,
                block_size: int,
                bos_token_id: int,
                eos_token_id: int,
            ):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [[bos_token_id] + t[i : i + block_size] + [eos_token_id] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                if 'input_ids' in result:
                    result['attention_mask'] = [[1] * len(seq) for seq in result['input_ids']]
                return result

            dataset = dataset.map(
                group_texts,
                num_proc=self.num_proc,
                batched=True,
                fn_kwargs={
                    'block_size': self.max_length - 2,  # -2 for the two special tokens
                    'bos_token_id': self.tokenizer.bos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                },
                load_from_cache_file=True,
                desc='Grouping',
            )
            dataset = dataset.with_format(type='torch')
        else:
            def tokenize_and_pad(examples):
                tokenized = self.tokenizer(
                    examples[self.text_column],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length - 2,
                    return_attention_mask=False,
                )
                input_ids = []
                attention_masks = []
                labels = []
                # Respect tokenizer's padding_side setting
                padding_side = getattr(self.tokenizer, 'padding_side', 'right')
                for seq in tokenized['input_ids']:
                    seq = [self.tokenizer.bos_token_id] + seq + [self.tokenizer.eos_token_id]
                    pad_len = self.max_length - len(seq)
                    if pad_len < 0:
                        seq = seq[: self.max_length - 1] + [self.tokenizer.eos_token_id]
                        pad_len = 0

                    # Apply padding based on tokenizer's padding_side setting
                    padding = [self.tokenizer.pad_token_id] * pad_len
                    if padding_side == 'left':
                        mask = [0] * pad_len + [1] * len(seq)
                        padded = padding + seq
                        label = [-100] * pad_len + seq
                    else:  # right padding (default)
                        mask = [1] * len(seq) + [0] * pad_len
                        padded = seq + padding
                        label = [tok if m == 1 else -100 for tok, m in zip(padded, mask, strict=False)]

                    input_ids.append(padded)
                    attention_masks.append(mask)
                    labels.append(label)
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_masks,
                    'labels': labels,
                }

            dataset = dataset.map(
                tokenize_and_pad,
                num_proc=self.num_proc,
                batched=True,
                load_from_cache_file=True,
                desc='Tokenizing',
            )
            dataset = dataset.select_columns(['input_ids', 'attention_mask', 'labels'])
            dataset = dataset.with_format(type='torch')
        if self.trainer.is_global_zero and self.save_dir is not None:
            dataset.save_to_disk(self.save_dir)
        return dataset

    def prepare_data(self) -> None:
        save_dir_exists = self.save_dir is not None and os.path.exists(self.save_dir)
        raw_save_exists = self.raw_save_dir is not None and os.path.exists(self.raw_save_dir)
        print_rank_zero(
            f'DataModuleForPT.prepare_data: cwd={os.getcwd()}, save_dir={self.save_dir}, '
            f'save_dir_exists={save_dir_exists}, raw_save_dir={self.raw_save_dir}, raw_save_exists={raw_save_exists}',
        )
        if save_dir_exists or raw_save_exists:
            print_rank_zero('DataModuleForPT.prepare_data: skipping load_dataset because cached data exists.')
            return
        super().prepare_data()

    def setup(self, stage=None):
        # Call parent setup to handle dataset loading and preprocessing
        # Always (re)build only once
        if not self._built:
            print_rank_zero(
                f'DataModuleForPT.setup: cwd={os.getcwd()}, save_dir={self.save_dir}, '
                f'save_dir_exists={self.save_dir is not None and os.path.exists(self.save_dir)}',
            )
            if self.save_dir is not None and os.path.exists(self.save_dir):
                print_rank_zero(f'DataModuleForPT.setup: loading dataset from {self.save_dir}')
                dataset = load_from_disk(self.save_dir)
                if not isinstance(dataset, DatasetDict):
                    dataset = DatasetDict({self.train_split: dataset})
                self._assign_dataset_splits(dataset)
            else:
                print_rank_zero('DataModuleForPT.setup: falling back to BaseDataModule.setup')
                super().setup(stage)
            self._built = True

        # Load precomputed latents if available
        if self.precomputed_latents_path:
            path = Path(self.precomputed_latents_path)
            if path.exists():
                print_rank_zero(f'Loading precomputed latents from {path}...')
                if path.is_dir():
                    loaded_data = load_from_disk(path)
                    if isinstance(loaded_data, DatasetDict):
                        for k in ('train', 'val'):
                            if k in loaded_data:
                                cols = [c for c in ['latent', 'idx'] if c in loaded_data[k].column_names]
                                self.precomputed_latents[k] = loaded_data[k].with_format(type='torch', columns=cols)
                                print_rank_zero(f"Loaded HF latents split '{k}' with {len(self.precomputed_latents[k])} samples.")
                    else:
                        cols = [c for c in ['latent', 'idx'] if c in loaded_data.column_names]
                        self.precomputed_latents['train'] = loaded_data.with_format(type='torch', columns=cols)
                        print_rank_zero(
                            f"Loaded HF latents dataset (no splits) with {len(self.precomputed_latents['train'])} samples.",
                        )
                else:
                    loaded_data = torch.load(path, map_location='cpu', weights_only=False)
                    print_rank_zero(f'Latents file top-level keys: {list(loaded_data.keys())}')
                    for k in ('train', 'val'):
                        if k in loaded_data:
                            print_rank_zero(f"'{k}' keys: {list(loaded_data[k].keys())}")
                        self.precomputed_latents[k] = loaded_data[k]
                self._add_precomputed_latents_to_datasets()
            else:
                print_rank_zero(f'Warning: Precomputed latents path not found: {path}')

    def _add_precomputed_latents_to_datasets(self):
        """Add precomputed latents to the datasets by wrapping them."""
        for split in ['train', 'val']:
            if split in self.datasets and (self.precomputed_latents[split] is not None):
                print_rank_zero(f'Adding precomputed latents to {split} dataset')
                latents = self.precomputed_latents[split]
                if hasattr(latents, 'column_names'):
                    has_latents = 'latent' in latents.column_names and len(latents) > 0
                else:
                    has_latents = latents and len(latents.get('latent', [])) > 0
                if has_latents:
                    # if hasattr(self.datasets[split], 'reset_format'):
                    #     self.datasets[split] = self.datasets[split].with_format(None)
                    # Wrap the existing dataset to include precomputed latents
                    print_rank_zero('Constructing Precomputed Latents Dataset...')
                    self.datasets[split] = DatasetWithPrecomputedLatents(
                        self.datasets[split],
                        latents,
                        self.verify_latents_match
                    )
