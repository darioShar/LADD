import os
import string
from pathlib import Path

import lightning as L
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from ..utils import print_rank_zero





class Text8Tokenizer:
    """A simple character-level tokenizer for the text8 dataset with special token support."""

    def __init__(
        self,
        bos_token: str = '<|endoftext|>',
        eos_token: str = '<|endoftext|>',
        pad_token: str = '<|endoftext|>',
        mask_token_id: int = 27,
    ):
        # Base characters: space + lowercase letters
        self.chars = [' '] + list(string.ascii_lowercase)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Add special token strings for compatibility with generative_perplexity.py
        # These are required for the tokenizer assertions, but we don't actually
        # increase vocab size - the IDs are set to None since text8 doesn't use them
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        # Set special token IDs to None to indicate they're not used in practice
        # This keeps vocab_size at 27 while satisfying the generative_perplexity assertions
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None

        # Mask token for discrete diffusion models
        self.mask_token_id = mask_token_id
        self.mask_token = '[MASK]'
        # Add mask token to idx_to_char for proper decoding
        self.idx_to_char[self.mask_token_id] = self.mask_token

        # Vocab size remains unchanged (space + a-z = 27)
        self.vocab_size = len(self.chars)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode.
            add_special_tokens: Ignored for text8 (kept for API compatibility).

        Returns:
            List of token IDs.
        """
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, tokens: list[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text.

        Args:
            tokens: Token IDs to decode.
            skip_special_tokens: Whether to skip special tokens (mask tokens) in output.

        Returns:
            Decoded text string.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()

        # Decode tokens, showing unknown tokens as [UNK_ID]
        if skip_special_tokens:
            # Skip mask tokens when requested
            return ''.join(
                self.idx_to_char.get(i, f'[UNK_{i}]')
                for i in tokens
                if i != self.mask_token_id
            )
        else:
            return ''.join(self.idx_to_char.get(i, f'[UNK_{i}]') for i in tokens)

    def batch_decode(self, tokens: list[list[int]] | torch.Tensor, skip_special_tokens: bool = False) -> list[str]:
        """Decode a batch of token IDs to text.

        Args:
            tokens: Batch of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in the output.

        Returns:
            List of decoded text strings.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        return [self.decode(t, skip_special_tokens=skip_special_tokens) for t in tokens]


class SlidingWindowDataset(Dataset):
    """Creates overlapping sequences using a sliding window approach.
    Used for training to maximize data utilization.
    """

    def __init__(self, tokenized_data: torch.Tensor, block_size: int, stride: int = 1):
        super().__init__()
        self.data = tokenized_data
        self.block_size = block_size
        self.stride = stride

    def __len__(self) -> int:
        return max(0, (len(self.data) - self.block_size) // self.stride + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        return {'input_ids': self.data[start : start + self.block_size]}


class ChunkedTextDataset(Dataset):
    """Splits data into non-overlapping chunks."""

    def __init__(
        self,
        tokenized_data: torch.Tensor,
        block_size: int,
        precomputed_latents=None,
        verify_latents_match: bool = True,
    ):
        super().__init__()
        num_chunks = len(tokenized_data) // block_size
        self.data = tokenized_data[: num_chunks * block_size]
        self.chunks = self.data.view(-1, block_size)
        self.precomputed_latents = precomputed_latents
        self.verify_latents_match = verify_latents_match

        if (self.precomputed_latents is not None) and self.verify_latents_match:
            latents_len = len(self.precomputed_latents)
            if not hasattr(self.precomputed_latents, 'column_names'):
                latents_len = len(self.precomputed_latents['latent'])
            if latents_len != len(self.chunks):
                raise ValueError(
                    f'Number of precomputed latents ({latents_len}) '
                    f'does not match number of chunks ({len(self.chunks)}) after adjustment.',
                )
            # Slice the precomputed latents to match the number of chunks
            # for key in self.precomputed_latents:
            #     self.precomputed_latents[key] = self.precomputed_latents[key][:num_chunks]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int | slice) -> dict[str, torch.Tensor]:
        item = {'input_ids': self.chunks[idx]}
        if self.precomputed_latents is not None:
            if hasattr(self.precomputed_latents, 'column_names'):
                latent_row = self.precomputed_latents[int(idx)]
                item['latent'] = latent_row['latent']
                if self.verify_latents_match and 'idx' in latent_row:
                    expected_idx = latent_row['idx']
                    if not isinstance(expected_idx, torch.Tensor):
                        expected_idx = torch.tensor(expected_idx)
                    if not torch.equal(torch.as_tensor(idx), expected_idx):
                        raise ValueError(f'Index mismatch: {idx} != {expected_idx}')
            else:
                item['latent'] = self.precomputed_latents['latent'][idx]
                if self.verify_latents_match:
                    # check that indices match
                    assert torch.all(torch.as_tensor(idx) == self.precomputed_latents['idx'][idx]), (
                        f'Index mismatch: {idx} != {self.precomputed_latents["idx"][idx]}'
                    )
        return item


class Text8DataModule(L.LightningDataModule):
    """
    An efficient DataModule for text8.

    - Tokenizes data once and saves to disk in `prepare_data`.
    - Uses non-overlapping chunks for all data splits.
    """

    def __init__(
        self,
        batch_size: int = 512,
        max_length: int = 256,
        data_dir: str = 'data/text8',
        num_workers: int | None = None,
        precomputed_latents_path: str | None = None,
        verify_latents_match: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = Text8Tokenizer()
        self.datasets: dict[str, Dataset] = {}
        self.tokenized_data_paths = {
            'train': Path(self.hparams.data_dir) / 'train_tokens.pt',
            'val': Path(self.hparams.data_dir) / 'val_tokens.pt',
        }
        self.precomputed_latents: dict[str, torch.Tensor | None] = {'train': None, 'val': None}
        self.verify_latents_match = verify_latents_match

    def prepare_data(self) -> None:
        """Download, tokenize, and save data to disk if not already present."""
        if all(p.exists() for p in self.tokenized_data_paths.values()):
            return

        print_rank_zero(f'Tokenizing and caching data to {self.hparams.data_dir}...')
        Path(self.hparams.data_dir).mkdir(parents=True, exist_ok=True)

        raw_datasets = load_dataset('afmck/text8', trust_remote_code=True)

        train_tokens = torch.tensor(
            self.tokenizer.encode(raw_datasets['train']['text'][0]),
            dtype=torch.long,
        )
        torch.save(train_tokens, self.tokenized_data_paths['train'])

        val_tokens = torch.tensor(
            self.tokenizer.encode(raw_datasets['validation']['text'][0]),
            dtype=torch.long,
        )
        torch.save(val_tokens, self.tokenized_data_paths['val'])

    def setup(self, stage: str | None = None) -> None:
        """Load data from disk and create datasets."""
        if self.hparams.precomputed_latents_path:
            path = Path(self.hparams.precomputed_latents_path)
            if path.exists():
                print_rank_zero(f'Loading precomputed latents from {path}...')
                if path.is_dir():
                    loaded_data = load_from_disk(path)
                    if isinstance(loaded_data, DatasetDict):
                        for k in ('train', 'val'):
                            if k in loaded_data:
                                cols = [c for c in ['latent', 'idx'] if c in loaded_data[k].column_names]
                                self.precomputed_latents[k] = loaded_data[k].with_format(type='torch', columns=cols)
                                print_rank_zero(
                                    f"Precomputed HF latents contain '{k}' split with {len(self.precomputed_latents[k])} samples.",
                                )
                    else:
                        cols = [c for c in ['latent', 'idx'] if c in loaded_data.column_names]
                        self.precomputed_latents['train'] = loaded_data.with_format(type='torch', columns=cols)
                        print_rank_zero(
                            f'Precomputed HF latents loaded with {len(self.precomputed_latents["train"])} samples.',
                        )
                else:
                    loaded_data = torch.load(path, map_location='cpu', weights_only=False)
                    print_rank_zero(f'Precomputed latents keys: {list(loaded_data.keys())}')
                    if 'train' in loaded_data:
                        print_rank_zero(
                            f"Precomputed latents contain 'train' split with {len(loaded_data['train']['latent'])} samples.",
                        )
                        self.precomputed_latents['train'] = loaded_data['train']
                    if 'val' in loaded_data:
                        print_rank_zero(
                            f"Precomputed latents contain 'val' split with {len(loaded_data['val']['latent'])} samples.",
                        )
                        self.precomputed_latents['val'] = loaded_data['val']
            else:
                print_rank_zero(f'Warning: Precomputed latents path not found: {path}')

        train_tokens = torch.load(self.tokenized_data_paths['train'], weights_only=False)
        val_tokens = torch.load(self.tokenized_data_paths['val'], weights_only=False)

        # As requested, using ChunkedTextDataset for training
        self.datasets['train'] = ChunkedTextDataset(
            train_tokens,
            self.hparams.max_length,
            precomputed_latents=self.precomputed_latents.get('train'),
            verify_latents_match=self.verify_latents_match,
        )

        self.datasets['val'] = ChunkedTextDataset(
            val_tokens,
            self.hparams.max_length,
            precomputed_latents=self.precomputed_latents.get('val'),
            verify_latents_match=self.verify_latents_match,
        )
        self.datasets['pred'] = self.datasets['val']

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['train'],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers or os.cpu_count(),
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['val'],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers or os.cpu_count(),
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['pred'],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers or os.cpu_count(),
        )
