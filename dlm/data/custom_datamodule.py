import importlib
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from dlm.data.custom_tokenizers import SimpleTokenizer
from dlm.utils import instantiate_from_config


class SimpleDiscreteDataset(Dataset):
    """Simple dataset for discrete diffusion that takes pre-created torch vectors.
    Each sample should be a sequence of token indices, optionally with labels.
    """

    def __init__(
        self,
        data: torch.Tensor,  # Shape: [num_samples, sequence_length]
        vocab_size: int,
        mask_token_id: int | None = None,
        labels: torch.Tensor | None = None,
    ):  # Shape: [num_samples] or [num_samples, label_dim]
        """Args:
        data: Tensor of token indices, shape [num_samples, sequence_length]
        vocab_size: Size of vocabulary
        mask_token_id: Token ID used for masking (will be set to vocab_size if None)
        labels: Optional tensor of labels for each sequence

        """
        self.data = data
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size
        self.labels = labels

        # Validate data
        assert data.dim() == 2, 'Data should be 2D: [num_samples, sequence_length]'
        assert data.max() < self.vocab_size, f'Data contains tokens >= vocab_size ({self.vocab_size})'
        assert data.min() >= 0, 'Data contains negative token indices'

        # Validate labels if provided
        if labels is not None:
            assert len(labels) == len(data), f'Labels length ({len(labels)}) should match data length ({len(data)})'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        result = {
            'input_ids': self.data[idx].long(),
        }
        # Add labels if available (using 'latent' key)
        if self.labels is not None:
            result['latent'] = self.labels[idx].float()

        return result


class CustomDataModule(L.LightningDataModule):
    """Custom data module for simple discrete diffusion."""

    def __init__(
        self,
        data_path: str,
        vocab_size: int = 1000,
        sequence_length: int = 12,
        mask_token_id: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        custom_data_function: str | None = None,
        custom_data_kwargs: dict[str, Any] | None = None,
        regenerate: bool = False,
        regenerate_each_epoch: bool = False,
        tokenizer_config: dict[str, Any] | None = None,
    ):
        """Args:
        data_path: Path to data file (ignored if regenerate=True)
        vocab_size: Size of vocabulary
        sequence_length: Length of each sequence
        mask_token_id: Mask token ID (defaults to vocab_size)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_split: Fraction of train_data to use for validation if val_data not provided
        custom_data_function: Function name to use for data generation (e.g., "binary_sawtooth")
        custom_data_kwargs: Keyword arguments to pass to the data generation function
        regenerate: If True, regenerate data using custom_data_function instead of loading from data_path
        regenerate_each_epoch: If True, regenerate data at the start of each epoch (requires regenerate=True)

        """
        super().__init__()
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.custom_data_function = custom_data_function
        self.custom_data_kwargs = custom_data_kwargs or {}
        self.regenerate = regenerate
        self.regenerate_each_epoch = regenerate_each_epoch

        # Instantiate tokenizer first (needed for data generation)
        if tokenizer_config is not None:
            self.tokenizer = instantiate_from_config(
                tokenizer_config,
                vocab_size=self.vocab_size,
                mask_token_id=self.mask_token_id,
            )
        else:
            self.tokenizer = SimpleTokenizer(self.vocab_size, self.mask_token_id)

        if self.regenerate_each_epoch or self.regenerate:
            self._regenerate_data()
        else:
            self._load_data_from_file()

    def _load_data_from_file(self):
        """Load data from the specified file path."""
        print(f'Loading data from: {self.data_path}')
        # security scan: make sure to use weights_only=True if you are loading from an untrusted source
        # In this case, we trust the user's data path
        try:
            data_dict = torch.load(self.data_path, weights_only=True)
        except (AttributeError, TypeError):
            # if the file was saved with an older version of torch without weights_only
            data_dict = torch.load(self.data_path)

        if isinstance(data_dict, dict):
            train_data = data_dict['train']
            val_data = data_dict.get('val', None)
            # Load labels if available
            train_labels = data_dict.get('train_labels', None)
            val_labels = data_dict.get('val_labels', None)
        else:
            train_data = data_dict
            val_data = None
            train_labels = None
            val_labels = None
        # Basic validation
        assert train_data.max() < self.vocab_size, f'Data contains tokens >= vocab_size ({self.vocab_size})'
        assert train_data.min() >= 0, 'Data contains negative tokens'
        assert train_data.shape[1] == self.sequence_length, (
            f'Data sequence length ({train_data.shape[1]}) does not match config sequence length ({self.sequence_length})'
        )

        self.train_data = train_data
        self.val_data = val_data
        self.train_labels = train_labels
        self.val_labels = val_labels

    def _regenerate_data(self):
        """Regenerate data using the specified custom data function."""
        if not self.custom_data_function:
            msg = 'custom_data_function must be specified when regenerate=True'
            raise ValueError(msg)

        # Dynamically import and get the data generation function
        try:
            # Split module path and function name
            if '.' in self.custom_data_function:
                module_path, func_name = self.custom_data_function.rsplit('.', 1)
                module = importlib.import_module(module_path)
                data_func = getattr(module, func_name)
            else:
                # If no module path, assume it's in the current module
                data_func = globals().get(self.custom_data_function)
                if data_func is None:
                    raise AttributeError(f"Function '{self.custom_data_function}' not found in current module")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import function '{self.custom_data_function}': {e}")

        # Verify that the imported object is callable
        if not callable(data_func):
            raise ValueError(f"'{self.custom_data_function}' is not a callable function")

        # Set default arguments and override with custom_data_kwargs
        default_kwargs = {
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
        }
        kwargs = {**default_kwargs, **self.custom_data_kwargs}

        # Generate data
        try:
            result = data_func(**kwargs)
            # possible encode data from string to token indices
            if hasattr(self.tokenizer, 'encode'):
                result = self.tokenizer.encode(result)
            if isinstance(result, tuple) and len(result) == 2:
                train_data, train_labels = result
            else:
                train_data = result
                train_labels = None
        except Exception as e:
            raise RuntimeError(f"Error calling data function '{self.custom_data_function}': {e}")

        # Ensure train_data is a tensor and convert to long tensor for token indices
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.tensor(train_data)
        train_data = train_data.long()

        # Basic validation
        assert train_data.max() < self.vocab_size, f'Generated data contains tokens >= vocab_size ({self.vocab_size})'
        assert train_data.min() >= 0, 'Generated data contains negative tokens'
        assert train_data.shape[1] == self.sequence_length, (
            f'Generated data sequence length ({train_data.shape[1]}) does not match config sequence length ({self.sequence_length})'
        )

        self.train_data = train_data
        self.val_data = None  # Will be split in setup()
        self.train_labels = train_labels
        self.val_labels = None  # Will be split in setup()

    def regenerate_data(self):
        """Regenerate data and rebuild datasets."""
        if not self.regenerate_each_epoch:
            print('regenerate is set to False. Cannot regenerate data.')
        else:
            self._regenerate_data()
            self.setup()

    def setup(self, stage: str | None = None):
        if self.val_data is None:
            # Split train_data into train and val
            num_val = int(len(self.train_data) * self.val_split)
            indices = torch.randperm(len(self.train_data))
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]

            val_data = self.train_data[val_indices]
            train_data = self.train_data[train_indices]

            # Also split labels if available
            if self.train_labels is not None:
                val_labels = self.train_labels[val_indices]
                train_labels = self.train_labels[train_indices]
            else:
                val_labels = None
                train_labels = None
        else:
            train_data = self.train_data
            val_data = self.val_data
            train_labels = self.train_labels
            val_labels = self.val_labels

        self.train_dataset = SimpleDiscreteDataset(
            train_data,
            self.vocab_size,
            self.mask_token_id,
            train_labels,
        )
        self.val_dataset = SimpleDiscreteDataset(
            val_data,
            self.vocab_size,
            self.mask_token_id,
            val_labels,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
