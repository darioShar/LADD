import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from dlm.data.custom_datasets import binary_sawtooth
from dlm.data.custom_tokenizers import SimpleTokenizer

from ..utils import print_rank_zero


class BinarySawtoothDataset(Dataset):
    """Dataset for binary sawtooth sequences with optional  latents.
    Similar to ChunkedTextDataset but for binary sawtooth data.
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        latents: torch.Tensor | None = None,
    ):
        """
        Args:
            sequences: Tensor of shape [num_samples, sequence_length] with binary sequences
            latents: Tensor of shape [num_samples, 1] with original latents from generation
        """
        super().__init__()
        self.sequences = sequences
        self.latents = latents

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {'input_ids': self.sequences[idx]}
        # use latents from generation
        if self.latents is not None:
            item['latent'] = self.latents[idx]

        return item


class BinarySawtoothDataModule(L.LightningDataModule):
    """
    DataModule for binary sawtooth dataset with latent support.
    Similar to Text8DataModule but for synthetic binary sawtooth data.
    """

    def __init__(
        self,
        batch_size: int = 512,
        sequence_length: int = 64,
        num_samples_train: int = 10000,
        num_samples_val: int = 1000,
        num_workers: int | None = None,
        # Binary sawtooth specific parameters
        num_saws: int = 2,
        scale: float = 0.05,
        uniform_shift: bool = True,
        seed: int | None = 42,
        regenerate_each_epoch: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = SimpleTokenizer(mask_token_id=2)
        self.datasets: dict[str, Dataset] = {}
        self._generated_epoch: int | None = None
        self._data_signatures: dict[str, tuple[int, int, float, float]] = {}

    def _generate_data(self, epoch: int = 0):
        """Internal method to generate data, seeded by the epoch number."""
        base_seed = self.hparams.seed

        # Use the epoch to generate a new, unique seed for this epoch's data
        train_seed = base_seed + epoch if base_seed is not None else None
        val_seed = base_seed + epoch + 10000000 if base_seed is not None else None  # Use a large offset for val seed

        train_sequences, train_latents = binary_sawtooth(
            num_samples=self.hparams.num_samples_train,
            sequence_length=self.hparams.sequence_length,
            vocab_size=2,
            num_saws=self.hparams.num_saws,
            scale=self.hparams.scale,
            uniform_shift=self.hparams.uniform_shift,
            seed=train_seed,
        )

        val_sequences, val_latents = binary_sawtooth(
            num_samples=self.hparams.num_samples_val,
            sequence_length=self.hparams.sequence_length,
            vocab_size=2,
            num_saws=self.hparams.num_saws,
            scale=self.hparams.scale,
            uniform_shift=self.hparams.uniform_shift,
            seed=val_seed,
        )

        train_signature = self._compute_signature(train_sequences, train_latents)
        val_signature = self._compute_signature(val_sequences, val_latents)
        self._log_signature('train', train_signature, epoch)
        self._log_signature('val', val_signature, epoch)

        if 'train' in self.datasets:
            self.datasets['train'].sequences = train_sequences
            self.datasets['train'].latents = train_latents
        else:
            self.datasets['train'] = BinarySawtoothDataset(train_sequences, train_latents)

        if 'val' in self.datasets:
            self.datasets['val'].sequences = val_sequences
            self.datasets['val'].latents = val_latents
        else:
            self.datasets['val'] = BinarySawtoothDataset(val_sequences, val_latents)

        self.datasets['pred'] = self.datasets['val']
        self._generated_epoch = epoch
        self._data_signatures['train'] = train_signature
        self._data_signatures['val'] = val_signature

    def prepare_data(self) -> None:
        """Generate binary sawtooth data. No file I/O needed for synthetic data."""

    # def regenerate_data(self):
    #     if self.hparams.regenerate_each_epoch:
    #         self.datasets = {}
    #         self.setup()

    def setup(self, stage: str | None = None) -> None:
        """Generate data and create datasets."""
        self._generate_data(epoch=0)

    def on_train_epoch_start(self):
        """Hook to regenerate data at the start of each training epoch."""
        current_epoch = self.trainer.current_epoch if self.trainer is not None else 0
        self._maybe_regenerate_data(current_epoch)

    def _maybe_regenerate_data(self, epoch: int) -> None:
        if not self.hparams.regenerate_each_epoch:
            return
        if self._generated_epoch == epoch:
            return
        if epoch > 0:
            print_rank_zero(f'Regenerating data for epoch {epoch}...')
        self._generate_data(epoch=epoch)

    def _compute_signature(
        self,
        sequences: torch.Tensor,
        latents: torch.Tensor | None,
        sample_size: int = 256,
    ) -> tuple[int, int, float, float]:
        sample_size = min(sample_size, sequences.shape[0])
        seq_sample = sequences[:sample_size].to(dtype=torch.int64)
        seq_positions = torch.arange(seq_sample.shape[1], dtype=torch.int64)
        token_sum = int(seq_sample.sum().item())
        weighted_token_sum = int((seq_sample * seq_positions).sum().item())

        if latents is None:
            return (token_sum, weighted_token_sum, 0.0, 0.0)

        latent_sample = latents[:sample_size].float()
        latent_weights = torch.arange(latent_sample.numel(), dtype=latent_sample.dtype)
        latent_sum = float(latent_sample.sum().item())
        weighted_latent_sum = float((latent_sample.view(-1) * latent_weights).sum().item())
        return (token_sum, weighted_token_sum, round(latent_sum, 6), round(weighted_latent_sum, 6))

    def _format_signature(self, signature: tuple[int, int, float, float]) -> str:
        token_sum, weighted_token_sum, latent_sum, weighted_latent_sum = signature
        return (
            f'tokens={token_sum} weighted_tokens={weighted_token_sum} '
            f'latents={latent_sum:.6f} weighted_latents={weighted_latent_sum:.6f}'
        )

    def _log_signature(self, split: str, signature: tuple[int, int, float, float], epoch: int) -> None:
        previous = self._data_signatures.get(split)
        if epoch > 0:
            if previous == signature:
                print_rank_zero(
                    f'Warning: {split} data signature unchanged at epoch {epoch}: '
                    f'{self._format_signature(signature)}',
                )
            else:
                print_rank_zero(f'{split} data signature at epoch {epoch}: {self._format_signature(signature)}')

    def train_dataloader(self) -> DataLoader:
        current_epoch = self.trainer.current_epoch if self.trainer is not None else 0
        self._maybe_regenerate_data(current_epoch)
        persistent = (self.hparams.num_workers or 0) > 0 and (not self.hparams.regenerate_each_epoch)
        num_workers = (self.hparams.num_workers or 0) if not self.hparams.regenerate_each_epoch else 0
        return DataLoader(
            self.datasets['train'],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent,
            pin_memory=not self.hparams.regenerate_each_epoch,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        current_epoch = self.trainer.current_epoch if self.trainer is not None else 0
        self._maybe_regenerate_data(current_epoch)
        persistent = (self.hparams.num_workers or 0) > 0 and (not self.hparams.regenerate_each_epoch)
        num_workers = (self.hparams.num_workers or 0) if not self.hparams.regenerate_each_epoch else 0
        return DataLoader(
            self.datasets['val'],
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            persistent_workers=persistent,
            pin_memory=not self.hparams.regenerate_each_epoch,
            drop_last=True,
        )

    def predict_dataloader(self) -> DataLoader:
        num_workers = (self.hparams.num_workers or 0) if not self.hparams.regenerate_each_epoch else 0
        return DataLoader(
            self.datasets['pred'],
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
        )
