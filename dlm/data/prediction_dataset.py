import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class PredictionDataset(Dataset):
    """
    Corrected placeholder dataset for prediction.
    It returns a single placeholder tensor for each item.
    """

    def __init__(self, num_data: int, sequence_length: int):
        """
        Args:
            num_data: Total number of placeholder samples in the dataset.
            sequence_length: The sequence length of each placeholder tensor.
        """
        self.num_data = num_data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_data

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a single placeholder tensor. The DataLoader will batch these.
        """
        # The returned item should have a shape of [sequence_length]
        return torch.zeros(self.sequence_length, dtype=torch.long)


class PredictionDataModule(L.LightningDataModule):
    """
    A LightningDataModule for the PredictionDataset.

    This DataModule is designed to create a dataloader that yields
    placeholder batches, which is useful for triggering a prediction or
    generation loop a specific number of times.
    """

    def __init__(
        self,
        num_data: int,
        sequence_length: int,
        batch_size: int,
        num_workers: int,
    ):
        """
        Args:
            num_data: Total number of placeholder samples to generate.
            sequence_length: The sequence length of each placeholder tensor.
            batch_size: How many samples per batch to load.
            num_workers: How many subprocesses to use for data loading.
        """
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        self.dataset = None

    def setup(self, stage: str):
        """
        Instantiate the dataset. This is called by Lightning automatically.
        We only set up the dataset for the 'predict' stage.
        """
        if stage == 'predict':
            self.dataset = PredictionDataset(
                num_data=self.hparams.num_data,
                sequence_length=self.hparams.sequence_length,
            )

    def predict_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the prediction loop.
        """
        if self.dataset is None:
            raise RuntimeError(
                "The setup for the 'predict' stage has not been called. Did you call trainer.predict(...)?",
            )

        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
