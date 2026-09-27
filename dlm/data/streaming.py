import os
import json

import torch

import lightning as L
import litdata as ld
from transformers import PreTrainedTokenizerBase
from ..utils import instantiate_from_config_hf_pretrained, instantiate_from_config


class SFTStreamingDataset(ld.StreamingDataset):
    def __getitem__(self, index):
        example = super().__getitem__(index)
        if "messages" in example:
            example["messages"] = json.loads(example["messages"])
        return example


class CollateFnWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        # transform accepts dict of list instead of list of dict
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = [example[key] for example in batch]
        return self.transform(batch_dict)


class StreamingDataModule(L.LightningDataModule):
    def __init__(
        self,
        path: str | list[str],
        weights: list[float] | None = None,
        data_kwargs: dict | None = None,
        tokenizer_config: dict | None = None,
        transform_config: dict | None = None,
        batch_size: int = 64,
        eval_batch_size: int | None = None,
        num_workers: int | None = None,
        val_size: float | None = None,
    ):
        super().__init__()
        if isinstance(path, str):
            path = [path]
        self.path = path
        self.weights = weights
        self.data_kwargs = data_kwargs
        self.batch_size = batch_size
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None else batch_size
        )
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        assert val_size is None or val_size < 1, "val_size must be less than 1"
        self.val_size = val_size
        self.tokenizer = (
            instantiate_from_config_hf_pretrained(tokenizer_config)
            if tokenizer_config is not None
            else None
        )
        self.collate_fn = None
        if transform_config is not None:
            self.collate_fn = CollateFnWrapper(
                instantiate_from_config(transform_config, tokenizer=self.tokenizer)
            )
        self.datasets = {}

    def setup(self, stage=None):
        train_datasets = []
        val_datasets = []
        val_ds = None
        for path in self.path:
            train_ds = SFTStreamingDataset(input_dir=path, **self.data_kwargs)
            if self.val_size is not None:
                train_ds, val_ds = ld.train_test_split(
                    train_ds, splits=[1 - self.val_size, self.val_size]
                )
                val_datasets.append(val_ds)
            train_datasets.append(train_ds)
        if len(train_datasets) > 1:
            train_ds = ld.CombinedStreamingDataset(
                train_datasets,
                weights=self.weights,
                iterate_over_all=self.weights is None,
            )
        else:
            train_ds = train_datasets[0]
        if len(val_datasets) > 1:
            val_ds = ld.CombinedStreamingDataset(
                val_datasets,
                weights=self.weights,
                iterate_over_all=self.weights is None,
            )
        elif self.val_size is not None:
            val_ds = val_datasets[0]
        train_ds.set_shuffle(True)
        self.datasets["train"] = train_ds
        if val_ds is not None:
            val_ds.set_shuffle(False)
            self.datasets["val"] = val_ds

    def train_dataloader(self):
        return ld.StreamingDataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return ld.StreamingDataLoader(
            self.datasets["val"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def state_dict(self):
        return self.trainer.train_dataloader.state_dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


class StreamingDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        assert self.tokenizer.bos_token_id is not None, "bos_token_id is not set"
        assert self.tokenizer.eos_token_id is not None, "eos_token_id is not set"

    def __call__(self, batch: list[torch.Tensor]):
        dtype = batch[0].dtype
        # append bos and eos tokens to the input text
        bos_token_id = torch.tensor([self.tokenizer.bos_token_id], dtype=dtype)
        eos_token_id = torch.tensor([self.tokenizer.eos_token_id], dtype=dtype)
        input_ids = [torch.cat([bos_token_id, seq, eos_token_id]) for seq in batch]
        return {"input_ids": torch.stack(input_ids)}


class StreamingDataModuleForPT(L.LightningDataModule):
    def __init__(
        self,
        train_config: dict,
        tokenizer_config: dict,
        max_length: int,
        batch_size: int,
        val_config: dict | None = None,
        eval_batch_size: int | None = None,
        num_workers: int = 0,
        val_size: float | int | None = None,
    ):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.tokenizer_config = tokenizer_config
        self.max_length = max_length
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.datasets = {}
        self._state_dict = None
        self.tokenizer = instantiate_from_config_hf_pretrained(self.tokenizer_config)
        self.collate_fn = StreamingDataCollator(self.tokenizer)

    def update_config(self, config: dict):
        config["item_loader"] = ld.TokensLoader(self.max_length - 2)
        config["drop_last"] = True
        return config

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = ld.StreamingDataset(**self.update_config(self.train_config))
        if self.val_config or self.val_size is not None:
            self.val_dataloader = self._val_dataloader
            if self.val_config:
                self.datasets["val"] = ld.StreamingDataset(
                    **self.update_config(self.val_config)
                )
            else:
                if isinstance(self.val_size, float):
                    assert self.val_size < 1, "val_size must be less than 1"
                    val_ratio = self.val_size
                else:
                    val_ratio = self.val_size / len(dataset)
                self.datasets["train"], self.datasets["val"] = ld.train_test_split(
                    dataset, splits=[1 - val_ratio, val_ratio]
                )
                self.datasets["val"] = self.datasets["val"].set_shuffle(False)
        else:
            self.datasets["train"] = dataset

    def train_dataloader(self):
        dataloader = ld.StreamingDataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        if self._state_dict is not None:
            self.trainer.print(f"Resuming dataloader {self._state_dict}")
            dataloader.load_state_dict(self._state_dict)
            self.trainer.print("Resumed dataloader successfully")
            self._state_dict = None

    def _val_dataloader(self):
        return ld.StreamingDataLoader(
            self.datasets["val"],
            batch_size=self.eval_batch_size or self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def state_dict(self):
        return self.trainer.train_dataloader.state_dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict
