from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import load_dataset
from datasets.download.download_config import DownloadConfig
from omegaconf import OmegaConf


def load_dataset_with_config(
    path: str,
    name: str | None = None,
    download_config: DownloadConfig | Mapping[str, Any] | None = None,
    **kwargs,
):
    if OmegaConf.is_config(download_config):
        download_config = OmegaConf.to_container(download_config, resolve=True)
    if isinstance(download_config, Mapping):
        download_config = DownloadConfig(**download_config)
    return load_dataset(path=path, name=name, download_config=download_config, **kwargs)
