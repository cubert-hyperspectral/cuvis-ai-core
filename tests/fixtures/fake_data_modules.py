"""Fake DataModule classes for registry-dispatch + base-class tests.

Importable by FQCN so ``register_preinstalled`` can resolve them like a real
plugin's provides entries, without any SDK or plugin install.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseHyperspectralDataModule


class _TinyDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {"x": torch.zeros(2), "idx": idx}


class FakeDataModule(BaseHyperspectralDataModule):
    """Minimal id-list + module-owned DataModule for tests."""

    DATA_MODULE_NAME = "fake"

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        if "required_key" in params and not params["required_key"]:
            raise ValueError("required_key must be truthy")

    def build_dataset(self, ids) -> Dataset:
        return _TinyDataset(len(ids) if ids else 4)

    def build_stage_dataset(self, stage: str) -> Dataset:
        return _TinyDataset(3)


class NotADataModule:
    """A non-DataModule class to prove kind=data_module routing rejects it."""

    DATA_MODULE_NAME = "bad"
