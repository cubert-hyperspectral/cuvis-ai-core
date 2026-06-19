"""Fake DataModule classes for registry-dispatch + base-class tests.

Importable by FQCN so ``register_preinstalled`` can resolve them like a real plugin's
provides entries, without any SDK or plugin install.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule


class _TinyDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {"x": torch.zeros(2), "idx": idx}


class FakeDataModule(BaseCuvisAIDataModule):
    """Minimal selector + module-owned DataModule for tests.

    Enumerates a fixed universe of ``n`` measurements of one source ``fake.cu3s``: even
    indices are tagged ``normal`` (no category), odd indices ``scrap`` (category id 1).
    """

    DATA_MODULE_NAME = "fake"

    def __init__(self, *, n: int = 6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._n = n

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        if "required_key" in params and not params["required_key"]:
            raise ValueError("required_key must be truthy")

    def enumerate(self, required_attrs: frozenset[str] = frozenset()) -> list:
        from cuvis_ai_schemas.training.data import SampleRef

        refs: list[SampleRef] = []
        for i in range(self._n):
            is_normal = i % 2 == 0
            refs.append(
                SampleRef(
                    source="fake.cu3s",
                    index=i,
                    label_id=i,
                    tags=(["normal"] if is_normal else ["scrap"])
                    if "tags" in required_attrs
                    else [],
                    category_ids=([] if is_normal else [1])
                    if "category_ids" in required_attrs
                    else [],
                )
            )
        return refs

    def category_name_to_id(self) -> dict[str, int]:
        return {"normal": 0, "scrap": 1}

    def build_dataset_from_refs(self, refs: list) -> Dataset:
        return _TinyDataset(len(refs))

    def build_stage_dataset(self, stage: str) -> Dataset:
        return _TinyDataset(3)


class NotADataModule:
    """A non-DataModule class to prove kind=data_module routing rejects it."""

    DATA_MODULE_NAME = "bad"
