"""Fake DataModule classes for registry-dispatch + base-class tests.

Importable by FQCN so ``register_plugins_installed`` can resolve them like a real plugin's
provides entries, without any SDK or plugin install.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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


class _FakeCu3sDataset(Dataset):
    """Deterministic synthetic hyperspectral frames keyed by each ref's universe index."""

    def __init__(
        self,
        refs: list,
        *,
        height: int,
        width: int,
        channels: int,
        wavelengths: torch.Tensor,
        seed: int,
    ) -> None:
        """Store the resolved refs and the generation parameters."""
        self._refs = list(refs)
        self._height = height
        self._width = width
        self._channels = channels
        self._wavelengths = wavelengths
        self._seed = seed

    def __len__(self) -> int:
        """Number of resolved refs in this subset."""
        return len(self._refs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate the frame for ``refs[idx]``, seeded by its universe index.

        Seeding per universe index (not subset position) keeps a frame's content
        identical no matter which split it lands in.
        """
        ref = self._refs[idx]
        generator = torch.Generator().manual_seed(self._seed + ref.index)
        cube = (
            torch.rand((self._height, self._width, self._channels), generator=generator)
            * 1000.0
        ).to(torch.uint16)
        return {"cube": cube, "wavelengths": self._wavelengths.clone()}


class FakeCu3sDataModule(BaseCuvisAIDataModule):
    """Synthetic stand-in for the dataloader plugin's ``cu3s`` module.

    Core's own test pipelines and trainruns select ``data_module: cu3s``, but the
    suite installs neither the cuvis SDK nor cuvis-ai-dataloader. The
    ``cuvis_ai_test_nodes`` manifest exposes this module under that name so the
    orchestrated Train path has a real dispatch target: a fixed universe of
    ``num_measurements`` frames of one source (the ``cu3s_file_path`` param,
    echoed verbatim into ``SampleRef.source`` so ``file_indices`` selectors
    match), served as deterministic uint16 cubes with int32 wavelengths — the
    batch layout ``LentilsAnomalyDataNode`` consumes.
    """

    DATA_MODULE_NAME = "cu3s"

    def __init__(
        self,
        *,
        cu3s_file_path: str = "",
        annotation_json_path: str = "",
        processing_mode: str = "Raw",
        num_measurements: int = 7,
        height: int = 64,
        width: int = 64,
        channels: int = 61,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Accept the real cu3s module's params plus synthetic-shape knobs."""
        super().__init__(**kwargs)
        self.cu3s_file_path = str(cu3s_file_path)
        self.annotation_json_path = str(annotation_json_path)
        self.processing_mode = processing_mode
        self._num_measurements = num_measurements
        self._height = height
        self._width = width
        self._channels = channels
        self._seed = seed
        self._wavelengths = torch.from_numpy(
            np.linspace(430.0, 910.0, channels).astype(np.int32)
        )

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """All params are optional; the synthetic module reads nothing from disk."""

    def enumerate(self, required_attrs: frozenset[str] = frozenset()) -> list:
        """Fixed universe over the configured source; even indices are normal."""
        from cuvis_ai_schemas.training.data import SampleRef

        refs: list[SampleRef] = []
        for i in range(self._num_measurements):
            is_normal = i % 2 == 0
            refs.append(
                SampleRef(
                    source=self.cu3s_file_path,
                    index=i,
                    label_id=i,
                    tags=(["normal"] if is_normal else ["anomalous"])
                    if "tags" in required_attrs
                    else [],
                    category_ids=([] if is_normal else [1])
                    if "category_ids" in required_attrs
                    else [],
                )
            )
        return refs

    def category_name_to_id(self) -> dict[str, int]:
        """Static two-class map standing in for the COCO annotation categories."""
        return {"normal": 0, "anomalous": 1}

    def build_dataset_from_refs(self, refs: list) -> Dataset:
        """Serve deterministic uint16 cube + int32 wavelength batches for the subset."""
        return _FakeCu3sDataset(
            refs,
            height=self._height,
            width=self._width,
            channels=self._channels,
            wavelengths=self._wavelengths,
            seed=self._seed,
        )


class NotADataModule:
    """A non-DataModule class to prove kind=data_module routing rejects it."""

    DATA_MODULE_NAME = "bad"
