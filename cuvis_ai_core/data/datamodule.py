"""SDK-free base DataModule and the registry-dispatch helper.

``BaseCuvisAIDataModule`` is the abstract, SDK-free contract every data module inherits
(mirrors how ``Node`` lives in core). It owns split resolution and the four
``*_dataloader()`` methods, so a concrete plugin module implements only the
format-specific reader plus:

* ``enumerate(required_attrs)`` -> the attributed ``SampleRef`` universe, and
* ``build_dataset_from_refs(refs)`` -> a torch ``Dataset`` for a resolved subset,

for the selector path (``DataConfig.splits`` set), or ``build_stage_dataset(stage)`` for a
module that owns its split semantics (``DataConfig.splits is None``).

Core ships **no** concrete DataModules; every DataModule comes from a plugin manifest
(see the ``cuvis-ai-dataloader`` plugin).
"""

from __future__ import annotations

from abc import ABC
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cuvis_ai_schemas.training.data import DataConfig, DataSplitConfig, SampleRef


class DataStage(StrEnum):
    """The Lightning ``DataModule.setup`` stages.

    These are the values Lightning passes to ``setup(stage)`` (plus ``None`` = all). They
    are distinct from ``cuvis_ai_schemas.enums.ExecutionStage`` (node-execution filtering:
    ``train``/``val``/``inference``); these mirror Lightning's ``fit``/``validate``/
    ``test``/``predict``. As a ``StrEnum``, a member compares equal to the raw string
    Lightning passes, so the branches below work whether ``setup`` is called with a member
    or a plain string.
    """

    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


class _RepeatDataset(Dataset):
    """Present ``base`` as ``n`` times longer, mapping ``idx -> base[idx % len(base)]``.

    Backs ``samples_per_frame``: each base sample is visited ``n`` times per epoch.
    Datasets return raw frames (cropping/augmentation happens downstream in the
    pipeline), so the repeated visits become independent transformed samples; a
    shuffled loader spreads a frame's ``n`` occurrences across the epoch. No data is
    copied — only indices are remapped.
    """

    def __init__(self, base: Dataset, n: int) -> None:
        if int(n) < 1:
            raise ValueError(f"repeat factor must be >= 1, got {n}")
        self._base = base
        self._n = int(n)

    def __len__(self) -> int:
        return len(self._base) * self._n  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Any:
        return self._base[idx % len(self._base)]  # type: ignore[arg-type]


class BaseCuvisAIDataModule(pl.LightningDataModule, ABC):
    """Shared split + dataloader plumbing for every data module.

    Concrete plugin modules set ``DATA_MODULE_NAME`` and implement ``validate_params()``
    plus either (``enumerate`` + ``build_dataset_from_refs``) for selector-driven splits or
    ``build_stage_dataset`` for module-owned splits. The base resolves selectors, runs the
    leakage validator, and serves the four dataloaders, so the subclass never re-implements
    that.
    """

    #: Unique registry key; the manifest ``data_module_name`` must equal it.
    DATA_MODULE_NAME: ClassVar[str] = ""

    def __init__(
        self,
        *,
        splits: DataSplitConfig | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        samples_per_frame: int = 1,
        **params: Any,
    ) -> None:
        super().__init__()
        # Within-epoch patch multiplicity: repeat each TRAIN sample this many times
        # per epoch (N independent downstream crops/frame). Loader-agnostic, applied
        # only to the train loader (see ``train_dataloader``).
        if int(samples_per_frame) < 1:
            raise ValueError(f"samples_per_frame must be >= 1, got {samples_per_frame}")
        self.samples_per_frame = int(samples_per_frame)
        # Coerce a plain dict / OmegaConf mapping (e.g. from `DataModule(**cfg.data)`)
        # into a DataSplitConfig so config-driven construction works uniformly.
        from cuvis_ai_schemas.training.data import DataSplitConfig as _DataSplitConfig

        if splits is not None and not isinstance(splits, _DataSplitConfig):
            splits = _DataSplitConfig(**dict(splits))
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.params = params
        self._refs: list[SampleRef] | None = None
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: Dataset | None = None
        self._predict_ds: Dataset | None = None

    # -- public per-stage dataset accessors ------------------------------------
    # Consumers read ``len(dm.predict_ds)`` / ``dm.train_ds.wavelengths_nm`` and sometimes
    # reassign (e.g. ``dm.predict_ds = Subset(...)``), so they are settable.
    @property
    def train_ds(self) -> Dataset | None:
        """The training dataset built by ``setup`` (``None`` until then)."""
        return self._train_ds

    @train_ds.setter
    def train_ds(self, value: Dataset | None) -> None:
        self._train_ds = value

    @property
    def val_ds(self) -> Dataset | None:
        """The validation dataset built by ``setup`` (``None`` until then)."""
        return self._val_ds

    @val_ds.setter
    def val_ds(self, value: Dataset | None) -> None:
        self._val_ds = value

    @property
    def test_ds(self) -> Dataset | None:
        """The test dataset built by ``setup`` (``None`` until then)."""
        return self._test_ds

    @test_ds.setter
    def test_ds(self, value: Dataset | None) -> None:
        self._test_ds = value

    @property
    def predict_ds(self) -> Dataset | None:
        """The predict dataset built by ``setup`` (``None`` until then)."""
        return self._predict_ds

    @predict_ds.setter
    def predict_ds(self, value: Dataset | None) -> None:
        self._predict_ds = value

    # -- subclass contract -----------------------------------------------------
    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Module-specific arg validation (required keys present, paths exist).

        Pure stdlib, no heavy imports. Default is a no-op; override to validate.
        """

    def enumerate(
        self, required_attrs: frozenset[str] = frozenset()
    ) -> list[SampleRef]:
        """Selector path: return the attributed ``SampleRef`` universe, canonically ordered.

        ``required_attrs`` (a subset of ``{"tags", "category_ids"}``) tells the module which
        metadata to populate; when empty, skip COCO / PNG parsing. Order must be stable
        (sort by ``source`` then ``index``) so positional ``dir_indices`` are reproducible.
        Default raises; override for a selector-driven module.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: implement enumerate() + build_dataset_from_refs() for "
            f"selector splits (DataConfig.splits), or build_stage_dataset() for module-owned splits."
        )

    def build_dataset_from_refs(self, refs: list[SampleRef]) -> Dataset:
        """Selector path: return the torch ``Dataset`` for an already-resolved subset.

        This is the one place heavy libs load (it reads cubes + attaches labels). Default
        raises; override for a selector-driven module.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: implement build_dataset_from_refs() for selector splits."
        )

    def category_name_to_id(self) -> dict[str, int] | None:
        """Map a ``categories`` selector's names to ids (``None`` if the module has no COCO).

        Default ``None``; override in a module whose labeler exposes a category map.
        """
        return None

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Module-owned-splits hook, called per stage when ``DataConfig.splits`` is None.

        Default raises; a module that owns its split semantics overrides this instead of
        ``enumerate`` / ``build_dataset_from_refs``.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: provide DataConfig.splits (selectors) + enumerate/"
            f"build_dataset_from_refs, or override build_stage_dataset to own splits."
        )

    # -- lightning hooks -------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if self.splits is not None:
            self._setup_from_selectors(stage)
        else:
            self._setup_module_owned(stage)

    def _enumerate_once(self, wanted: frozenset[str]) -> list[SampleRef]:
        if self._refs is None:
            self._refs = self.enumerate(wanted)
        return self._refs

    def _setup_from_selectors(self, stage: str | None) -> None:
        from cuvis_ai_core.data.selectors import (
            required_attrs,
            resolve_selectors,
            validate_leakage,
        )
        from cuvis_ai_core.data.splits_io import verify_universe

        splits = self.splits
        assert splits is not None
        refs = self._enumerate_once(required_attrs(splits))
        verify_universe(splits, refs)
        name_to_id = self.category_name_to_id()

        def resolve(stage_selectors: list[Any]) -> list[SampleRef]:
            if not stage_selectors:
                return []
            return resolve_selectors(stage_selectors, refs, name_to_id=name_to_id)

        # Leakage is a global property; resolve train/val/test and check before building.
        train = resolve(splits.train)
        val = resolve(splits.val)
        test = resolve(splits.test)
        validate_leakage(train, val, test, mode=splits.leakage_check)

        if stage in (DataStage.FIT, None) and train:
            self._train_ds = self.build_dataset_from_refs(train)
        if stage in (DataStage.FIT, DataStage.VALIDATE, None) and val:
            self._val_ds = self.build_dataset_from_refs(val)
        if stage in (DataStage.TEST, None) and test:
            self._test_ds = self.build_dataset_from_refs(test)
        if stage in (DataStage.PREDICT, None):
            # Empty predict -> the whole universe (an inference run iterates all samples).
            predict = resolve(splits.predict) if splits.predict else list(refs)
            self._predict_ds = self.build_dataset_from_refs(predict)

    def _setup_module_owned(self, stage: str | None) -> None:
        if stage in (DataStage.FIT, None):
            self._train_ds = self.build_stage_dataset("train")
        if stage in (DataStage.FIT, DataStage.VALIDATE, None):
            self._val_ds = self.build_stage_dataset("val")
        if stage in (DataStage.TEST, None):
            self._test_ds = self.build_stage_dataset("test")
        if stage in (DataStage.PREDICT, None):
            self._predict_ds = self.build_stage_dataset("predict")

    def _loader(
        self, dataset: Dataset | None, *, shuffle: bool, name: str
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError(
                f"{type(self).__name__}: {name} dataset is not built; "
                f"call setup(stage={name!r}) (or setup() ) first."
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        # ``samples_per_frame`` multiplicity is applied here, train split only: the
        # returned loader's ``dataset`` is a ``_RepeatDataset`` of length
        # ``N * len(train_ds)``, while the ``train_ds`` property stays the unwrapped frame
        # count. Read multiplicity off the loader (``len(loader.dataset)`` / iteration),
        # never off ``train_ds``. A map-style repeat keeps this DDP-safe: Lightning's
        # automatic ``DistributedSampler`` shards the repeated dataset like any other.
        ds = self._train_ds
        if self.samples_per_frame > 1 and ds is not None:
            ds = _RepeatDataset(ds, self.samples_per_frame)
        return self._loader(ds, shuffle=True, name="train")

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val_ds, shuffle=False, name="val")

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test_ds, shuffle=False, name="test")

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self._predict_ds, shuffle=False, name="predict")


def create_data_module(registry: Any, data_config: DataConfig) -> BaseCuvisAIDataModule:
    """Build the DataModule named by ``data_config.data_module`` from a registry.

    ``registry`` is any object exposing a ``data_modules`` dict (e.g. ``NodeRegistry``).
    Looks up the class, validates the module-specific params, and constructs it from
    ``splits`` + ``batch_size`` + ``num_workers`` + ``params``.
    """
    data_modules = getattr(registry, "data_modules", {})
    cls = data_modules.get(data_config.data_module)
    if cls is None:
        raise ValueError(
            f"no plugin provides data module {data_config.data_module!r}; "
            f"available: {sorted(data_modules)}"
        )
    cls.validate_params(data_config.params)
    return cls(
        splits=data_config.splits,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        **data_config.params,
    )


__all__ = [
    "BaseCuvisAIDataModule",
    "DataStage",
    "create_data_module",
]
