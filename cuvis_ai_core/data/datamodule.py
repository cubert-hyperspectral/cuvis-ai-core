"""SDK-free base DataModule and the registry-dispatch helper.

``BaseHyperspectralDataModule`` is the abstract, SDK-free contract every data
module inherits (mirrors how ``Node`` lives in core). It owns the split-to-stage
mapping and the four ``*_dataloader()`` methods so a concrete plugin module
implements only the format-specific reader (``build_dataset`` for id-list splits,
or ``build_stage_dataset`` for module-owned splits) plus ``validate_params``.

Core ships **no** concrete DataModules; every DataModule comes from a plugin
manifest (see the ``cuvis-ai-dataloader`` plugin).
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_core.utils.general import expand_range_selectors

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cuvis_ai_schemas.training.data import DataConfig, DataSplitConfig


class BaseHyperspectralDataModule(pl.LightningDataModule, ABC):
    """Shared split + dataloader plumbing for every data module.

    Concrete plugin modules set ``DATA_MODULE_NAME`` and implement
    ``validate_params()`` plus either ``build_dataset()`` (id-list splits) or
    ``build_stage_dataset()`` (module-owned splits). The base applies the splits
    and serves the four dataloaders, so the subclass never re-implements that.
    """

    #: Unique registry key; the manifest ``data_module_name`` must equal it.
    DATA_MODULE_NAME: ClassVar[str] = ""

    def __init__(
        self,
        *,
        splits: DataSplitConfig | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **params: Any,
    ) -> None:
        super().__init__()
        # Coerce a plain dict / OmegaConf mapping (e.g. from `DataModule(**cfg.data)`)
        # into a DataSplitConfig so config-driven construction works uniformly.
        from cuvis_ai_schemas.training.data import DataSplitConfig as _DataSplitConfig

        if splits is not None and not isinstance(splits, _DataSplitConfig):
            splits = _DataSplitConfig(**dict(splits))
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.params = params
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: Dataset | None = None
        self._predict_ds: Dataset | None = None

    # -- subclass contract -----------------------------------------------------
    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Module-specific arg validation (required keys present, paths exist).

        Pure stdlib, no heavy imports. Default is a no-op; override to validate.
        """

    def build_dataset(self, ids: Sequence[int | str] | None) -> Dataset:
        """Id-list path: return the torch ``Dataset`` for the given sample ids.

        ``None`` means all samples (the ``predict`` case). Called per stage when
        ``DataConfig.splits`` is provided. This is the one place heavy libs load.
        Default raises; override for an id-list module.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: implement build_dataset for id-list splits "
            f"(DataConfig.splits), or build_stage_dataset for module-owned splits."
        )

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Module-owned-splits hook, called per stage when ``DataConfig.splits`` is None.

        Default raises; a module that owns its split semantics (e.g. cu3s_multi)
        overrides this instead of ``build_dataset``.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: provide DataConfig.splits (id-lists) and "
            f"build_dataset, or override build_stage_dataset to own splits."
        )

    # -- lightning hooks -------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if self.splits is not None:
            self._setup_from_splits(stage)
        else:
            self._setup_module_owned(stage)

    def _setup_from_splits(self, stage: str | None) -> None:
        # Range strings ("0-100", "0-10:2") in any id list expand to ints here, so
        # every id-list module gets range selectors without its own parsing.
        splits = self.splits
        assert splits is not None
        if stage in ("fit", None):
            if splits.train_ids:
                self._train_ds = self.build_dataset(
                    expand_range_selectors(splits.train_ids)
                )
            if splits.val_ids:
                self._val_ds = self.build_dataset(
                    expand_range_selectors(splits.val_ids)
                )
        if stage in ("test", None) and splits.test_ids:
            self._test_ds = self.build_dataset(expand_range_selectors(splits.test_ids))
        if stage in ("predict", None):
            predict_ids = (
                expand_range_selectors(splits.predict_ids)
                if splits.predict_ids
                else None
            )
            self._predict_ds = self.build_dataset(predict_ids)

    def _setup_module_owned(self, stage: str | None) -> None:
        if stage in ("fit", None):
            self._train_ds = self.build_stage_dataset("train")
            self._val_ds = self.build_stage_dataset("val")
        if stage in ("test", None):
            self._test_ds = self.build_stage_dataset("test")
        if stage in ("predict", None):
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
        return self._loader(self._train_ds, shuffle=True, name="train")

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val_ds, shuffle=False, name="val")

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test_ds, shuffle=False, name="test")

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self._predict_ds, shuffle=False, name="predict")


def create_data_module(
    registry: Any, data_config: DataConfig
) -> BaseHyperspectralDataModule:
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


__all__ = ["BaseHyperspectralDataModule", "create_data_module"]
