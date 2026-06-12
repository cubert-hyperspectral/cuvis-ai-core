"""Tests for BaseHyperspectralDataModule split application + dispatch."""

from __future__ import annotations

import pytest

from cuvis_ai_core.data.datamodule import create_data_module
from cuvis_ai_schemas.training.data import DataConfig, DataSplitConfig
from tests.fixtures.fake_data_modules import FakeDataModule


def test_setup_from_splits():
    dm = FakeDataModule(
        splits=DataSplitConfig(train_ids=[0, 1, 2], val_ids=[0], test_ids=[0, 1]),
        batch_size=2,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3
    assert len(dm._val_ds) == 1
    dm.setup(stage="test")
    assert len(dm._test_ds) == 2
    batch = next(iter(dm.train_dataloader()))
    assert batch["x"].shape[0] == 2


def test_predict_empty_ids_iterates_all():
    dm = FakeDataModule(splits=DataSplitConfig(predict_ids=[]), batch_size=1)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 4  # build_dataset(None) -> all


def test_setup_from_splits_expands_range_strings():
    dm = FakeDataModule(
        splits=DataSplitConfig(train_ids=["0-2"], val_ids=[0, "4-5"]),
        batch_size=1,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3  # "0-2" -> [0, 1, 2]
    assert len(dm._val_ds) == 3  # [0] + "4-5" -> [0, 4, 5]


def test_setup_predict_expands_range_strings():
    dm = FakeDataModule(splits=DataSplitConfig(predict_ids=["0-3"]), batch_size=1)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 4  # "0-3" -> [0, 1, 2, 3]


def test_public_dataset_properties_get_and_set():
    # The former SingleCu3sDataModule exposed train/val/test/predict_ds publicly;
    # consumers read them and sometimes reassign (e.g. predict_ds = Subset(...)).
    dm = FakeDataModule(splits=DataSplitConfig(train_ids=[0, 1], predict_ids=[]))
    assert dm.train_ds is None and dm.predict_ds is None  # before setup
    dm.setup(stage="fit")
    assert dm.train_ds is dm._train_ds
    assert len(dm.train_ds) == 2
    sentinel = object()
    dm.predict_ds = sentinel  # settable
    assert dm._predict_ds is sentinel


def test_module_owned_splits():
    dm = FakeDataModule(splits=None)
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3  # build_stage_dataset path
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 3


def test_dataloader_before_setup_raises():
    dm = FakeDataModule(splits=DataSplitConfig(train_ids=[0]))
    with pytest.raises(RuntimeError, match="not built"):
        dm.test_dataloader()


def test_validate_params_default_is_noop():
    # No required keys -> no raise.
    FakeDataModule.validate_params({})


def test_create_data_module_dispatch():
    class _Reg:
        data_modules = {"fake": FakeDataModule}

    dc = DataConfig(
        data_module="fake", splits=DataSplitConfig(train_ids=[0, 1]), batch_size=2
    )
    dm = create_data_module(_Reg(), dc)
    assert isinstance(dm, FakeDataModule)
    assert dm.batch_size == 2


def test_create_data_module_unknown_raises():
    class _Reg:
        data_modules = {}

    with pytest.raises(ValueError, match="no plugin provides data module"):
        create_data_module(_Reg(), DataConfig(data_module="nope"))


def test_create_data_module_validates_params():
    class _Reg:
        data_modules = {"fake": FakeDataModule}

    dc = DataConfig(data_module="fake", params={"required_key": ""})
    with pytest.raises(ValueError, match="required_key"):
        create_data_module(_Reg(), dc)
