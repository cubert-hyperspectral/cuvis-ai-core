"""Tests for BaseCuvisAIDataModule selector resolution + dispatch."""

from __future__ import annotations

import pytest

from cuvis_ai_core.data.datamodule import create_data_module
from cuvis_ai_core.data.selectors import SplitLeakageError
from cuvis_ai_schemas.training.data import (
    DataConfig,
    DataSplitConfig,
    Selector,
    SelectorKind,
)
from tests.fixtures.fake_data_modules import FakeDataModule


def _fi(ids):
    return Selector(kind=SelectorKind.FILE_INDICES, source="fake.cu3s", ids=ids)


def test_setup_from_selectors_fit_and_test():
    dm = FakeDataModule(
        splits=DataSplitConfig(
            train=[_fi([0, 1, 2])], val=[_fi([3])], test=[_fi([4, 5])]
        ),
        batch_size=2,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3
    assert len(dm._val_ds) == 1
    dm.setup(stage="test")
    assert len(dm._test_ds) == 2
    batch = next(iter(dm.train_dataloader()))
    assert batch["x"].shape[0] == 2


def test_predict_empty_iterates_whole_universe():
    dm = FakeDataModule(splits=DataSplitConfig(predict=[]), batch_size=1)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 6  # empty predict -> all


def test_file_indices_expand_range_strings():
    dm = FakeDataModule(
        splits=DataSplitConfig(train=[_fi(["0-2"])], val=[_fi([3, "4-5"])]),
        batch_size=1,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3  # "0-2" -> [0, 1, 2]
    assert len(dm._val_ds) == 3  # [3] + "4-5" -> [3, 4, 5]


def test_dir_indices_select_positions():
    dm = FakeDataModule(
        splits=DataSplitConfig(
            predict=[Selector(kind=SelectorKind.DIR_INDICES, ids=[0, 2, 4])]
        ),
        batch_size=1,
    )
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 3


def test_tag_and_categories_resolve_by_metadata():
    dm = FakeDataModule(
        splits=DataSplitConfig(
            train=[Selector(kind=SelectorKind.TAG, any_of=["normal"])],
            test=[Selector(kind=SelectorKind.CATEGORIES, any_of=["scrap"])],
        ),
        batch_size=1,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3  # even indices tagged normal
    dm.setup(stage="test")
    assert len(dm._test_ds) == 3  # odd indices, category scrap


def test_unknown_category_name_raises():
    dm = FakeDataModule(
        splits=DataSplitConfig(
            test=[Selector(kind=SelectorKind.CATEGORIES, any_of=["ghost"])]
        ),
    )
    with pytest.raises(ValueError, match="not in dataset categories"):
        dm.setup(stage="test")


def test_selector_matching_zero_raises():
    dm = FakeDataModule(splits=DataSplitConfig(train=[_fi([99])]))
    with pytest.raises(ValueError, match="matched 0 samples"):
        dm.setup(stage="fit")


def test_leakage_error_by_default():
    dm = FakeDataModule(
        splits=DataSplitConfig(train=[_fi([0, 1])], test=[_fi([1, 2])]),
    )
    with pytest.raises(SplitLeakageError, match="shared between train and test"):
        dm.setup(stage="fit")


def test_leakage_warn_allows_overlap():
    dm = FakeDataModule(
        splits=DataSplitConfig(
            train=[_fi([0, 1])], test=[_fi([1, 2])], leakage_check="warn"
        ),
    )
    dm.setup(stage="fit")  # logs, does not raise
    assert len(dm._train_ds) == 2


def test_predict_may_overlap_test():
    dm = FakeDataModule(
        splits=DataSplitConfig(test=[_fi([1])], predict=[_fi([1])]),
    )
    dm.setup()  # predict overlapping test is allowed
    assert len(dm._predict_ds) == 1


def test_public_dataset_properties_get_and_set():
    dm = FakeDataModule(splits=DataSplitConfig(train=[_fi([0, 1])]))
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
    dm = FakeDataModule(splits=DataSplitConfig(train=[_fi([0])]))
    with pytest.raises(RuntimeError, match="not built"):
        dm.test_dataloader()


def test_validate_params_default_is_noop():
    FakeDataModule.validate_params({})


def test_splits_path_loads_selectors_from_file(tmp_path):
    from cuvis_ai_core.data.splits_io import save_splits

    save_splits(
        DataSplitConfig(train=[_fi([0, 1, 2])], test=[_fi([4, 5])]),
        tmp_path / "splits.json",
    )
    dm = FakeDataModule(
        splits=DataSplitConfig(splits_path=str(tmp_path / "splits.json")),
        batch_size=1,
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert len(dm._train_ds) == 3  # loaded from file, not empty
    assert len(dm._test_ds) == 2


def test_splits_path_inline_stage_overrides_file(tmp_path):
    from cuvis_ai_core.data.splits_io import save_splits

    # File assigns train; inline adds val and must not lose the file's train.
    save_splits(DataSplitConfig(train=[_fi([0, 1, 2])]), tmp_path / "splits.json")
    dm = FakeDataModule(
        splits=DataSplitConfig(
            splits_path=str(tmp_path / "splits.json"), val=[_fi([3])]
        ),
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3  # from the file
    assert len(dm._val_ds) == 1  # from the inline override


def test_effective_splits_without_path_is_identity():
    splits = DataSplitConfig(train=[_fi([0])])
    dm = FakeDataModule(splits=splits)
    assert dm._effective_splits() is splits  # no file load when splits_path unset


def test_splits_path_leakage_check_is_file_owned(tmp_path):
    """``leakage_check`` comes from the file even when also set inline.

    Documents the file-owned limitation: the enum has a default so a plain ``or``
    cannot distinguish "unset" from "default", so the file value wins. The inline
    stages still merge, and the file's own stages survive the ``model_copy`` rebuild.
    """
    from cuvis_ai_core.data.splits_io import save_splits

    save_splits(
        DataSplitConfig(train=[_fi([0, 1])], leakage_check="warn"),
        tmp_path / "splits.json",
    )
    dm = FakeDataModule(
        splits=DataSplitConfig(
            splits_path=str(tmp_path / "splits.json"),
            leakage_check="off",  # inline value must NOT win
        ),
    )
    effective = dm._effective_splits()
    assert effective.leakage_check == "warn"  # file-owned, inline "off" ignored
    assert len(effective.train) == 1  # file stage survived the model_copy rebuild


def test_create_data_module_dispatch():
    class _Reg:
        data_modules = {"fake": FakeDataModule}

    dc = DataConfig(
        data_module="fake", splits=DataSplitConfig(train=[_fi([0, 1])]), batch_size=2
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


class _SelectorOnlyModule(FakeDataModule):
    """A module that enumerates a universe but does not own split semantics."""

    DATA_MODULE_NAME = "selector_only"
    OWNS_SPLITS = False


def test_owns_splits_false_refuses_splitless_training_stages():
    dm = _SelectorOnlyModule()
    for stage in ("fit", "validate", "test", None):
        with pytest.raises(ValueError, match="does not own split semantics"):
            dm.setup(stage)


def test_owns_splits_false_allows_splitless_predict():
    dm = _SelectorOnlyModule()
    dm.setup("predict")
    assert len(dm._predict_ds) == 3  # build_stage_dataset("predict")


def test_owns_splits_false_with_splits_trains_normally():
    dm = _SelectorOnlyModule(
        splits=DataSplitConfig(train=[_fi([0, 1])], val=[_fi([2])], test=[_fi([3])])
    )
    dm.setup("fit")
    assert len(dm._train_ds) == 2


def test_owns_splits_default_true_keeps_module_owned_path():
    dm = FakeDataModule()
    dm.setup("fit")
    assert len(dm._train_ds) == 3  # module-owned build_stage_dataset unchanged
