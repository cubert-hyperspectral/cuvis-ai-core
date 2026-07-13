"""Tests for samples_per_frame (within-epoch train multiplicity) in the base data module."""

from __future__ import annotations

import pytest
from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import _RepeatDataset
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind
from tests.fixtures.fake_data_modules import FakeDataModule


def _fi(ids):
    return Selector(kind=SelectorKind.FILE_INDICES, source="fake.cu3s", ids=ids)


class _Seq(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


# --------------------------------------------------------------------- unit


@pytest.mark.unit
def test_repeat_dataset_length_and_index_remap():
    r = _RepeatDataset(_Seq(5), 3)
    assert len(r) == 15
    assert [r[i] for i in range(15)] == [i % 5 for i in range(15)]


@pytest.mark.unit
def test_repeat_dataset_rejects_below_one():
    with pytest.raises(ValueError):
        _RepeatDataset(_Seq(5), 0)


@pytest.mark.unit
def test_samples_per_frame_validation():
    with pytest.raises(ValueError):
        FakeDataModule(splits=DataSplitConfig(train=[_fi([0, 1])]), samples_per_frame=0)


@pytest.mark.unit
def test_default_is_noop():
    """Default samples_per_frame=1 leaves the train loader unchanged (bit-for-bit)."""
    dm = FakeDataModule(splits=DataSplitConfig(train=[_fi([0, 1, 2])]), batch_size=1)
    assert dm.samples_per_frame == 1
    dm.setup(stage="fit")
    assert len(dm.train_dataloader().dataset) == 3


@pytest.mark.unit
def test_train_repeated_val_untouched_selector_path():
    dm = FakeDataModule(
        splits=DataSplitConfig(train=[_fi([0, 1, 2])], val=[_fi([3, 4])]),
        batch_size=2,
        samples_per_frame=3,
    )
    dm.setup(stage="fit")
    # train_ds property stays the unwrapped base (frame count), only the loader repeats.
    assert len(dm.train_ds) == 3
    assert len(dm.train_dataloader().dataset) == 9  # 3 frames x 3
    assert len(dm.val_dataloader().dataset) == 2  # val never repeated


@pytest.mark.unit
def test_module_owned_path_also_repeats():
    """Works for build_stage_dataset (module-owned splits), not just the selector path."""
    dm = FakeDataModule(samples_per_frame=4)  # no splits -> module-owned
    dm.setup(stage="fit")
    assert len(dm.train_dataloader().dataset) == 12  # _TinyDataset(3) x 4
    assert len(dm.val_dataloader().dataset) == 3  # val unchanged


# -------------------------------------------------------------- integration


@pytest.mark.integration
def test_train_loader_yields_n_times_the_samples():
    dm = FakeDataModule(
        splits=DataSplitConfig(train=[_fi([0, 1, 2, 3])]), batch_size=2, samples_per_frame=3
    )
    dm.setup(stage="fit")
    seen = sum(b["x"].shape[0] for b in dm.train_dataloader())
    assert seen == 12  # 4 frames x 3 = 12 samples across the epoch


@pytest.mark.integration
def test_each_frame_visited_n_times():
    dm = FakeDataModule(
        splits=DataSplitConfig(train=[_fi([0, 1, 2, 3])]), batch_size=1, samples_per_frame=3
    )
    dm.setup(stage="fit")
    from collections import Counter

    idxs = Counter(int(b["idx"][0]) for b in dm.train_dataloader())
    assert set(idxs) == {0, 1, 2, 3}
    assert all(c == 3 for c in idxs.values())  # each base frame seen exactly 3x
