"""Unit tests for resolve_selectors / leakage / splits_io over a SampleRef universe."""

from __future__ import annotations

import pytest

from cuvis_ai_core.data.selectors import (
    SplitLeakageError,
    required_attrs,
    resolve_selectors,
    validate_leakage,
)
from cuvis_ai_core.data.splits_io import (
    has_dir_indices,
    load_splits,
    save_splits,
    universe_hash,
    verify_universe,
)
from cuvis_ai_schemas.training.data import (
    DataSplitConfig,
    SampleRef,
    Selector,
    SelectorKind,
)


def _universe():
    # Two sources; a.cu3s has measurements 0..2 (a normal, two scrap), b.cu3s one whole file.
    return [
        SampleRef(
            source="a.cu3s",
            index=0,
            label_id=0,
            stem="a",
            tags=["normal"],
            category_ids=[],
        ),
        SampleRef(
            source="a.cu3s",
            index=1,
            label_id=1,
            stem="a",
            tags=["scrap"],
            category_ids=[1],
        ),
        SampleRef(
            source="a.cu3s",
            index=2,
            label_id=2,
            stem="a",
            tags=["scrap"],
            category_ids=[1, 2],
        ),
        SampleRef(
            source="b.cu3s",
            index=0,
            label_id=0,
            stem="b",
            tags=["normal"],
            category_ids=[],
        ),
    ]


def _uids(refs):
    return [r.uid for r in refs]


def test_files_and_file_indices():
    refs = _universe()
    got = resolve_selectors([Selector(kind=SelectorKind.FILES, paths=["b.cu3s"])], refs)
    assert _uids(got) == ["b.cu3s#0"]
    got = resolve_selectors(
        [Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=["0-1"])], refs
    )
    assert _uids(got) == ["a.cu3s#0", "a.cu3s#1"]


def test_dir_indices_positions_and_bounds():
    refs = _universe()
    got = resolve_selectors([Selector(kind=SelectorKind.DIR_INDICES, ids=[0, 3])], refs)
    assert _uids(got) == ["a.cu3s#0", "b.cu3s#0"]
    with pytest.raises(ValueError, match="out of range"):
        resolve_selectors([Selector(kind=SelectorKind.DIR_INDICES, ids=[99])], refs)


def test_stems_glob_tag_categories_all():
    refs = _universe()
    assert _uids(
        resolve_selectors([Selector(kind=SelectorKind.STEMS, stems=["b"])], refs)
    ) == ["b.cu3s#0"]
    assert (
        len(resolve_selectors([Selector(kind=SelectorKind.GLOB, pattern="a*")], refs))
        == 3
    )
    assert (
        len(
            resolve_selectors(
                [Selector(kind=SelectorKind.TAG, any_of=["normal"])], refs
            )
        )
        == 2
    )
    cats = resolve_selectors(
        [Selector(kind=SelectorKind.CATEGORIES, any_of=["scrap"])],
        refs,
        name_to_id={"scrap": 1},
    )
    assert _uids(cats) == ["a.cu3s#1", "a.cu3s#2"]
    assert len(resolve_selectors([Selector(kind=SelectorKind.ALL)], refs)) == 4


def test_unknown_category_name_raises():
    refs = _universe()
    with pytest.raises(ValueError, match="not in dataset categories"):
        resolve_selectors(
            [Selector(kind=SelectorKind.CATEGORIES, any_of=["ghost"])],
            refs,
            name_to_id={"scrap": 1},
        )
    with pytest.raises(ValueError, match="needs a category name"):
        resolve_selectors(
            [Selector(kind=SelectorKind.CATEGORIES, any_of=["scrap"])], refs
        )


def test_union_except_intersect():
    refs = _universe()
    union = Selector(
        kind=SelectorKind.UNION,
        of=[
            Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=[0]),
            Selector(kind=SelectorKind.FILES, paths=["b.cu3s"]),
        ],
    )
    assert _uids(resolve_selectors([union], refs)) == ["a.cu3s#0", "b.cu3s#0"]

    exc = Selector(
        kind=SelectorKind.EXCEPT,
        of=[
            Selector(kind=SelectorKind.FILES, paths=["a.cu3s"]),
            Selector(kind=SelectorKind.TAG, any_of=["scrap"]),
        ],
    )
    assert _uids(resolve_selectors([exc], refs)) == ["a.cu3s#0"]  # a.cu3s minus scrap

    inter = Selector(
        kind=SelectorKind.INTERSECT,
        of=[
            Selector(kind=SelectorKind.FILES, paths=["a.cu3s"]),
            Selector(kind=SelectorKind.TAG, any_of=["scrap"]),
        ],
    )
    assert _uids(resolve_selectors([inter], refs)) == ["a.cu3s#1", "a.cu3s#2"]


def test_disjoint_intersect_is_loud():
    refs = _universe()
    inter = Selector(
        kind=SelectorKind.INTERSECT,
        of=[
            Selector(kind=SelectorKind.FILES, paths=["a.cu3s"]),
            Selector(kind=SelectorKind.FILES, paths=["b.cu3s"]),
        ],
    )
    with pytest.raises(ValueError, match="matched 0 samples"):
        resolve_selectors([inter], refs)


def test_union_dedup_and_order_preserved():
    refs = _universe()
    sels = [
        Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=[2, 0]),
        Selector(
            kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=[0, 1]
        ),  # 0 is a dup
    ]
    # Within a selector matches come back in canonical universe order (deterministic);
    # across selectors the stage list is concatenated with first-occurrence dedup.
    assert _uids(resolve_selectors(sels, refs)) == ["a.cu3s#0", "a.cu3s#2", "a.cu3s#1"]


def test_required_attrs():
    splits = DataSplitConfig(
        train=[Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=[0])],
        test=[Selector(kind=SelectorKind.CATEGORIES, any_of=["scrap"])],
    )
    assert required_attrs(splits) == frozenset({"category_ids"})
    none_needed = DataSplitConfig(
        train=[Selector(kind=SelectorKind.FILES, paths=["a"])]
    )
    assert required_attrs(none_needed) == frozenset()


def test_validate_leakage_modes():
    refs = _universe()
    train = refs[:2]
    test = refs[1:3]  # shares refs[1]
    with pytest.raises(SplitLeakageError, match="train and test"):
        validate_leakage(train, [], test, mode="error")
    validate_leakage(train, [], test, mode="warn")  # no raise
    validate_leakage(train, [], test, mode="off")  # no raise
    # predict is excluded; train/val/test disjoint -> ok
    validate_leakage(refs[:1], refs[1:2], refs[2:3], mode="error")


def test_splits_io_roundtrip_and_universe_hash(tmp_path):
    refs = _universe()
    splits = DataSplitConfig(
        train=[Selector(kind=SelectorKind.DIR_INDICES, ids=[0, 1])],
        universe_hash=universe_hash(refs),
    )
    path = tmp_path / "splits.json"
    save_splits(splits, path)
    loaded = load_splits(path)
    assert loaded.train[0].kind == SelectorKind.DIR_INDICES
    assert has_dir_indices(loaded)
    verify_universe(loaded, refs)  # matches -> no raise
    with pytest.raises(ValueError, match="universe changed"):
        verify_universe(loaded, refs[:3])  # different universe


def test_splits_io_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path / "nope.json")
