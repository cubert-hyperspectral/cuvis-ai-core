"""Tests for general utility helpers."""

from __future__ import annotations

import pytest

from cuvis_ai_core.utils.general import expand_range_selectors


def test_expand_range_selectors_inclusive_range() -> None:
    assert expand_range_selectors(["0-3"]) == [0, 1, 2, 3]


def test_expand_range_selectors_with_step() -> None:
    assert expand_range_selectors(["0-10:2"]) == [0, 2, 4, 6, 8, 10]


def test_expand_range_selectors_single_value_range() -> None:
    assert expand_range_selectors(["3-3"]) == [3]


def test_expand_range_selectors_passes_through_ints_and_keys() -> None:
    assert expand_range_selectors([0, 2, "scrap_02"]) == [0, 2, "scrap_02"]


def test_expand_range_selectors_mixed_list() -> None:
    assert expand_range_selectors(["0-2", 5, "stem"]) == [0, 1, 2, 5, "stem"]


def test_expand_range_selectors_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="start 5 must be <= stop 2"):
        expand_range_selectors(["5-2"])


def test_expand_range_selectors_rejects_zero_step() -> None:
    with pytest.raises(ValueError, match="step must be > 0"):
        expand_range_selectors(["0-10:0"])
