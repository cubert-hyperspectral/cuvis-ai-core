"""Tests for general utility helpers."""

from __future__ import annotations

import pytest

from cuvis_ai_core.utils.general import _resolve_measurement_indices


def test_resolve_measurement_indices_requires_indices_or_max_index() -> None:
    with pytest.raises(
        ValueError, match="Either indices or max_index must be provided"
    ):
        _resolve_measurement_indices(None)


def test_resolve_measurement_indices_expands_open_ended_range() -> None:
    assert _resolve_measurement_indices(range(2, -1), max_index=6) == [2, 3, 4, 5]


def test_resolve_measurement_indices_keeps_regular_range() -> None:
    assert _resolve_measurement_indices(range(1, 4), max_index=10) == [1, 2, 3]


def test_resolve_measurement_indices_open_ended_range_requires_max_index() -> None:
    with pytest.raises(
        ValueError, match="max_index is required when using an open-ended range"
    ):
        _resolve_measurement_indices(range(4, -1))


def test_resolve_measurement_indices_allows_empty_only_for_zero_length_selection() -> (
    None
):
    assert _resolve_measurement_indices([], max_index=0) == []


def test_resolve_measurement_indices_rejects_out_of_bounds_values() -> None:
    with pytest.raises(IndexError, match="max_index=3"):
        _resolve_measurement_indices([0, 3], max_index=3)
