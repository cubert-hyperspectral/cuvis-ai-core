"""Tests for lazy exports in ``cuvis_ai_core.data``."""

from __future__ import annotations

import pytest

import cuvis_ai_core.data as data_mod
from cuvis_ai_core.data.public_datasets import PublicDatasets


def test_data_module_lazy_exports_and_missing_attribute() -> None:
    assert data_mod.PublicDatasets is PublicDatasets
    assert callable(data_mod.rle_list_to_mask)

    with pytest.raises(AttributeError, match="has no attribute 'missing_export'"):
        getattr(data_mod, "missing_export")
