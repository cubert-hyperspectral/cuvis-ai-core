"""The dtype and shape normalisers of scripts/emit_metadata.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.emit_metadata import _dtype_to_string, _shape_to_int_list


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, "float32"),
        (np.dtype("uint8"), "uint8"),
        (np.float64, "float64"),
        (torch.Tensor, ""),
        (None, ""),
        (object(), ""),
    ],
)
def test_dtype_to_string(dtype, expected):
    assert _dtype_to_string(dtype) == expected


def test_shape_to_int_list_replaces_symbolic_dims():
    assert _shape_to_int_list((-1, "H", 3)) == [-1, -1, 3]
    assert _shape_to_int_list(()) == []
