from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def check_ndim(
    x: np.ndarray, expected_dims: Sequence[int], name: str = "array"
) -> None:
    """Check number of dimensions (generic; works for NumPy/Torch tensors already on CPU)."""
    if x.ndim not in expected_dims:
        raise ValueError(
            f"{name} must have ndim in {expected_dims}, got {x.ndim} (shape={x.shape})"
        )


def normalize_per_channel_vectorized(
    array: np.ndarray, range_min: float = 0.0, range_max: float = 1.0
) -> np.ndarray:
    arr = array.astype(np.float32)
    min_vals = arr.min(axis=(0, 1), keepdims=True)
    max_vals = arr.max(axis=(0, 1), keepdims=True)
    denom = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
    norm = (arr - min_vals) / denom
    norm = norm * (range_max - range_min) + range_min
    return norm


def _resolve_measurement_indices(
    indices: Sequence[int] | Iterable[int] | None,
    max_index: int | None = None,
) -> list[int]:
    """Coerce, validate and store indices."""

    if indices is None:
        if max_index is None:
            raise ValueError("Either indices or max_index must be provided.")
        resolved = list(range(max_index))

    elif isinstance(indices, range):
        # Interpret range(start, -1) as range(start, max_index)
        if len(indices) == 0 and indices.step > 0 and indices.stop == -1:
            if max_index is None:
                raise ValueError(
                    "max_index is required when using an open-ended range like range(10, -1)."
                )
            resolved = list(range(indices.start, max_index, indices.step))
        else:
            resolved = list(indices)

    else:
        resolved = list(indices)

    if not resolved:
        if max_index == 0:
            return []
        raise ValueError("At least one index is required.")

    if max_index is not None:
        invalid_indices = [idx for idx in resolved if idx < 0 or idx >= max_index]
        if invalid_indices:
            raise IndexError(
                f"Indices {invalid_indices} are out of bounds for selection with max_index={max_index}."
            )

    if len(set(resolved)) != len(resolved):
        raise ValueError("Indices contain duplicates; provide unique indices.")

    return resolved
