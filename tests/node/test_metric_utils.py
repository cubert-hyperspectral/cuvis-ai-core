"""``node.metric_utils``: the spatial subsample and its one-shot warning.

The metric nodes in the node library and in the plugins share these two
helpers, so their behaviour is pinned here once: stride 1 is the identity,
a stride thins both spatial dimensions without copying, and the
cutoff warning fires exactly once per node no matter how many batches run.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

from cuvis_ai_core.node.metric_utils import (
    VECTORIZED_CUTOFF,
    subsample_hw,
    warn_below_vectorized_cutoff,
)


@pytest.fixture
def warnings_seen():
    """Collect loguru WARNING messages emitted during the test."""
    records: list[str] = []
    sink_id = logger.add(
        lambda message: records.append(message.record["message"]), level="WARNING"
    )
    try:
        yield records
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# subsample_hw
# ---------------------------------------------------------------------------


def test_stride_one_returns_the_same_tensor() -> None:
    x = torch.rand(2, 8, 8, 1)
    assert subsample_hw(x, 1) is x


def test_stride_thins_both_spatial_dimensions() -> None:
    x = torch.arange(2 * 8 * 6 * 3, dtype=torch.float32).reshape(2, 8, 6, 3)

    out = subsample_hw(x, 2)

    assert out.shape == (2, 4, 3, 3)
    torch.testing.assert_close(out, x[:, ::2, ::2])
    # A strided view, not a copy: no allocation for the metric to pay.
    assert out.data_ptr() == x.data_ptr()


def test_non_divisible_stride_rounds_up() -> None:
    x = torch.rand(1, 7, 5)
    out = subsample_hw(x, 3)
    assert out.shape == (1, 3, 2)


def test_tensors_without_a_spatial_grid_pass_through() -> None:
    # Image-level scores are [B] or [B, 1]: nothing to subsample.
    per_image = torch.rand(4)
    assert subsample_hw(per_image, 4) is per_image
    assert subsample_hw(torch.rand(4, 1), 4).shape == (4, 1)


@pytest.mark.parametrize("stride", [0, -1])
def test_stride_below_one_is_rejected(stride: int) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        subsample_hw(torch.rand(1, 4, 4), stride)


@pytest.mark.parametrize("stride", [2.0, "2", None, True])
def test_non_integer_stride_is_rejected(stride: object) -> None:
    with pytest.raises(TypeError, match="int"):
        subsample_hw(torch.rand(1, 4, 4), stride)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# warn_below_vectorized_cutoff
# ---------------------------------------------------------------------------


def test_warns_once_across_batches(warnings_seen) -> None:
    state: dict = {}

    for _ in range(5):
        warn_below_vectorized_cutoff("pixel_auroc", 40_000, 1_080_000, state)

    assert len(warnings_seen) == 1
    assert "pixel_auroc" in warnings_seen[0]
    assert "40000" in warnings_seen[0] and "1080000" in warnings_seen[0]


def test_each_node_gets_its_own_warning(warnings_seen) -> None:
    # The state dict is per node, so two metric nodes both report.
    warn_below_vectorized_cutoff("a", 10, 1_000_000, {})
    warn_below_vectorized_cutoff("b", 10, 1_000_000, {})
    assert len(warnings_seen) == 2


def test_no_warning_when_the_full_frame_was_already_below_the_cutoff(
    warnings_seen,
) -> None:
    warn_below_vectorized_cutoff("small", 100, VECTORIZED_CUTOFF, {})
    assert warnings_seen == []


def test_no_warning_when_the_subsample_stays_above_the_cutoff(warnings_seen) -> None:
    warn_below_vectorized_cutoff("big", VECTORIZED_CUTOFF + 1, 1_000_000, {})
    assert warnings_seen == []
