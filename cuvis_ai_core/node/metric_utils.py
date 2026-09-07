"""Helpers shared by the metric nodes that score dense per-pixel predictions.

A pixel metric over a full frame hands torchmetrics one element per pixel.
Past roughly 50k elements the binned implementations stop being the cheap
path: they allocate an ``[N, thresholds]`` int64 confusion tensor, which at
a megapixel frame costs hundreds of MiB of transient memory per update, on
top of the detector's own activations.

Both helpers exist so the metric nodes in the node library and in the
plugins compute the same thing:

- :func:`subsample_hw` takes every ``stride``-th row and column, which
  keeps the estimate unbiased for spatially smooth score maps while
  cutting the element count by ``stride ** 2``.
- :func:`warn_below_vectorized_cutoff` warns once when the subsample has
  dropped a frame below that 50k mark, so a stride chosen for memory is
  not silently also changing which code path torchmetrics takes.
"""

from __future__ import annotations

import torch
from loguru import logger

# Element count below which the binned metrics take their vectorized path.
# A subsample that crosses it changes more than the memory footprint, so the
# crossing is worth one log line.
VECTORIZED_CUTOFF = 50_000


def subsample_hw(x: torch.Tensor, stride: int) -> torch.Tensor:
    """Take every ``stride``-th row and column of a ``[B, H, W, ...]`` tensor.

    Parameters
    ----------
    x : torch.Tensor
        Tensor whose dimensions 1 and 2 are the spatial ones. Tensors with
        fewer than three dimensions are returned unchanged: there is no
        spatial grid to thin.
    stride : int
        Spatial stride, ``>= 1``. ``1`` is the identity and returns ``x``
        itself, so the caller pays nothing for leaving subsampling off.

    Returns
    -------
    torch.Tensor
        A strided view of ``x`` (no copy, no allocation).

    Raises
    ------
    TypeError
        If ``stride`` is not an integer (a float stride would silently
        truncate).
    ValueError
        If ``stride`` is smaller than 1.
    """
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise TypeError(f"stride must be an int >= 1, got {stride!r}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride == 1 or x.ndim < 3:
        return x
    return x[:, ::stride, ::stride]


def warn_below_vectorized_cutoff(
    node_name: str,
    n_sub: int,
    n_full: int,
    state: dict,
) -> None:
    """Warn once per node when subsampling crossed the vectorized cutoff.

    Parameters
    ----------
    node_name : str
        Name used in the log line, so a pipeline with several metric nodes
        says which one is affected.
    n_sub : int
        Element count after subsampling.
    n_full : int
        Element count of the full frame.
    state : dict
        Per-node mutable dict the caller owns (any dict will do). The
        warning sets a key in it, so later batches stay quiet.
    """
    if state.get("_warned_vectorized_cutoff"):
        return
    if not (n_sub <= VECTORIZED_CUTOFF < n_full):
        return
    state["_warned_vectorized_cutoff"] = True
    logger.warning(
        f"{node_name}: subsampling reduced the pixel metric from {n_full} to "
        f"{n_sub} elements, crossing the {VECTORIZED_CUTOFF}-element cutoff "
        "below which the binned metrics switch to their vectorized path. The "
        "metric now costs far less memory, and its value comes from a "
        "different code path than an unsubsampled run."
    )
