"""Helpers for nodes that fit statistics in Phase 1 (``statistical_initialization``).

- :func:`random_cap` is a seeded row cap that bounds the memory of a fit, the
  same subset for the same seed on every machine.
- :func:`require_fitted` is the guard at the top of a fitted node's
  ``forward``: a node that has neither run its Phase 1 nor received fitted
  weights fails with one clear message instead of computing on its initial
  buffers.
"""

from __future__ import annotations

import torch

from cuvis_ai_core.node.node import Node


def random_cap(
    rows: torch.Tensor, max_rows: int, generator: torch.Generator
) -> torch.Tensor:
    """Seeded random subset of at most ``max_rows`` rows of ``rows``.

    Parameters
    ----------
    rows : torch.Tensor
        Rows along dimension 0 (for example ``[N, D]`` feature vectors).
    max_rows : int
        Largest number of rows to keep.
    generator : torch.Generator
        Source of the permutation. It is drawn from only when rows are
        dropped, so a fit that stays within the cap does not shift the
        generator's state for later draws.

    Returns
    -------
    torch.Tensor
        ``rows`` itself when it holds at most ``max_rows`` rows, else
        ``max_rows`` rows picked by a seeded permutation, on ``rows``' device.
    """
    if rows.shape[0] <= max_rows:
        return rows
    keep = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
    return rows[keep.to(rows.device)]


def require_fitted(node: Node) -> None:
    """Raise ``RuntimeError`` when a fitted node runs before Phase 1 or a weights load.

    Parameters
    ----------
    node : Node
        The node about to run its ``forward``.

    Raises
    ------
    RuntimeError
        If ``node`` is not statistically initialized.
    """
    if not node._statistically_initialized:
        raise RuntimeError(
            f"{type(node).__name__} requires statistical_initialization() (Phase 1) "
            "or loaded weights before forward()."
        )
