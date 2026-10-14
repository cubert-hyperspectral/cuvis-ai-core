"""``node.fit_utils``: the seeded row cap and the fitted-node guard."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.node.fit_utils import random_cap, require_fitted

pytestmark = pytest.mark.unit


def test_random_cap_within_the_cap_returns_the_rows_and_leaves_the_generator() -> None:
    rows = torch.arange(10).reshape(5, 2)
    gen = torch.Generator().manual_seed(3)
    before = gen.get_state()
    assert random_cap(rows, 5, gen) is rows
    assert torch.equal(gen.get_state(), before)


def test_random_cap_is_a_seeded_subset() -> None:
    rows = torch.arange(200).reshape(100, 2)
    a = random_cap(rows, 7, torch.Generator().manual_seed(1))
    b = random_cap(rows, 7, torch.Generator().manual_seed(1))
    c = random_cap(rows, 7, torch.Generator().manual_seed(2))
    assert a.shape == (7, 2) and torch.equal(a, b) and not torch.equal(a, c)
    picked = {tuple(r.tolist()) for r in a}
    assert picked <= {tuple(r.tolist()) for r in rows} and len(picked) == 7


class _Fitted:
    """Stand-in with the one attribute the guard reads."""

    def __init__(self, fitted: bool) -> None:
        self._statistically_initialized = fitted


def test_require_fitted() -> None:
    require_fitted(_Fitted(True))
    with pytest.raises(
        RuntimeError, match=r"_Fitted requires statistical_initialization"
    ):
        require_fitted(_Fitted(False))
