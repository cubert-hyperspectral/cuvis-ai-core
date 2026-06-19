"""Property-based tests for cuvis_ai_core.utils.general.expand_range_selectors.

Verifies inclusive range expansion, int/non-range passthrough, step handling,
and error cases. No CUDA, no I/O.
"""

pytest = __import__("pytest")
pytest.importorskip("hypothesis")

# ruff: noqa: E402 — imports must follow importorskip to avoid ImportError on collection
import pytest as _pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cuvis_ai_core.utils.general import expand_range_selectors

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_small_int = st.integers(min_value=0, max_value=200)
_non_range_str = st.text(
    min_size=1,
    max_size=20,
    alphabet="abcdefghijklmnopqrstuvwxyz_.",
)
_range_str = st.builds(
    lambda start, length, step: f"{start}-{start + length}"
    + (f":{step}" if step > 1 else ""),
    start=st.integers(min_value=0, max_value=100),
    length=st.integers(min_value=0, max_value=50),
    step=st.integers(min_value=1, max_value=10),
)

_selector_item = st.one_of(_small_int, _non_range_str, _range_str)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(items=st.lists(_small_int, max_size=20))
def test_expand_range_selectors_passes_integers_through(items: list[int]) -> None:
    result = expand_range_selectors(items)
    assert result == items


@given(items=st.lists(_non_range_str, max_size=20))
def test_expand_range_selectors_passes_non_range_strings_through(
    items: list[str],
) -> None:
    result = expand_range_selectors(items)
    assert result == items


@given(
    start=st.integers(min_value=0, max_value=100),
    length=st.integers(min_value=0, max_value=30),
)
@settings(max_examples=100)
def test_expand_range_selectors_range_is_inclusive(start: int, length: int) -> None:
    stop = start + length
    result = expand_range_selectors([f"{start}-{stop}"])
    assert isinstance(result[0], int)
    assert result[0] == start
    assert result[-1] == stop
    assert len(result) == length + 1


@given(
    start=st.integers(min_value=0, max_value=50),
    length=st.integers(min_value=1, max_value=20),
    step=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_expand_range_selectors_step_controls_stride(
    start: int, length: int, step: int
) -> None:
    stop = start + length * step
    result = expand_range_selectors([f"{start}-{stop}:{step}"])
    assert all(isinstance(x, int) for x in result)
    assert result[0] == start
    assert all(result[i + 1] - result[i] == step for i in range(len(result) - 1))


@given(items=st.lists(_selector_item, min_size=1, max_size=10))
def test_expand_range_selectors_output_is_list(items: list) -> None:
    result = expand_range_selectors(items)
    assert isinstance(result, list)


def test_expand_range_selectors_rejects_zero_step() -> None:
    with _pytest.raises(ValueError, match="step must be > 0"):
        expand_range_selectors(["0-10:0"])


def test_expand_range_selectors_rejects_start_greater_than_stop() -> None:
    with _pytest.raises(ValueError, match="start"):
        expand_range_selectors(["10-5"])
