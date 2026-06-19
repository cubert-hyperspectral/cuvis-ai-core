"""Property-based tests for cuvis_ai_core.data.splits_io.

Verifies universe_hash determinism + order-sensitivity, and has_dir_indices
structural detection. No CUDA, no I/O.
"""

pytest = __import__("pytest")
pytest.importorskip("hypothesis")

# ruff: noqa: E402 — imports must follow importorskip to avoid ImportError on collection
from hypothesis import given, settings
from hypothesis import strategies as st
from cuvis_ai_schemas.testing.strategies import (
    data_split_config_strategy,
    sample_ref_strategy,
)

from cuvis_ai_core.data.splits_io import has_dir_indices, universe_hash

# ---------------------------------------------------------------------------
# universe_hash properties
# ---------------------------------------------------------------------------


@given(refs=st.lists(sample_ref_strategy(), max_size=10))
def test_universe_hash_is_deterministic(refs) -> None:
    assert universe_hash(refs) == universe_hash(refs)


@given(refs=st.lists(sample_ref_strategy(), max_size=10))
def test_universe_hash_is_hex_64_chars(refs) -> None:
    h = universe_hash(refs)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


@given(
    refs=st.lists(sample_ref_strategy(), min_size=2, max_size=8),
)
@settings(max_examples=80)
def test_universe_hash_is_order_sensitive(refs) -> None:
    """Reversing a non-palindromic list changes the hash."""
    reversed_refs = list(reversed(refs))
    uids = [r.uid for r in refs]
    reversed_uids = [r.uid for r in reversed_refs]
    if uids == reversed_uids:
        return  # palindrome — skip
    assert universe_hash(refs) != universe_hash(reversed_refs)


# ---------------------------------------------------------------------------
# has_dir_indices properties
# ---------------------------------------------------------------------------


@given(splits=data_split_config_strategy())
@settings(max_examples=80)
def test_has_dir_indices_returns_bool(splits) -> None:
    result = has_dir_indices(splits)
    assert isinstance(result, bool)


@given(splits=data_split_config_strategy())
@settings(max_examples=80)
def test_has_dir_indices_is_deterministic(splits) -> None:
    assert has_dir_indices(splits) == has_dir_indices(splits)
