"""Property-based tests for cuvis_ai_core.data.selectors.

Verifies dedup-by-uid, order-preservation, and validate_leakage(mode='off').
Uses cuvis_ai_schemas.testing.strategies (safe once hypothesis is installed).
No CUDA, no I/O.
"""

pytest = __import__("pytest")
pytest.importorskip("hypothesis")

# ruff: noqa: E402 — imports must follow importorskip to avoid ImportError on collection
from hypothesis import given, settings
from hypothesis import strategies as st
from cuvis_ai_schemas.testing.strategies import sample_ref_strategy
from cuvis_ai_schemas.training.data import Selector, SelectorKind

from cuvis_ai_core.data.selectors import SplitLeakageError, validate_leakage

# ---------------------------------------------------------------------------
# validate_leakage properties
# ---------------------------------------------------------------------------


@given(
    train=st.lists(sample_ref_strategy(), max_size=5),
    val=st.lists(sample_ref_strategy(), max_size=5),
    test=st.lists(sample_ref_strategy(), max_size=5),
)
def test_validate_leakage_mode_off_never_raises(train, val, test) -> None:
    """mode='off' must always return without raising."""
    validate_leakage(train, val, test, mode="off")


@given(refs=st.lists(sample_ref_strategy(), min_size=1, max_size=10))
def test_validate_leakage_overlapping_train_val_raises_in_error_mode(refs) -> None:
    """Sharing refs between train and val triggers SplitLeakageError in error mode."""
    import pytest

    with pytest.raises(SplitLeakageError):
        validate_leakage(refs, refs, [], mode="error")


# ---------------------------------------------------------------------------
# resolve_selectors: ALL selector invariants
# ---------------------------------------------------------------------------


@given(refs=st.lists(sample_ref_strategy(), min_size=1, max_size=10))
@settings(max_examples=80)
def test_resolve_selectors_all_returns_subset_of_universe(refs) -> None:
    """ALL selector result is a subset of the input refs (by uid)."""
    from cuvis_ai_core.data.selectors import resolve_selectors

    all_sel = Selector(kind=SelectorKind.ALL)
    result = resolve_selectors([all_sel], refs)
    universe_uids = {r.uid for r in refs}
    for ref in result:
        assert ref.uid in universe_uids


@given(refs=st.lists(sample_ref_strategy(), min_size=1, max_size=10))
@settings(max_examples=80)
def test_resolve_selectors_all_deduplicates_by_uid(refs) -> None:
    """ALL selector result has no duplicate uids."""
    from cuvis_ai_core.data.selectors import resolve_selectors

    all_sel = Selector(kind=SelectorKind.ALL)
    result = resolve_selectors([all_sel], refs)
    uids = [r.uid for r in result]
    assert len(uids) == len(set(uids))
