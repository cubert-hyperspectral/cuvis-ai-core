"""Generic JSON round-trip property tests for cuvis_ai_schemas models.

Uses MODEL_STRATEGIES from cuvis_ai_schemas.testing — one strategy per schema
model that core re-exports or consumes. Proves that every model serialises to
a dict, back to a model, and back to a dict without loss. Also serves as a
smoke test that the cross-repo testing module is importable inside core's env.

Skipped gracefully when hypothesis is not installed (uv sync without --all-extras).
"""

pytest = __import__("pytest")
pytest.importorskip("hypothesis")

# ruff: noqa: E402 — imports must follow importorskip to avoid ImportError on collection
import pytest
from hypothesis import given, settings
from cuvis_ai_schemas.testing import MODEL_STRATEGIES, assert_dict_json_roundtrip

_MODELS = list(MODEL_STRATEGIES)


@pytest.mark.parametrize("model_cls", _MODELS, ids=lambda m: m.__name__)
def test_schema_model_json_roundtrip(model_cls) -> None:
    """Each schema model round-trips through dict -> JSON -> dict without loss."""
    strategy = MODEL_STRATEGIES[model_cls]

    @given(instance=strategy)
    @settings(max_examples=30)
    def _inner(instance) -> None:
        assert_dict_json_roundtrip(instance)

    _inner()
