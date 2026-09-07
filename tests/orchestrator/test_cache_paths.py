"""Tests for the cache-root leaf shared by the composer, the model cache and the crash logs."""

from __future__ import annotations

import ast
import inspect

from cuvis_ai_core.orchestrator import cache_paths, composer, crash_logs, model_cache


def test_resolve_cache_root_precedence(monkeypatch, tmp_path):
    """override -> $CUVIS_RUN_CACHE_DIR -> ~/.cuvis_runs."""
    monkeypatch.delenv(cache_paths.RUN_CACHE_ROOT_ENV, raising=False)
    assert cache_paths.resolve_cache_root(None) == cache_paths.DEFAULT_RUN_CACHE_ROOT
    monkeypatch.setenv(cache_paths.RUN_CACHE_ROOT_ENV, str(tmp_path / "from-env"))
    assert cache_paths.resolve_cache_root(None) == tmp_path / "from-env"
    assert (
        cache_paths.resolve_cache_root(tmp_path / "override") == tmp_path / "override"
    )


def test_composer_and_model_cache_share_constants(monkeypatch, tmp_path):
    """The composer's private aliases and the model cache read the one leaf."""
    assert composer._DEFAULT_CACHE_ROOT_ENV == cache_paths.RUN_CACHE_ROOT_ENV
    assert composer._DEFAULT_CACHE_ROOT == cache_paths.DEFAULT_RUN_CACHE_ROOT
    assert composer.resolve_cache_root is cache_paths.resolve_cache_root
    monkeypatch.delenv(cache_paths.MODEL_CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv("CUVIS_RUNTIME_CRASH_DIR", raising=False)
    monkeypatch.setenv(cache_paths.RUN_CACHE_ROOT_ENV, str(tmp_path))
    assert model_cache.model_cache_dir() == tmp_path / cache_paths.MODEL_CACHE_DIRNAME
    assert crash_logs.crash_dir_root() == tmp_path / crash_logs.CRASH_LOGS_DIRNAME


def _imported_modules(module) -> set[str]:
    """Every module an ``import`` / ``from ... import`` statement in ``module`` names."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_leaf_and_light_consumers_never_import_the_composer():
    """The layering fix: the leaf is stdlib-only and its consumers import it, not the composer.

    (The package ``__init__`` still imports the composer for its public API, so
    ``sys.modules`` cannot show this; the modules' own import statements can.)
    """
    leaf_imports = _imported_modules(cache_paths)
    assert not any(name.startswith("cuvis_ai_core") for name in leaf_imports), (
        leaf_imports
    )
    for consumer in (model_cache, crash_logs):
        imports = _imported_modules(consumer)
        assert "cuvis_ai_core.orchestrator.cache_paths" in imports, consumer.__name__
        assert "cuvis_ai_core.orchestrator.composer" not in imports, consumer.__name__
