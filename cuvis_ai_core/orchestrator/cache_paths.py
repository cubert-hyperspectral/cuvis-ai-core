"""Cache-root constants shared by the composer, the model cache and the crash logs.

A leaf on purpose: stdlib only, no imports from the rest of the orchestrator.
Light consumers (``model_cache``, ``crash_logs``, ``data.model_weights``) used to
import the composer lazily just to learn where the cache root is, which inverted
the layering (a leaf depending on the heaviest module) and forced every reader
to defer the import. They import this module at top level instead.
"""

from __future__ import annotations

import os
from pathlib import Path

RUN_CACHE_ROOT_ENV = "CUVIS_RUN_CACHE_DIR"
"""Operator override for the cache root that holds composed envs and the model cache."""

DEFAULT_RUN_CACHE_ROOT = Path.home() / ".cuvis_runs"
"""Cache root when :data:`RUN_CACHE_ROOT_ENV` is unset."""

MODEL_CACHE_DIRNAME = "model_cache"
"""Name of the shared model-weight cache directory under the cache root."""

MODEL_CACHE_DIR_ENV = "CUVIS_MODEL_CACHE_DIR"
"""Operator override for the shared model-weight cache location."""


def resolve_cache_root(override: Path | None = None) -> Path:
    """Resolve the composed-env / model cache root.

    Resolves ``override`` -> ``$CUVIS_RUN_CACHE_DIR`` -> the default
    ``~/.cuvis_runs``. Shared with the model-weight cache (``model_cache``) so
    the venv and weight caches sit under one root.
    """
    if override is not None:
        return Path(override)
    env_val = os.environ.get(RUN_CACHE_ROOT_ENV)
    if env_val:
        return Path(env_val)
    return DEFAULT_RUN_CACHE_ROOT
