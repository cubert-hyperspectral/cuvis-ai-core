"""Shared model-weight cache for child runtimes.

The spawner points each per-run child at ONE persistent weight cache instead of
letting ``huggingface_hub`` / ``torch.hub`` / ``timm`` write into the per-run
scratch ``HOME`` (which is wiped every run). Name-keyed loaders (huggingface_hub,
timm, anomalib, OpenCLIP) and path-keyed loaders then resolve weights offline
after the first fetch, and an operator can pre-populate the cache once for
offline / shipped installs.

Import-light on purpose: module-level imports are stdlib only, and the composer
cache-root lookup is imported lazily inside the function, so importing this
module never drags ``torch`` or the heavy runtime graph.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

# Operator override for the shared cache location. When unset the cache lives
# under the composer's cache root so venv + weight caches share one tree.
_MODEL_CACHE_DIR_ENV = "CUVIS_MODEL_CACHE_DIR"
_MODEL_CACHE_DIRNAME = "model_cache"

# HF cache vars an operator may already point at their own cache. If any is
# already set we leave HF alone rather than override the operator's choice.
# ``HF_HOME`` is deliberately excluded as an INJECTION target: setting it would
# relocate the HF token file into the shared dir. The child never receives a
# token anyway (the spawner strips it) and runs offline against the cache.
_HF_CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")


def model_cache_dir() -> Path:
    """Resolve the shared model-weight cache directory.

    ``$CUVIS_MODEL_CACHE_DIR`` when set, else ``<composer cache root>/model_cache``.
    """
    override = os.environ.get(_MODEL_CACHE_DIR_ENV)
    if override:
        return Path(override)
    # Lazy import keeps this module free of the composer's (and its transitive)
    # import cost for light consumers and tests.
    from cuvis_ai_core.orchestrator.composer import resolve_cache_root

    return resolve_cache_root(None) / _MODEL_CACHE_DIRNAME


def model_cache_env(parent_env: Mapping[str, str]) -> dict[str, str]:
    """Env additions that point the child's weight caches at the shared dir.

    Sets ``HF_HUB_CACHE`` (+ legacy ``HUGGINGFACE_HUB_CACHE``) and ``TORCH_HOME``
    unless the operator already set them in ``parent_env`` (their choice wins).
    Also sets ``HF_HUB_OFFLINE=1`` (unless the operator already set it) so the
    child resolves weights from the pre-provisioned cache rather than reaching
    the network: the child runs untrusted plugin code and gets no HF token, so a
    gated fetch could not succeed anyway. Gated weights are provisioned
    out-of-band by a trusted tool (the ``download-model`` CLI or the CuvisNEXT
    action). Creates the target dirs so the first write succeeds.
    """
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    # Single discoverable root, always exported. Plugins whose bespoke loaders
    # (gdown, plain urllib, anomalib's DinoV2Loader) do NOT honor
    # HF_HUB_CACHE/TORCH_HOME read this to place their cache under the shared
    # tree instead of the wiped per-run HOME.
    additions: dict[str, str] = {"CUVIS_MODEL_CACHE_DIR": str(cache)}

    if not any(parent_env.get(var) for var in _HF_CACHE_VARS):
        hf_hub = cache / "hf"
        hf_hub.mkdir(parents=True, exist_ok=True)
        additions["HF_HUB_CACHE"] = str(hf_hub)
        additions["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)

    if not parent_env.get("TORCH_HOME"):
        torch_home = cache / "torch"
        torch_home.mkdir(parents=True, exist_ok=True)
        additions["TORCH_HOME"] = str(torch_home)

    # Resolve HF weights from the pre-provisioned cache only; never reach the
    # network from the untrusted, token-less child. An operator override wins.
    if not parent_env.get("HF_HUB_OFFLINE"):
        additions["HF_HUB_OFFLINE"] = "1"

    return additions
