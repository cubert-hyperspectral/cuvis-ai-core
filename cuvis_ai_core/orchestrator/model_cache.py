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


def hf_cache_dir(env: Mapping[str, str]) -> Path:
    """Resolve the ONE Hugging Face hub cache shared by provisioner + child.

    Mirrors ``huggingface_hub``'s own precedence so a pre-provisioned weight
    (written by ``download-model`` in the trusted parent) and the offline child
    always target the SAME directory:

    ``$HF_HUB_CACHE`` -> ``$HUGGINGFACE_HUB_CACHE`` -> ``$HF_HOME/hub`` ->
    ``<shared model cache>/hf`` (the clean-host default).

    Keyed off the standard HF variables, so no cuvis-specific variable is
    introduced for HF weights. ``HF_HOME`` is honored as a *read* here (its
    ``hub`` subdir) but is never re-exported to the child, which would relocate
    the HF token file; the child instead gets ``HF_HUB_CACHE`` pointing here,
    and that outranks ``HF_HOME/hub`` in ``huggingface_hub``.
    """
    explicit = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = env.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return model_cache_dir() / "hf"


def model_cache_env(parent_env: Mapping[str, str]) -> dict[str, str]:
    """Env additions that point the child's weight caches at the shared dir.

    Always sets ``HF_HUB_CACHE`` (+ legacy ``HUGGINGFACE_HUB_CACHE``) to the one
    HF cache both the provisioner and the child resolve to (:func:`hf_cache_dir`:
    an operator ``HF_HUB_CACHE`` / ``HF_HOME`` wins, else the shared cache), so a
    pre-provisioned weight and the offline child never disagree. Sets
    ``TORCH_HOME`` and ``HF_HUB_OFFLINE=1`` unless the operator already set them.
    ``HF_HUB_OFFLINE=1`` keeps the untrusted, token-less child off the network;
    gated weights are provisioned out-of-band by a trusted tool (the
    ``download-model`` CLI or the CuvisNEXT action). Creates the target dirs so
    the first write succeeds.
    """
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    # Single discoverable root, always exported. Plugins whose bespoke loaders
    # (gdown, plain urllib, anomalib's DinoV2Loader) do NOT honor
    # HF_HUB_CACHE/TORCH_HOME read this to place their cache under the shared
    # tree instead of the wiped per-run HOME.
    additions: dict[str, str] = {"CUVIS_MODEL_CACHE_DIR": str(cache)}

    # Point HF at ONE cache resolved identically by the trusted provisioner
    # (download-model) and this child, and export it explicitly so the child's
    # wiped HOME cannot redirect it. HF_HUB_CACHE outranks HF_HOME/hub in
    # huggingface_hub, so this wins even when the operator manages their cache
    # via HF_HOME.
    hf_hub = hf_cache_dir(parent_env)
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
