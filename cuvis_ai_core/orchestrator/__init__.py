"""Per-run child-env orchestrator.

Composes a cached venv per pipeline plugin set, then spawns a child
runtime service inside it. The server itself never imports plugin
modules.
"""

from cuvis_ai_core.orchestrator.cache_key import CacheKey, compute_cache_key
from cuvis_ai_core.orchestrator.composer import ComposerError, compose_env
from cuvis_ai_core.orchestrator.plugin_capabilities import (
    NodePortSpec,
    PluginCapabilities,
    PluginCapabilityEntry,
    load_capabilities,
)
from cuvis_ai_core.orchestrator.venv_paths import venv_bin_dir, venv_python

__all__ = [
    "CacheKey",
    "ComposerError",
    "NodePortSpec",
    "PluginCapabilities",
    "PluginCapabilityEntry",
    "compose_env",
    "compute_cache_key",
    "load_capabilities",
    "venv_bin_dir",
    "venv_python",
]
