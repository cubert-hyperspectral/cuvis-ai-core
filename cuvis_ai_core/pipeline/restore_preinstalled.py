"""Preinstalled-venv restore: skip clone / install / sys.path mangling.

When the orchestrator runs a pipeline inside a child runtime, every
plugin in the child's venv is already a real installed Python package
(uv put it there during ``compose_env``). The traditional
``NodeRegistry.load_plugin`` flow — clone the source, install deps,
prepend to ``sys.path``, then ``importlib`` — is wrong for this
context: the work is done, and re-running it would shadow the
already-installed package.

This module exposes :func:`load_preinstalled_plugins`, which takes the
resolved plugin dict the parent computed via
:func:`cuvis_ai_core.utils.plugin_resolver.resolve_pipeline_plugins`
and registers each plugin's classes into a given session's
``NodeRegistry`` via a plain ``importlib.import_module``. The child
runtime calls this against ``session.node_registry`` in
``InitializeSession``, before ``LoadPipeline`` builds the pipeline
against that same registry.
"""

from __future__ import annotations

from typing import Mapping

from loguru import logger

from cuvis_ai_core.utils import git_and_os
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.plugin import GitPluginConfig, LocalPluginConfig

PluginConfig = GitPluginConfig | LocalPluginConfig


def load_preinstalled_plugins(
    registry: NodeRegistry,
    resolved_plugins: Mapping[str, PluginConfig],
) -> None:
    """Register classes from already-installed plugin packages.

    Mirrors the registration tail of ``NodeRegistry.load_plugin`` —
    same ``plugin_registry[class_name] = node_class`` shape and same
    ``plugin_configs[name] = config`` bookkeeping — without the clone
    / install / ``sys.path`` steps that the preinstalled venv has
    already obviated.
    """
    for name, config in resolved_plugins.items():
        imported_nodes = git_and_os.import_plugin_nodes(
            [node.class_name for node in config.provides],
            clear_cache=False,  # nothing stale to clear in a fresh child
        )
        for class_name, node_class in imported_nodes.items():
            registry.plugin_registry[class_name] = node_class
            logger.debug(
                f"Registered preinstalled plugin node '{class_name}' from '{name}'"
            )
        registry.plugin_configs[name] = config
        logger.info(
            f"Loaded preinstalled plugin '{name}' with {len(config.provides)} nodes"
        )
