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
and registers each plugin's classes into the session's
``NodeRegistry`` via a plain ``importlib.import_module``. Sibling
helpers :func:`restore_pipeline_preinstalled` and
:func:`restore_trainrun_preinstalled` are thin wrappers that load the
plugins first, then delegate to the regular restore entry points with
the per-pipeline install path disabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from cuvis_ai_core.utils import git_and_os
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig

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
                f"Registered preinstalled plugin node '{class_name}' "
                f"from '{name}'"
            )
        registry.plugin_configs[name] = config
        logger.info(
            f"Loaded preinstalled plugin '{name}' with "
            f"{len(config.provides)} nodes"
        )


def restore_pipeline_preinstalled(
    pipeline_path: str | Path,
    resolved_plugins: Mapping[str, PluginConfig],
    *,
    weights_path: str | Path | None = None,
    device: str = "auto",
    **kwargs: Any,
):
    """``restore_pipeline`` against a venv whose plugins are already installed.

    Delegates to :func:`cuvis_ai_core.utils.restore.restore_pipeline`
    with the manifest-driven ``plugins_path`` / ``plugins_dirs``
    arguments left unset so the legacy clone+install branch is never
    reached. Plugin class registration happens here, before the
    pipeline factory tries to resolve any class name.
    """
    from cuvis_ai_core.utils.restore import restore_pipeline

    # The pipeline builder reaches for nodes via the
    # *global* NodeRegistry; we register classes there so
    # ``CuvisPipeline`` lookups resolve preinstalled plugins by class
    # name without any session plumbing.
    load_preinstalled_plugins(NodeRegistry(), resolved_plugins)
    return restore_pipeline(
        pipeline_path,
        weights_path=weights_path,
        device=device,
        plugins_path=None,
        plugins_dirs=None,
        **kwargs,
    )


def restore_trainrun_preinstalled(
    trainrun_path: str | Path,
    resolved_plugins: Mapping[str, PluginConfig],
    *,
    mode: str = "info",
    **kwargs: Any,
):
    """``restore_trainrun`` against a venv whose plugins are already installed."""
    from cuvis_ai_core.utils.restore import restore_trainrun

    load_preinstalled_plugins(NodeRegistry(), resolved_plugins)
    return restore_trainrun(trainrun_path, mode=mode, **kwargs)
