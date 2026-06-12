"""OS / import helper utilities for plugin registration.

These helpers back the import-only plugin registration path. Plugins are
expected to be installed in the active environment already (an editable
``[tool.uv.sources]`` entry in dev, ``uv pip install`` / the ``provision``
helper, or the orchestrator's composed child venv). Registration is a plain
``importlib.import_module`` of each provided class; nothing here clones a
repo, installs dependencies, or mutates ``sys.path``.

Both the in-process front doors
(:meth:`cuvis_ai_core.utils.node_registry.NodeRegistry.register_plugins` /
``register_plugin``) and the orchestrator's child runtime
(:meth:`cuvis_ai_core.utils.node_registry.NodeRegistry.register_preinstalled`)
funnel through :func:`import_plugin_nodes`.
"""

from __future__ import annotations

import errno
import importlib
import inspect
import os
import shutil
import stat
import sys
import time
from pathlib import Path

from loguru import logger


def safe_rmtree(path: Path) -> None:
    """Remove a directory tree with Windows-friendly permission handling.

    Retained for the plugin-cache-clearing gRPC RPC
    (``PluginService.clear_plugin_cache``), which sweeps any leftover plugin
    cache directories.
    """

    def _handle_remove_readonly(func, target_path, exc_info):
        exc = exc_info[1]
        if isinstance(exc, PermissionError) or getattr(exc, "errno", None) in (
            errno.EACCES,
            errno.EPERM,
        ):
            try:
                os.chmod(target_path, stat.S_IWRITE)
            except OSError:
                pass
            func(target_path)
        else:
            raise exc

    last_error = None
    for attempt in range(3):
        try:
            shutil.rmtree(path, onerror=_handle_remove_readonly)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2)
                continue
            raise
        except FileNotFoundError:
            return
    if last_error:
        raise last_error


def _import_from_path(import_path: str, clear_cache: bool = False) -> type:
    """Import a class from a full module path.

    The plugin package must already be importable in the active environment;
    this does not install or clone anything.
    """
    try:
        parts = import_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid import path: '{import_path}'")

        module_path, class_name = parts

        if clear_cache:
            parts_to_clear = module_path.split(".")
            for i in range(len(parts_to_clear), 0, -1):
                partial_path = ".".join(parts_to_clear[:i])
                if partial_path in sys.modules:
                    del sys.modules[partial_path]
                    logger.debug(f"Cleared cached module: {partial_path}")

        module = importlib.import_module(module_path)

        if not hasattr(module, class_name):
            raise AttributeError(
                f"Module '{module_path}' has no class '{class_name}'. Available: {dir(module)}"
            )

        node_class = getattr(module, class_name)

        if not inspect.isclass(node_class):
            raise TypeError(f"'{import_path}' is not a class, got {type(node_class)}")

        return node_class

    except ImportError as exc:
        raise ImportError(
            f"Failed to import module for '{import_path}': {exc}\n"
            f"Ensure the module is installed and the path is correct."
        ) from exc
    except AttributeError as exc:
        raise AttributeError(f"Failed to load class '{import_path}': {exc}") from exc


def extract_package_prefixes(class_paths: list[str]) -> set[str]:
    """Extract top-level package prefixes from class paths.

    Args:
        class_paths: List of fully qualified class paths
                    (e.g., ["cuvis_ai.node.MyNode", "foo.bar.Baz"])

    Returns:
        Set of top-level package names (e.g., {"cuvis_ai", "foo"})

    Example:
        >>> extract_package_prefixes(["cuvis_ai.node.MyNode", "foo.bar.Baz"])
        {'cuvis_ai', 'foo'}
    """
    package_prefixes = set()
    for class_path in class_paths:
        top_package = class_path.split(".")[0]
        package_prefixes.add(top_package)
    return package_prefixes


def clear_package_modules(prefixes: set[str]) -> None:
    """Clear all modules with given package prefixes from sys.modules.

    Args:
        prefixes: Set of top-level package names to clear (e.g., {"cuvis_ai"})

    Example:
        >>> clear_package_modules({"cuvis_ai"})
        # Clears sys.modules["cuvis_ai"], sys.modules["cuvis_ai.node"], etc.
    """
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        for prefix in prefixes:
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                modules_to_clear.append(module_name)
                break

    for module_name in modules_to_clear:
        del sys.modules[module_name]
        logger.debug(f"Cleared cached module: {module_name}")

    if modules_to_clear:
        logger.info(
            f"Cleared {len(modules_to_clear)} cached modules for packages: {prefixes}"
        )


def import_plugin_nodes(
    class_paths: list[str], clear_cache: bool = True
) -> dict[str, type]:
    """Import node classes from fully qualified paths.

    Shared by the in-process front doors and the orchestrator's child runtime.
    Every plugin is already an installed package, so this only iterates the
    FQCN list and collects classes; no install, no clone, no ``sys.path``
    mutation.

    Args:
        class_paths: List of fully qualified class paths to import
                    (e.g., ["cuvis_ai.node.MyNode"])
        clear_cache: Whether to clear module cache before importing (default: True)

    Returns:
        Dict mapping class names to node classes
        (e.g., {"MyNode": <class 'cuvis_ai.node.MyNode'>})

    Raises:
        ImportError: If module import fails
        AttributeError: If class not found in module

    Example:
        >>> nodes = import_plugin_nodes(["cuvis_ai.node.MyNode"])
        >>> nodes["MyNode"]
        <class 'cuvis_ai.node.MyNode'>
    """
    imported_nodes = {}
    for class_path in class_paths:
        node_class = _import_from_path(class_path, clear_cache=clear_cache)
        class_name = node_class.__name__
        imported_nodes[class_name] = node_class
        logger.debug(f"Imported plugin node '{class_name}' from '{class_path}'")

    return imported_nodes


__all__ = [
    "safe_rmtree",
    "_import_from_path",
    "extract_package_prefixes",
    "clear_package_modules",
    "import_plugin_nodes",
]
