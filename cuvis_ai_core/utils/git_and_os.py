"""Git and OS helper utilities."""

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
from typing import Optional

import git
from loguru import logger


def safe_rmtree(path: Path) -> None:
    """Remove a directory tree with Windows-friendly permission handling."""

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


def resolve_tag_commit(repo: "git.Repo", tag: str) -> Optional[str]:
    """Resolve a Git tag to its commit hash.

    Args:
        repo: GitPython Repo object
        tag: Tag name (e.g., v1.2.3)

    Returns:
        Commit hash if tag exists, None otherwise
    """
    tag = tag.strip()

    # Try with refs/tags/ prefix
    try:
        return repo.commit(f"refs/tags/{tag}").hexsha
    except Exception:
        pass

    # Try without prefix
    try:
        return repo.commit(tag).hexsha
    except Exception:
        pass

    return None


def _import_from_path(import_path: str, clear_cache: bool = False) -> type:
    """Import a class from a full module path."""
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


def _verify_tag_matches(repo_path: Path, expected_tag: str) -> bool:
    """Verify that cached repository is at the expected tag.

    Args:
        repo_path: Path to cached repository
        expected_tag: Expected tag name

    Returns:
        True if cache is at expected tag, False otherwise
    """
    try:
        repo = git.Repo(repo_path)
        current_commit = repo.head.commit.hexsha
        expected_tag = expected_tag.strip()

        resolved_commit = resolve_tag_commit(repo, expected_tag)

        # If tag not found in cache, try fetching it
        if resolved_commit is None:
            if repo.remotes:
                try:
                    repo.git.fetch(
                        "origin",
                        f"refs/tags/{expected_tag}:refs/tags/{expected_tag}",
                        depth=1,
                    )
                    resolved_commit = resolve_tag_commit(repo, expected_tag)
                except git.GitCommandError as exc:
                    logger.debug(
                        f"Failed to fetch tag '{expected_tag}' for cache verification: {exc}"
                    )

        if resolved_commit is None:
            logger.warning(f"Tag '{expected_tag}' not found in cached repo {repo_path}")
            return False

        return current_commit.startswith(resolved_commit)
    except Exception as exc:
        logger.warning(f"Cache verification failed for {repo_path}: {exc}")
        return False


def _clone_repository(repo_url: str, dest_path: Path, tag: str) -> Path:
    """Clone Git repository and checkout specific tag.

    Args:
        repo_url: Git repository URL
        dest_path: Destination path for clone
        tag: Git tag name (e.g., v1.2.3)

    Returns:
        Path to cloned repository

    Raises:
        RuntimeError: If tag not found or clone fails
    """
    logger.info(f"Cloning {repo_url} (tag: {tag}) to {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Shallow clone without specific branch
        repo = git.Repo.clone_from(repo_url, dest_path, branch=None, depth=1)

        # Fetch and checkout the specific tag
        try:
            repo.git.fetch("origin", f"refs/tags/{tag}:refs/tags/{tag}", depth=1)
            repo.git.checkout(tag)
            logger.info(f"Successfully checked out tag '{tag}'")
            return dest_path
        except git.GitCommandError as tag_error:
            # Tag not found - clean up and raise clear error
            if dest_path.exists():
                safe_rmtree(dest_path)
            raise RuntimeError(
                f"Tag '{tag}' not found in repository '{repo_url}'.\n"
                f"Only Git tags are supported (e.g., v1.2.3, v0.1.0-alpha).\n"
                f"Branches and commit hashes are NOT supported.\n"
                f"Error: {tag_error}"
            )
    except RuntimeError:
        # Re-raise our custom errors
        raise
    except Exception as exc:
        # Unexpected error - clean up
        if dest_path.exists():
            safe_rmtree(dest_path)
        raise RuntimeError(
            f"Failed to clone repository '{repo_url}' at tag '{tag}': {exc}"
        ) from exc


def _add_to_sys_path(path: Path) -> None:
    """Add path to sys.path if not already present."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        logger.debug(f"Added to sys.path: {path_str}")


def _install_plugin_dependencies(plugin_path: Path, plugin_name: str) -> None:
    """
    Detect and install plugin dependencies from pyproject.toml.

    This method enforces PEP 621 compliance by requiring plugins to have
    a pyproject.toml file with proper dependency specifications.
    """
    pyproject_file = plugin_path / "pyproject.toml"

    if not pyproject_file.exists():
        raise FileNotFoundError(
            f"Plugin '{plugin_name}' must have a pyproject.toml file.\n"
            f"PEP 621 (https://peps.python.org/pep-0621/) specifies pyproject.toml "
            f"as the standard for Python project metadata and dependencies.\n"
            f"Expected location: {pyproject_file}"
        )

    deps = _extract_deps_from_pyproject(pyproject_file)

    if not deps:
        logger.debug(f"No dependencies found for plugin '{plugin_name}'")
        return

    logger.info(f"Installing {len(deps)} dependencies for plugin '{plugin_name}'...")
    _install_dependencies_with_uv(deps, plugin_name)


def _extract_deps_from_pyproject(pyproject_path: Path) -> list[str]:
    """Extract dependencies from pyproject.toml using tomllib (Python 3.11+)."""
    import tomllib  # stdlib in Python 3.11+

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])

    filtered_deps = [
        dep.strip()
        for dep in deps
        if dep and dep.strip() and not dep.strip().startswith("#")
    ]

    logger.debug(f"Extracted {len(filtered_deps)} dependencies from {pyproject_path}")
    return filtered_deps


def _install_dependencies_with_uv(deps: list[str], plugin_name: str) -> None:
    """Install dependencies using 'uv pip install'."""
    import subprocess

    logger.info(f"Dependencies to install: {', '.join(deps)}")

    cmd = ["uv", "pip", "install"] + deps

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        logger.info(f"✓ Plugin '{plugin_name}' dependencies installed successfully")

        if result.stdout:
            logger.debug(f"uv output: {result.stdout}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Plugin '{plugin_name}' dependency installation timed out (>5 min). "
            f"This may indicate a network issue or very large dependencies."
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to install dependencies for plugin '{plugin_name}'.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error: {exc.stderr}\n\n"
            f"This may indicate version conflicts or missing packages. "
            f"uv could not resolve the dependency tree."
        ) from exc


def parse_plugin_config(
    name: str, config: dict, manifest_dir: Optional[Path] = None
) -> tuple:
    """Parse and validate plugin config, resolving paths and ensuring plugin exists.

    Args:
        name: Plugin identifier
        config: Plugin configuration dict with repo+tag or path+provides
        manifest_dir: Optional base directory for resolving local plugin paths

    Returns:
        Tuple of (plugin_config, plugin_path) where:
        - plugin_config: Validated GitPluginConfig or LocalPluginConfig
        - plugin_path: Path to the plugin directory

    Raises:
        ValueError: If config missing repo/path or plugin validation fails
        FileNotFoundError: If local plugin path doesn't exist
    """
    from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig

    # Validate and parse config
    if "repo" in config:
        plugin_config = GitPluginConfig.model_validate(config)
        # For Git plugins, we need to ensure it's cloned/cached
        from cuvis_ai_core.utils.node_registry import NodeRegistry

        plugin_path = NodeRegistry._ensure_git_plugin(name, plugin_config)
    elif "path" in config:
        if manifest_dir is not None:
            # Resolve local paths relative to manifest directory
            config = dict(config)
            config["path"] = str(LocalPluginConfig(**config).resolve_path(manifest_dir))
        plugin_config = LocalPluginConfig.model_validate(config)
        # For local plugins, validate the path exists
        from cuvis_ai_core.utils.node_registry import NodeRegistry

        plugin_path = NodeRegistry._ensure_local_plugin(name, plugin_config)
    else:
        raise ValueError(
            f"Plugin '{name}' must have either 'repo' (Git) or 'path' (local)"
        )

    return plugin_config, plugin_path


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
        # Extract top-level package name
        top_package = class_path.split(".")[0]
        package_prefixes.add(top_package)
    return package_prefixes


def clear_package_modules(prefixes: set[str]) -> None:
    """Clear all modules with given package prefixes from sys.modules.

    This is critical when loading plugins that may conflict with already-imported
    packages. For example, if cuvis_ai is installed in the environment and we're
    loading it as a plugin from a different path, we need to clear ALL cuvis_ai.*
    modules to ensure Python reimports from the plugin path.

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
        # Import the class with cache clearing if requested
        node_class = _import_from_path(class_path, clear_cache=clear_cache)
        class_name = node_class.__name__
        imported_nodes[class_name] = node_class
        logger.debug(f"Imported plugin node '{class_name}' from '{class_path}'")

    return imported_nodes


__all__ = [
    "safe_rmtree",
    "resolve_tag_commit",
    "_import_from_path",
    "_verify_tag_matches",
    "_clone_repository",
    "_add_to_sys_path",
    "_install_plugin_dependencies",
    "_extract_deps_from_pyproject",
    "_install_dependencies_with_uv",
    "parse_plugin_config",
    "extract_package_prefixes",
    "clear_package_modules",
    "import_plugin_nodes",
]
