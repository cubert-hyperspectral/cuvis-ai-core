"""Node registry for managing built-in and custom node types."""

import importlib
import inspect
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Union
from cuvis_ai_core.utils.plugin_config import (
    GitPluginConfig,
    LocalPluginConfig,
)


try:
    import git
except ImportError:
    git = None  # Git operations will fail gracefully with clear error

from loguru import logger


class NodeRegistry:
    """
    Hybrid registry supporting both built-in and custom nodes.

    Built-in nodes are registered via @register decorator for O(1) lookup.
    Custom nodes are loaded via importlib using full import paths.

    Examples:
        # Built-in node (registry lookup)
        normalizer_class = NodeRegistry.get("MinMaxNormalizer")

        # Custom node (importlib)
        custom_class = NodeRegistry.get("my_package.nodes.CustomRXDetector")
    """

    _builtin_registry: Dict[str, type] = {}  # Existing: @register decorated nodes
    _plugin_registry: Dict[str, type] = {}  # NEW: Plugin nodes by class name
    _plugin_configs: Dict[
        str, Union[GitPluginConfig, LocalPluginConfig]
    ] = {}  # NEW: Track loaded plugins
    _plugin_class_map: Dict[str, str] = {}  # NEW: class_path → plugin_name
    _cache_dir: Path = Path.home() / ".cuvis_plugins"  # NEW: Git cache directory

    @classmethod
    def register(cls, node_class: type) -> type:
        """
        Decorator to register a built-in node class.

        Args:
            node_class: The node class to register

        Returns:
            The same node class (for decorator chaining)

        Example:
            @NodeRegistry.register
            class MinMaxNormalizer(Node):
                pass
        """
        class_name = node_class.__name__

        if class_name in cls._builtin_registry:
            raise ValueError(
                f"Node '{class_name}' is already registered. "
                f"Existing: {cls._builtin_registry[class_name]}, "
                f"New: {node_class}"
            )

        cls._builtin_registry[class_name] = node_class
        return node_class

    @classmethod
    def get(cls, class_identifier: str) -> type:
        """
        Get node class by name or full import path.

        Resolution order:
        1. Check built-in registry (O(1) lookup)
        2. Try importlib for full paths (e.g., "my_package.MyNode")
        3. Raise clear error if not found

        Args:
            class_identifier: Either a simple class name for built-in nodes
                            or full import path for custom nodes

        Returns:
            The node class

        Raises:
            KeyError: If node not found in registry or via import
            ImportError: If custom node path is invalid
            AttributeError: If module doesn't contain the class

        Examples:
            # Built-in node
            cls = NodeRegistry.get("MinMaxNormalizer")

            # Custom node with full path
            cls = NodeRegistry.get("my_company.detectors.AdvancedRXDetector")
        """
        # Try built-in registry first
        if class_identifier in cls._builtin_registry:
            return cls._builtin_registry[class_identifier]

        # Try plugin registry (NEW)
        if class_identifier in cls._plugin_registry:
            return cls._plugin_registry[class_identifier]

        # For full paths, also check if last component is in plugin registry
        if "." in class_identifier:
            class_name = class_identifier.rsplit(".", 1)[1]
            if class_name in cls._plugin_registry:
                return cls._plugin_registry[class_name]

        # Try importlib for custom nodes (must have dot in path)
        if "." in class_identifier:
            return cls._import_from_path(class_identifier)

        # Not found
        available = cls.list_builtin_nodes() + cls.list_plugin_nodes()
        raise KeyError(
            f"Node '{class_identifier}' not found in registry.\n"
            f"For custom nodes, provide full import path (e.g., 'my_package.MyNode').\n"
            f"Available nodes: {available}"
        )

    @classmethod
    def _import_from_path(cls, import_path: str) -> type:
        """
        Import a class from a full module path.

        Args:
            import_path: Full import path (e.g., "my_package.nodes.CustomNode")

        Returns:
            The imported class

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If class doesn't exist in module
        """
        try:
            # Split into module path and class name
            parts = import_path.rsplit(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid import path: '{import_path}'")

            module_path, class_name = parts

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            if not hasattr(module, class_name):
                raise AttributeError(
                    f"Module '{module_path}' has no class '{class_name}'. Available: {dir(module)}"
                )

            node_class = getattr(module, class_name)

            # Verify it's a class
            if not inspect.isclass(node_class):
                raise TypeError(
                    f"'{import_path}' is not a class, got {type(node_class)}"
                )

            return node_class

        except ImportError as e:
            raise ImportError(
                f"Failed to import module for '{import_path}': {e}\n"
                f"Ensure the module is installed and the path is correct."
            ) from e
        except AttributeError as e:
            raise AttributeError(f"Failed to load class '{import_path}': {e}") from e

    @classmethod
    def list_builtin_nodes(cls) -> list[str]:
        """
        List all registered built-in node names.

        Returns:
            Sorted list of node class names
        """
        return sorted(cls._builtin_registry.keys())

    @classmethod
    def auto_register_package(
        cls, package_name: str, base_class_path: str = "cuvis_ai_core.node.node.Node"
    ) -> int:
        """
        Auto-register all Node classes from a package.

        Searches the package for classes that inherit from Node
        and registers them automatically.

        Args:
            package_name: Full package name (e.g., "cuvis_ai.node")
            base_class_path: Full import path to the base Node class

        Returns:
            Number of classes registered

        Example:
            NodeRegistry.auto_register_package("cuvis_ai.node")
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            raise ImportError(f"Failed to import package '{package_name}': {e}") from e

        # Get the package directory
        if not hasattr(package, "__path__"):
            raise ValueError(f"'{package_name}' is not a package (has no __path__)")

        # Import the base Node class first, outside the loop
        try:
            Node = cls._import_from_path(base_class_path)
        except Exception as e:
            raise ImportError(
                f"Failed to import base class '{base_class_path}': {e}"
            ) from e

        registered_count = 0

        # Import all modules in the package
        package_dir = Path(package.__path__[0])
        for module_file in package_dir.glob("*.py"):
            if module_file.name.startswith("_"):
                continue

            module_name = f"{package_name}.{module_file.stem}"
            try:
                module = importlib.import_module(module_name)

                # Find all Node subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    try:
                        # Check if this is a Node subclass
                        if issubclass(obj, Node) and obj is not Node:
                            # Only register if defined in this module
                            if obj.__module__ == module_name:
                                cls.register(obj)
                                registered_count += 1
                    except TypeError:
                        # issubclass can fail for some special class-like objects
                        # Skip these and continue
                        continue

            except Exception as e:
                # Log warning but continue
                print(f"Warning: Failed to auto-register from {module_name}: {e}")

        return registered_count

    @classmethod
    def list_plugin_nodes(cls) -> list[str]:
        """
        List all nodes provided by plugins.

        Returns:
            Sorted list of plugin node class names
        """
        return sorted(cls._plugin_registry.keys())

    @classmethod
    def load_plugins(cls, manifest_path: Union[str, Path]) -> int:
        """
        Load multiple plugins from a YAML manifest file.

        Args:
            manifest_path: Path to plugins.yaml file

        Returns:
            Number of plugins loaded

        Example:
            count = NodeRegistry.load_plugins("plugins.yaml")
            print(f"Loaded {count} plugins")
        """
        from cuvis_ai_core.utils.plugin_config import PluginManifest

        manifest_path = Path(manifest_path)
        manifest = PluginManifest.from_yaml(manifest_path)
        manifest_dir = manifest_path.resolve().parent

        loaded = 0
        for plugin_name, config in manifest.plugins.items():
            cls.load_plugin(plugin_name, config.model_dump(), manifest_dir=manifest_dir)
            loaded += 1

        logger.info(f"Loaded {loaded} plugins from {manifest_path}")
        return loaded

    @classmethod
    def load_plugin(
        cls,
        name: str,
        config: dict,
        manifest_dir: Optional[Path] = None,
    ) -> None:
        """
        Load a single plugin from a configuration dict.

        Args:
            name: Plugin identifier (e.g., "adaclip")
            config: Plugin configuration dict with:
                - repo + ref (for Git plugins)
                - path (for local plugins)
                - provides: list of class paths

        Examples:
            # Git plugin
            NodeRegistry.load_plugin("adaclip", {
                "repo": "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git",
                "ref": "v1.2.3",
                "provides": ["cuvis_ai_adaclip.node.AdaCLIPDetector"]
            })

            # Local plugin
            NodeRegistry.load_plugin("local_dev", {
                "path": "../my-plugin",
                "provides": ["my_plugin.MyNode"]
            })
        """
        from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig

        # Already loaded?
        if name in cls._plugin_configs:
            logger.debug(f"Plugin '{name}' already loaded, skipping")
            return

        # Validate and parse config
        if "repo" in config:
            plugin_config = GitPluginConfig.model_validate(config)
            plugin_path = cls._ensure_git_plugin(name, plugin_config)
        elif "path" in config:
            if manifest_dir is not None:
                config = dict(config)
                config["path"] = str(
                    LocalPluginConfig(**config).resolve_path(manifest_dir)
                )
            plugin_config = LocalPluginConfig.model_validate(config)
            plugin_path = cls._ensure_local_plugin(name, plugin_config)
        else:
            raise ValueError(
                f"Plugin '{name}' must have either 'repo' (Git) or 'path' (local)"
            )

        # Add to sys.path
        cls._add_to_sys_path(plugin_path)

        # Import and register all provided classes
        for class_path in plugin_config.provides:
            try:
                node_class = cls._import_from_path(class_path)
                class_name = node_class.__name__

                # Register in plugin registry
                cls._plugin_registry[class_name] = node_class
                cls._plugin_class_map[class_path] = name

                logger.debug(f"Registered plugin node '{class_name}' from '{name}'")
            except Exception as e:
                logger.warning(
                    f"Failed to import '{class_path}' from plugin '{name}': {e}"
                )

        # Track plugin config
        cls._plugin_configs[name] = plugin_config
        logger.info(f"Loaded plugin '{name}' with {len(plugin_config.provides)} nodes")

    @classmethod
    def unload_plugin(cls, name: str) -> None:
        """
        Unload a plugin and remove its nodes from the registry.

        Args:
            name: Plugin identifier to unload

        Note:
            Does not remove the plugin from sys.path (Python limitation).
            Cached Git repositories are NOT deleted (use clear_plugin_cache).
        """
        if name not in cls._plugin_configs:
            logger.warning(f"Plugin '{name}' not loaded, nothing to unload")
            return

        config = cls._plugin_configs[name]

        # Remove nodes from registry
        for class_path in config.provides:
            class_name = class_path.rsplit(".", 1)[1]
            cls._plugin_registry.pop(class_name, None)
            cls._plugin_class_map.pop(class_path, None)

        # Remove config
        del cls._plugin_configs[name]
        logger.info(f"Unloaded plugin '{name}'")

    @classmethod
    def list_plugins(cls) -> list[str]:
        """
        List all loaded plugin names.

        Returns:
            Sorted list of plugin names
        """
        return sorted(cls._plugin_configs.keys())

    @classmethod
    def clear_plugins(cls) -> None:
        """Unload all plugins and clear plugin registries."""
        cls._plugin_registry.clear()
        cls._plugin_configs.clear()
        cls._plugin_class_map.clear()
        logger.info("Cleared all plugins")

    @classmethod
    def clear_plugin_cache(cls, plugin_name: Optional[str] = None) -> None:
        """
        Clear cached Git repositories.

        Args:
            plugin_name: If provided, clear only this plugin's cache.
                        If None, clear all cached plugins.
        """
        if plugin_name:
            for cache_entry in cls._cache_dir.glob(f"{plugin_name}@*"):
                logger.info(f"Removing cache: {cache_entry}")
                shutil.rmtree(cache_entry)
        else:
            if cls._cache_dir.exists():
                logger.info(f"Clearing all plugin caches in {cls._cache_dir}")
                shutil.rmtree(cls._cache_dir)

    @classmethod
    def set_cache_dir(cls, path: Union[str, Path]) -> None:
        """
        Set the cache directory for Git plugins.

        Args:
            path: Directory path for caching cloned repositories
        """
        cls._cache_dir = Path(path)
        logger.debug(f"Plugin cache directory set to: {cls._cache_dir}")

    # === Internal Plugin Helpers ===

    @classmethod
    def _ensure_git_plugin(cls, plugin_name: str, config: "GitPluginConfig") -> Path:
        """Clone or reuse cached Git repository."""
        if git is None:
            raise ImportError(
                "GitPython is required for Git plugins. "
                "Install with: uv add gitpython>=3.1.40"
            )

        cache_dir = cls._cache_dir / f"{plugin_name}@{config.ref}"

        if cache_dir.exists():
            if cls._verify_ref_matches(cache_dir, config.ref):
                logger.info(f"Using cached plugin '{plugin_name}' at {cache_dir}")
                return cache_dir
            else:
                logger.warning(f"Cache mismatch for '{plugin_name}', re-cloning...")
                shutil.rmtree(cache_dir)

        return cls._clone_repository(config.repo, cache_dir, config.ref)

    @classmethod
    def _ensure_local_plugin(cls, plugin_name: str, config: LocalPluginConfig) -> Path:
        """Resolve and validate local plugin path."""
        plugin_path = Path(config.path)
        if not plugin_path.is_absolute():
            plugin_path = plugin_path.resolve()

        if not plugin_path.exists():
            raise FileNotFoundError(f"Local plugin path not found: {plugin_path}")

        if not plugin_path.is_dir():
            raise ValueError(f"Plugin path must be a directory: {plugin_path}")

        logger.info(f"Using local plugin '{plugin_name}' at {plugin_path}")
        return plugin_path

    @classmethod
    def _verify_ref_matches(cls, repo_path: Path, expected_ref: str) -> bool:
        """Verify that cached repository is at the expected ref."""
        try:
            repo = git.Repo(repo_path)
            current_commit = repo.head.commit.hexsha[:7]

            # Try as tag
            if expected_ref in [tag.name for tag in repo.tags]:
                return current_commit == repo.tags[expected_ref].commit.hexsha[:7]

            # Try as branch
            remote_branches = [ref.name for ref in repo.refs if "origin/" in ref.name]
            if f"origin/{expected_ref}" in remote_branches:
                return (
                    current_commit == repo.commit(f"origin/{expected_ref}").hexsha[:7]
                )

            # Try as commit hash
            return current_commit == expected_ref[:7]
        except Exception as e:
            logger.warning(f"Cache verification failed for {repo_path}: {e}")
            return False

    @classmethod
    def _clone_repository(cls, repo_url: str, dest_path: Path, ref: str) -> Path:
        """Clone Git repository and checkout specific ref."""
        logger.info(f"Cloning {repo_url} (ref: {ref}) to {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            repo = git.Repo.clone_from(repo_url, dest_path, branch=None, depth=1)
            try:
                repo.git.checkout(ref)
            except git.GitCommandError:
                logger.info(f"Ref '{ref}' not in shallow clone, fetching...")
                repo.git.fetch("origin", ref, depth=1)
                repo.git.checkout(ref)

            logger.info(f"Successfully cloned and checked out {ref}")
            return dest_path
        except git.GitCommandError as e:
            if dest_path.exists():
                shutil.rmtree(dest_path)
            raise RuntimeError(
                f"Failed to clone repository '{repo_url}' at ref '{ref}': {e}"
            ) from e

    @classmethod
    def _add_to_sys_path(cls, path: Path) -> None:
        """Add path to sys.path if not already present."""
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            logger.debug(f"Added to sys.path: {path_str}")

    @classmethod
    def clear(cls):
        """Clear all registered nodes (primarily for testing)."""
        cls._builtin_registry.clear()
