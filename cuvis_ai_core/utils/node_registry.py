"""Node registry for managing built-in and custom node types."""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger

import cuvis_ai_core.utils.git_and_os as git_os
from cuvis_ai_core.utils.plugin_config import (
    GitPluginConfig,
    LocalPluginConfig,
)


class NodeRegistry:
    """
    Hybrid registry supporting both class and instance usage.

    CLASS MODE (built-ins only):
        NodeRegistry.get("MinMaxNormalizer")  # No instantiation needed

    INSTANCE MODE (built-ins + plugins):
        registry = NodeRegistry()
        registry.load_plugin("adaclip", config)
        registry.get("AdaCLIPDetector")  # Access both built-ins and plugins

    Built-in nodes are registered via @register decorator for O(1) lookup.
    Plugin nodes are loaded into instances for session isolation.
    """

    # ========== CLASS-LEVEL: Built-in nodes (singleton) ==========
    _builtin_registry: Dict[str, type] = {}
    _cache_dir: Path = Path.home() / ".cuvis_plugins"

    def __init__(self):
        """Create instance for plugin support."""
        self.plugin_registry: Dict[str, type] = {}
        self.plugin_configs: Dict[str, Union[GitPluginConfig, LocalPluginConfig]] = {}
        self.plugin_class_map: Dict[str, str] = {}
        self.cache_dir: Path = Path.home() / ".cuvis_plugins"

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
    def get(
        cls, class_identifier: str, instance: Optional["NodeRegistry"] = None
    ) -> type:
        """
        Get node class - works as both class and instance method!

        Auto-detects usage:
        - Class call → only built-ins
        - Instance call → built-ins + plugins

        Resolution order (instance mode):
        1. Check instance plugins (if instance mode)
        2. Check built-in registry (O(1) lookup)
        3. Try importlib for full paths (e.g., "my_package.MyNode")

        Args:
            class_identifier: Either a simple class name for built-in nodes
                            or full import path for custom nodes
            instance: Optional NodeRegistry instance for plugin lookup (auto-filled when called on instance)

        Returns:
            The node class

        Raises:
            KeyError: If node not found in registry or via import
            ImportError: If custom node path is invalid
            AttributeError: If module doesn't contain the class

        Examples:
            # Built-in node (class call)
            cls = NodeRegistry.get("MinMaxNormalizer")

            # Plugin node (instance call)
            registry = NodeRegistry()
            registry.load_plugin("adaclip", config)
            cls = registry.get("AdaCLIPDetector")

            # Custom node with full path
            cls = NodeRegistry.get("my_company.detectors.AdvancedRXDetector")
        """
        # 1. Check instance plugins first (if instance provided)
        if instance is not None:
            if class_identifier in instance.plugin_registry:
                return instance.plugin_registry[class_identifier]
            # For full paths, also check if last component is in plugins
            if "." in class_identifier:
                class_name = class_identifier.rsplit(".", 1)[1]
                if class_name in instance.plugin_registry:
                    return instance.plugin_registry[class_name]

        # 2. Check built-in registry (both modes)
        if class_identifier in cls._builtin_registry:
            return cls._builtin_registry[class_identifier]

        # 3. Try importlib for custom nodes (must have dot in path)
        if "." in class_identifier:
            return cls._import_from_path(class_identifier)

        # Not found - provide helpful error
        available = cls.list_builtin_nodes()
        if instance is not None:
            available = available + sorted(instance.plugin_registry.keys())

        # Check if it looks like a plugin node (has multiple dots or known plugin pattern)
        looks_like_plugin = class_identifier.count(".") >= 2 or any(
            pkg in class_identifier.lower()
            for pkg in ["plugin", "adaclip", "cuvis_ai_"]
        )

        error_msg = f"Node '{class_identifier}' not found in registry.\n"

        if looks_like_plugin and instance is None:
            error_msg += (
                "\n⚠️  This appears to be an external plugin node!\n"
                "   Did you forget to load plugins?\n\n"
                "   For CLI usage:\n"
                "     uv run restore-pipeline --pipeline-path <path> --plugins-path examples/adaclip/plugins.yaml\n\n"
                "   For Python usage:\n"
                "     registry = NodeRegistry()\n"
                "     registry.load_plugins('path/to/plugins.yaml')\n"
                "     pipeline = CuvisPipeline.load_pipeline(..., node_registry=registry)\n\n"
            )
        elif (
            looks_like_plugin
            and instance is not None
            and len(instance.plugin_configs) == 0
        ):
            error_msg += (
                "\n⚠️  This appears to be an external plugin node, but no plugins are loaded!\n"
                "   Load plugins before building pipeline:\n"
                "     registry.load_plugins('path/to/plugins.yaml')\n\n"
            )
        else:
            error_msg += "For custom nodes, provide full import path (e.g., 'my_package.MyNode').\n"

        error_msg += f"Available nodes: {available}"

        raise KeyError(error_msg)

    def __getattribute__(self, name: str):
        """
        Override to make get() work seamlessly on instances.

        When calling registry.get("Node"), this intercepts and wraps the classmethod
        to automatically pass the instance.
        """
        attr = object.__getattribute__(self, name)
        if name == "get" and callable(attr):
            # Return a wrapper that passes this instance to the classmethod
            def instance_get(class_identifier: str) -> type:
                return type(self).get(class_identifier, instance=self)

            return instance_get
        return attr

    @classmethod
    def _import_from_path(cls, import_path: str, clear_cache: bool = False) -> type:
        """Import a class from a full module path."""
        return git_os._import_from_path(import_path, clear_cache=clear_cache)

    @classmethod
    def list_builtin_nodes(cls) -> list[str]:
        """
        List all registered built-in node names.

        Returns:
            Sorted list of node class names
        """
        return sorted(cls._builtin_registry.keys())

    @classmethod
    def get_builtin_class(cls, class_name: str) -> type:
        """
        Get a built-in node class by name.

        Args:
            class_name: Name of the built-in node class

        Returns:
            The node class

        Raises:
            KeyError: If node not found in builtin registry
        """
        if class_name not in cls._builtin_registry:
            raise KeyError(
                f"Builtin node '{class_name}' not found. "
                f"Available: {sorted(cls._builtin_registry.keys())}"
            )
        return cls._builtin_registry[class_name]

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

    def load_plugins(self, manifest_path: Union[str, Path]) -> int:
        """
        Load multiple plugins from a YAML manifest file.

        Args:
            manifest_path: Path to plugins.yaml file

        Returns:
            Number of plugins loaded

        Example:
            registry = NodeRegistry()
            count = registry.load_plugins("plugins.yaml")
            print(f"Loaded {count} plugins")
        """
        if not hasattr(self, "plugin_registry"):
            raise RuntimeError(
                "load_plugins() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        from cuvis_ai_core.utils.plugin_config import PluginManifest

        manifest_path = Path(manifest_path)
        manifest = PluginManifest.from_yaml(manifest_path)
        manifest_dir = manifest_path.resolve().parent

        loaded = 0
        for plugin_name, config in manifest.plugins.items():
            self.load_plugin(
                plugin_name, config.model_dump(), manifest_dir=manifest_dir
            )
            loaded += 1

        logger.info(f"Loaded {loaded} plugins from {manifest_path}")
        return loaded

    def load_plugin(
        self,
        name: str,
        config: dict,
        manifest_dir: Optional[Path] = None,
    ) -> None:
        """
        Load a single plugin into THIS INSTANCE.

        IMPORTANT: This is an INSTANCE METHOD - you must create a NodeRegistry instance first!

        Unlike get() which works as both class and instance method, load_plugin() requires
        an instance for plugin isolation. This is by design from Phase 4's hybrid architecture:
        - Built-in nodes: accessed via class (NodeRegistry.get("MinMaxNormalizer"))
        - Plugin nodes: require instance (registry = NodeRegistry(); registry.load_plugin(...))

        Args:
            name: Plugin identifier (e.g., "adaclip")
            config: Plugin configuration dict with:
                - repo + ref (for Git plugins)
                - path (for local plugins)
                - provides: list of class paths
            manifest_dir: Optional base directory for resolving local plugin paths

        Examples:
            # ✅ CORRECT: Create instance first
            registry = NodeRegistry()
            registry.load_plugin("adaclip", {
                "repo": "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git",
                "tag": "v1.2.3",
                "provides": ["cuvis_ai_adaclip.node.AdaCLIPDetector"]
            })
            # Then use get() to retrieve the node class
            AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")

            # ✅ CORRECT: Local plugin
            registry = NodeRegistry()
            registry.load_plugin("local_dev", {
                "path": "../my-plugin",
                "provides": ["my_plugin.MyNode"]
            })

            # ❌ WRONG: Don't call as class method
            # NodeRegistry.load_plugin(...)  # TypeError: missing 'self'
        """
        # Check instance mode
        if not hasattr(self, "plugin_registry"):
            raise RuntimeError(
                "load_plugin() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        # Early exit if already loaded in this instance
        if name in self.plugin_configs:
            logger.debug(f"Plugin '{name}' already loaded, skipping")
            return

        # Parse and validate config, get plugin path
        plugin_config, plugin_path = git_os.parse_plugin_config(
            name, config, manifest_dir
        )

        # Install plugin dependencies automatically
        self._install_plugin_dependencies(plugin_path, name)

        # Add to sys.path
        self._add_to_sys_path(plugin_path)

        # Extract package prefixes and clear module cache
        package_prefixes = git_os.extract_package_prefixes(plugin_config.provides)
        git_os.clear_package_modules(package_prefixes)

        # Import all provided node classes
        imported_nodes = git_os.import_plugin_nodes(
            plugin_config.provides, clear_cache=True
        )

        # Register all imported nodes in instance registries
        for class_name, node_class in imported_nodes.items():
            self.plugin_registry[class_name] = node_class
            # Find the original class_path for this class_name
            for class_path in plugin_config.provides:
                if class_path.endswith(f".{class_name}") or class_path.endswith(
                    class_name
                ):
                    self.plugin_class_map[class_path] = name
                    break
            logger.debug(f"Registered plugin node '{class_name}' from '{name}'")

        # Track plugin config
        self.plugin_configs[name] = plugin_config

        logger.info(f"Loaded plugin '{name}' with {len(plugin_config.provides)} nodes")

    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin and remove its nodes from THIS INSTANCE.

        Args:
            name: Plugin identifier to unload

        Note:
            Does not remove the plugin from sys.path (Python limitation).
            Cached Git repositories are NOT deleted (use clear_plugin_cache).
        """
        if not hasattr(self, "plugin_registry"):
            raise RuntimeError(
                "unload_plugin() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        if name not in self.plugin_configs:
            logger.warning(f"Plugin '{name}' not loaded, nothing to unload")
            return

        config = self.plugin_configs[name]

        # Remove nodes from registry
        for class_path in config.provides:
            class_name = class_path.rsplit(".", 1)[1]
            self.plugin_registry.pop(class_name, None)
            self.plugin_class_map.pop(class_path, None)

        # Remove config
        del self.plugin_configs[name]
        logger.info(f"Unloaded plugin '{name}'")

    def list_plugins(self) -> list[str]:
        """
        List all loaded plugin names in THIS INSTANCE.

        Returns:
            Sorted list of plugin names
        """
        if not hasattr(self, "plugin_registry"):
            return []
        return sorted(self.plugin_configs.keys())

    def clear_plugins(self) -> None:
        """Unload all plugins and clear plugin registries in THIS INSTANCE."""
        if not hasattr(self, "plugin_registry"):
            raise RuntimeError(
                "clear_plugins() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        self.plugin_registry.clear()
        self.plugin_configs.clear()
        self.plugin_class_map.clear()
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
                git_os.safe_rmtree(cache_entry)
        else:
            if cls._cache_dir.exists():
                logger.info(f"Clearing all plugin caches in {cls._cache_dir}")
                git_os.safe_rmtree(cls._cache_dir)

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
        cache_dir = cls._cache_dir / f"{plugin_name}@{config.tag}"

        if cache_dir.exists():
            if cls._verify_tag_matches(cache_dir, config.tag):
                logger.info(f"Using cached plugin '{plugin_name}' at {cache_dir}")
                return cache_dir
            else:
                logger.warning(f"Cache mismatch for '{plugin_name}', re-cloning...")
                try:
                    git_os.safe_rmtree(cache_dir)
                except PermissionError as exc:
                    raise PermissionError(
                        f"Failed to remove cached plugin '{plugin_name}' at {cache_dir}. "
                        "A file is likely locked or marked read-only. Close any process "
                        "using the cache or delete it manually, then retry."
                    ) from exc

        return cls._clone_repository(config.repo, cache_dir, config.tag)

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
    def _verify_tag_matches(cls, repo_path: Path, expected_tag: str) -> bool:
        """Verify that cached repository is at the expected tag."""
        return git_os._verify_tag_matches(repo_path, expected_tag)

    @classmethod
    def _clone_repository(cls, repo_url: str, dest_path: Path, tag: str) -> Path:
        """Clone Git repository and checkout specific tag."""
        return git_os._clone_repository(repo_url, dest_path, tag)

    @classmethod
    def _add_to_sys_path(cls, path: Path) -> None:
        """Add path to sys.path if not already present."""
        git_os._add_to_sys_path(path)

    @classmethod
    def _install_plugin_dependencies(cls, plugin_path: Path, plugin_name: str) -> None:
        """Detect and install plugin dependencies from pyproject.toml."""
        git_os._install_plugin_dependencies(plugin_path, plugin_name)

    @classmethod
    def _extract_deps_from_pyproject(cls, pyproject_path: Path) -> list[str]:
        """Extract dependencies from pyproject.toml using tomllib (Python 3.11+)."""
        return git_os._extract_deps_from_pyproject(pyproject_path)

    @classmethod
    def _install_dependencies_with_uv(cls, deps: list[str], plugin_name: str) -> None:
        """Install dependencies using 'uv pip install'."""
        git_os._install_dependencies_with_uv(deps, plugin_name)

    @classmethod
    def clear(cls):
        """Clear all registered nodes (primarily for testing)."""
        cls._builtin_registry.clear()
