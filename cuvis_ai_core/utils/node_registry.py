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
        elif looks_like_plugin and len(instance.plugin_configs) == 0:
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
        """
        Import a class from a full module path.

        Args:
            import_path: Full import path (e.g., "my_package.nodes.CustomNode")
            clear_cache: If True, clear module cache before importing (for plugin reloading)

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

            # Clear module cache if requested (for plugin reloading)
            if clear_cache:
                parts_to_clear = module_path.split(".")
                for i in range(len(parts_to_clear), 0, -1):
                    partial_path = ".".join(parts_to_clear[:i])
                    if partial_path in sys.modules:
                        del sys.modules[partial_path]
                        logger.debug(f"Cleared cached module: {partial_path}")

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
                "ref": "v1.2.3",
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

        from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig

        # Check if already loaded in this instance
        if name in self.plugin_configs:
            logger.debug(f"Plugin '{name}' already loaded, skipping")
            return

        # Validate and parse config
        if "repo" in config:
            plugin_config = GitPluginConfig.model_validate(config)
            plugin_path = self._ensure_git_plugin(name, plugin_config)
        elif "path" in config:
            if manifest_dir is not None:
                config = dict(config)
                config["path"] = str(
                    LocalPluginConfig(**config).resolve_path(manifest_dir)
                )
            plugin_config = LocalPluginConfig.model_validate(config)
            plugin_path = self._ensure_local_plugin(name, plugin_config)
        else:
            raise ValueError(
                f"Plugin '{name}' must have either 'repo' (Git) or 'path' (local)"
            )

        # Install plugin dependencies automatically
        self._install_plugin_dependencies(plugin_path, name)

        # Add to sys.path
        self._add_to_sys_path(plugin_path)

        # Import and register all provided classes
        for class_path in plugin_config.provides:
            # Clear cache for plugins to ensure fresh import
            node_class = self._import_from_path(class_path, clear_cache=True)
            class_name = node_class.__name__

            # Register in instance plugin registry
            self.plugin_registry[class_name] = node_class
            self.plugin_class_map[class_path] = name
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
    def _install_plugin_dependencies(cls, plugin_path: Path, plugin_name: str) -> None:
        """
        Detect and install plugin dependencies from pyproject.toml.

        This method enforces PEP 621 compliance by requiring plugins to have
        a pyproject.toml file with proper dependency specifications.

        Args:
            plugin_path: Path to the plugin directory
            plugin_name: Name of the plugin (for logging)

        Raises:
            FileNotFoundError: If pyproject.toml is not found
            RuntimeError: If dependency installation fails
        """
        pyproject_file = plugin_path / "pyproject.toml"

        # Require pyproject.toml (PEP 621)
        if not pyproject_file.exists():
            raise FileNotFoundError(
                f"Plugin '{plugin_name}' must have a pyproject.toml file.\n"
                f"PEP 621 (https://peps.python.org/pep-0621/) specifies pyproject.toml "
                f"as the standard for Python project metadata and dependencies.\n"
                f"Expected location: {pyproject_file}"
            )

        # Extract dependencies
        deps = cls._extract_deps_from_pyproject(pyproject_file)

        if not deps:
            logger.debug(f"No dependencies found for plugin '{plugin_name}'")
            return

        # Install with uv pip install (let uv handle conflicts)
        logger.info(
            f"Installing {len(deps)} dependencies for plugin '{plugin_name}'..."
        )
        cls._install_dependencies_with_uv(deps, plugin_name)

    @classmethod
    def _extract_deps_from_pyproject(cls, pyproject_path: Path) -> list[str]:
        """
        Extract dependencies from pyproject.toml using tomllib (Python 3.11+).

        Args:
            pyproject_path: Path to pyproject.toml file

        Returns:
            List of dependency specifiers
        """
        import tomllib  # stdlib in Python 3.11+

        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        deps = data.get("project", {}).get("dependencies", [])

        # Filter out comments and empty strings
        filtered_deps = [
            dep.strip()
            for dep in deps
            if dep and dep.strip() and not dep.strip().startswith("#")
        ]

        logger.debug(
            f"Extracted {len(filtered_deps)} dependencies from {pyproject_path}"
        )
        return filtered_deps

    @classmethod
    def _install_dependencies_with_uv(cls, deps: list[str], plugin_name: str) -> None:
        """
        Install dependencies using 'uv pip install'.

        This uses runtime-only installation that doesn't modify the project's
        pyproject.toml. Dependency conflicts are delegated to uv for resolution.

        Args:
            deps: List of dependency specifiers
            plugin_name: Name of the plugin (for logging)

        Raises:
            RuntimeError: If installation fails or times out
        """
        import subprocess

        logger.info(f"Dependencies to install: {', '.join(deps)}")

        # Use uv pip install (runtime only, doesn't modify pyproject.toml)
        cmd = ["uv", "pip", "install"] + deps

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
            )
            logger.info(f"✓ Plugin '{plugin_name}' dependencies installed successfully")

            if result.stdout:
                logger.debug(f"uv output: {result.stdout}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Plugin '{plugin_name}' dependency installation timed out (>5 min). "
                f"This may indicate a network issue or very large dependencies."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to install dependencies for plugin '{plugin_name}'.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {e.stderr}\n\n"
                f"This may indicate version conflicts or missing packages. "
                f"uv could not resolve the dependency tree."
            ) from e

    @classmethod
    def clear(cls):
        """Clear all registered nodes (primarily for testing)."""
        cls._builtin_registry.clear()
