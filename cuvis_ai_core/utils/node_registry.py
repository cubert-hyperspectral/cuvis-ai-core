"""Node registry for managing built-in and custom node types."""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from loguru import logger

import cuvis_ai_core.utils.git_and_os as git_os
from cuvis_ai_schemas.plugin import (
    PluginManifest,
    load_plugin_manifest,
)


class NodeRegistry:
    """
    Hybrid registry supporting both class and instance usage.

    CLASS MODE (built-ins only):
        NodeRegistry.get("MinMaxNormalizer")  # No instantiation needed

    INSTANCE MODE (built-ins + plugins):
        registry = NodeRegistry()
        registry.register_plugin("configs/plugins/adaclip.yaml")
        registry.get("AdaCLIPDetector")  # Access both built-ins and plugins

    Built-in nodes are registered via @register decorator for O(1) lookup.
    Plugin nodes are loaded into instances for session isolation.
    """

    # ========== CLASS-LEVEL: Built-in nodes (singleton) ==========
    _builtin_registry: Dict[str, type] = {}

    def __init__(self):
        """Create instance for plugin support."""
        # Every known plugin's config (registered or loaded) — the single
        # source of plugin config. Anything that needs a config reads it here;
        # `register_plugins_catalog` records entries; `register_plugins_installed`
        # records and imports them.
        self.plugin_catalog: Dict[str, PluginManifest] = {}
        # Loaded node classes, keyed by class name. Membership here *is* the
        # loaded state: a plugin is loaded iff its provided classes are present.
        self.loaded_plugin_nodes: Dict[str, type] = {}
        # The same classes keyed by full path: the manifest's class path and the
        # class's own module path. A full path in a pipeline yaml resolves here
        # first, so two classes that share a simple name stay apart.
        self.loaded_plugin_paths: Dict[str, type] = {}
        # The plugins that registered a class under each simple name, in load
        # order: (plugin name, class, manifest class path). The last one answers
        # lookups by simple name; when it unloads, the one loaded before it takes
        # the name back, and a class that another loaded manifest still lists
        # keeps its full-path entries.
        self.simple_name_providers: Dict[str, list[tuple[str, type, str | None]]] = {}
        # Loaded DataModule classes, keyed by DATA_MODULE_NAME (globally unique).
        # A `kind: data_module` capability entry registers here instead of into
        # loaded_plugin_nodes, and never appears in the node palette.
        self.data_modules: Dict[str, type] = {}

    @staticmethod
    def _entry_kind(node) -> str:
        """The static kind of a capability entry; defaults to ``node``."""
        return getattr(node, "kind", "node") or "node"

    def _provided_class_names(self, cfg: PluginManifest) -> list[str]:
        """Simple (unqualified) class names of the ``kind=='node'`` entries."""
        return [
            node.class_name.rsplit(".", 1)[-1]
            for node in cfg.capabilities
            if self._entry_kind(node) == "node"
        ]

    def _provided_data_module_names(self, cfg: PluginManifest) -> list[str]:
        """DATA_MODULE_NAMEs of the ``kind=='data_module'`` entries."""
        return [
            node.data_module_name
            for node in cfg.capabilities
            if self._entry_kind(node) == "data_module"
        ]

    def _is_loaded(self, name: str) -> bool:
        """Whether ALL of a catalog plugin's node classes are currently loaded.

        Used for listing and the "already fully loaded" early-exit. A
        partially-loaded plugin reads as *not* loaded here.
        """
        cfg = self.plugin_catalog.get(name)
        if cfg is None:
            return False
        node_names = self._provided_class_names(cfg)
        dm_names = self._provided_data_module_names(cfg)
        if not node_names and not dm_names:
            return False
        return all(self._provides(name, n) for n in node_names) and all(
            d in self.data_modules for d in dm_names
        )

    def _has_loaded_node(self, name: str) -> bool:
        """Whether ANY of a catalog plugin's node classes are currently loaded.

        Used for mutation guards (catalog replacement, unload): a plugin with
        even one live class must still block a config swap and remain unloadable.
        """
        cfg = self.plugin_catalog.get(name)
        if cfg is None:
            return False
        return any(
            self._provides(name, n) for n in self._provided_class_names(cfg)
        ) or any(d in self.data_modules for d in self._provided_data_module_names(cfg))

    def _provides(self, plugin: str, class_name: str) -> bool:
        """Whether ``plugin`` has a class loaded under the simple name ``class_name``.

        Answered from ``simple_name_providers`` when it knows the name, so a plugin
        whose namesake another loaded plugin holds is not taken for loaded; a name
        set in ``loaded_plugin_nodes`` by other means counts by its presence.
        """
        providers = self.simple_name_providers.get(class_name)
        if providers:
            return any(p[0] == plugin for p in providers)
        return class_name in self.loaded_plugin_nodes

    @staticmethod
    def _class_path(node_class: type) -> str:
        """The import path of a class: its module and qualified name."""
        return f"{node_class.__module__}.{node_class.__qualname__}"

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
        0. A full path resolves to the plugin or built-in class at exactly that
           path, so a plugin class never shadows a built-in of the same name
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
            registry.register_plugin("configs/plugins/adaclip.yaml")
            cls = registry.get("AdaCLIPDetector")

            # Custom node with full path
            cls = NodeRegistry.get("my_company.detectors.AdvancedRXDetector")
        """
        # 0. An exact full path wins over any simple-name match below.
        if "." in class_identifier:
            if instance is not None:
                exact = instance.loaded_plugin_paths.get(class_identifier)
                if exact is not None:
                    return exact
            builtin = cls._builtin_registry.get(class_identifier.rsplit(".", 1)[1])
            if builtin is not None and cls._class_path(builtin) == class_identifier:
                return builtin

        # 1. Check instance plugins first (if instance provided)
        if instance is not None:
            if class_identifier in instance.loaded_plugin_nodes:
                return instance.loaded_plugin_nodes[class_identifier]
            # For full paths, also check if last component is in plugins
            if "." in class_identifier:
                class_name = class_identifier.rsplit(".", 1)[1]
                if class_name in instance.loaded_plugin_nodes:
                    return instance.loaded_plugin_nodes[class_name]

        # 2. Check built-in registry (both modes)
        if class_identifier in cls._builtin_registry:
            return cls._builtin_registry[class_identifier]

        # 3. Try importlib for custom nodes (must have dot in path)
        if "." in class_identifier:
            return cls._import_from_path(class_identifier)

        # Not found - provide helpful error
        available = cls.list_builtin_nodes()
        if instance is not None:
            available = available + sorted(instance.loaded_plugin_nodes.keys())

        # Check if it looks like a plugin node (has multiple dots or known plugin pattern)
        looks_like_plugin = class_identifier.count(".") >= 2 or any(
            pkg in class_identifier.lower() for pkg in ["plugin", "cuvis_ai_"]
        )

        error_msg = f"Node '{class_identifier}' not found in registry.\n"

        if looks_like_plugin and instance is None:
            error_msg += (
                "\n⚠️  This appears to be an external plugin node!\n"
                "   Did you forget to load plugins?\n\n"
                "   For CLI usage (the pipeline yaml must declare its plugins: field):\n"
                "     uv run restore-pipeline --pipeline-path <path> --plugins-dir configs/plugins\n\n"
                "   For Python usage:\n"
                "     registry = NodeRegistry()\n"
                "     registry.register_plugin('path/to/plugin.yaml')\n"
                "     pipeline = CuvisPipeline.load_pipeline(..., node_registry=registry)\n\n"
            )
        elif (
            looks_like_plugin
            and instance is not None
            and not instance.loaded_plugin_nodes
        ):
            error_msg += (
                "\n⚠️  This appears to be an external plugin node, but no plugins are loaded!\n"
                "   Load plugins before building pipeline:\n"
                "     registry.register_plugin('path/to/plugin.yaml')\n\n"
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
    def _import_from_path(cls, import_path: str) -> type:
        """Import a class from a full module path."""
        return git_os._import_from_path(import_path)

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
    def auto_register_package(cls, package_name: str) -> int:
        """
        Auto-register all Node classes from a package.

        Searches the package for classes that inherit from Node
        and registers them automatically.

        Args:
            package_name: Full package name (e.g., "cuvis_ai.node")

        Returns:
            Number of classes registered

        Example:
            NodeRegistry.auto_register_package("cuvis_ai.node")
        """
        from cuvis_ai_core.node.node import Node

        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            raise ImportError(f"Failed to import package '{package_name}': {e}") from e

        # Get the package directory
        if not hasattr(package, "__path__"):
            raise ValueError(f"'{package_name}' is not a package (has no __path__)")

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
                logger.warning(f"Failed to auto-register from {module_name}: {e}")

        return registered_count

    def register_plugin(self, manifest_path: Union[str, Path]) -> None:
        """Register the plugin declared in a bare manifest YAML into THIS INSTANCE.

        Import-only front door for in-process use (CLI, notebooks).
        One yaml file is one plugin: it loads the bare manifest (resolving a
        local plugin's relative path), keys it by ``manifest.name``, and hands
        it to :meth:`register_plugins_installed`. The plugin must already be
        importable in the active environment (an editable ``[tool.uv.sources]``
        entry in dev, or ``uv pip install`` / the ``provision`` helper
        otherwise); this never clones or installs.

        Args:
            manifest_path: Path to a single-plugin manifest YAML.

        Example:
            registry = NodeRegistry()
            registry.register_plugin("configs/plugins/sam3.yaml")
        """
        if not hasattr(self, "loaded_plugin_nodes"):
            raise RuntimeError(
                "register_plugin() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        manifest_path = Path(manifest_path)
        manifest = load_plugin_manifest(manifest_path)

        self.register_plugins_installed({manifest.name: manifest})
        logger.info(f"Registered plugin '{manifest.name}' from {manifest_path}")

    def register_plugins_catalog(
        self,
        configs: Dict[str, PluginManifest],
    ) -> None:
        """
        Register plugin metadata into the session's catalog WITHOUT installing.

        A catalog entry is "this plugin is known and the session can
        materialise it on demand". Calling this does NOT clone, install
        dependencies, mutate sys.path, or import any modules — those side
        effects are deferred to :meth:`register_plugins_installed`, which the
        child runtime calls once the resolved packages are installed.

        Idempotent re-registration with a different config logs an override
        and replaces the catalog entry. A loaded plugin's entry is *not*
        replaced: the live class objects came from the old config, so swapping
        it would make the catalog describe a source the loaded classes never
        came from (and would desync ``unload_plugin``, which pops
        ``loaded_plugin_nodes`` by the catalog entry's ``capabilities``). Such an
        override is logged and ignored; the caller must ``unload_plugin`` first.

        Args:
            configs: dict mapping plugin name → its parsed ``PluginManifest``
                (source paths already resolved).

        Example:
            from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins
            resolved = resolve_pipeline_plugins(pipeline_cfg, [Path("configs/plugins")])
            registry = NodeRegistry()
            registry.register_plugins_catalog(resolved)
            # registry.plugin_catalog["adaclip"] now holds the metadata.
            # Nothing is imported until register_plugins_installed materialises it.
        """
        if not hasattr(self, "loaded_plugin_nodes"):
            raise RuntimeError(
                "register_plugins_catalog() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        for name, cfg in configs.items():
            if name in self.plugin_catalog:
                existing = self.plugin_catalog[name]
                if existing.model_dump() == cfg.model_dump():
                    continue
                if self._has_loaded_node(name):
                    logger.warning(
                        f"Plugin '{name}' is loaded; ignoring catalog override "
                        f"with a different config. Call unload_plugin('{name}') "
                        "first to change it."
                    )
                    continue
                logger.info(
                    f"Plugin '{name}' catalog entry overridden "
                    f"(was: {existing.model_dump()}; now: {cfg.model_dump()})"
                )
            self.plugin_catalog[name] = cfg

    def _register_node_classes(self, name: str, config: PluginManifest) -> None:
        """Import a plugin's provided classes into ``loaded_plugin_nodes``.

        The shared registration core: the in-process front door
        (:meth:`register_plugin`) and the dict entry point
        (:meth:`register_plugins_installed`) funnel through here. It
        imports the FQCNs via ``importlib`` only; it never clones, installs, or
        mutates ``sys.path``. The plugin package must already be importable, so a
        missing one is re-raised with a hint pointing at the ``provision`` helper.
        """
        class_paths = [node.class_name for node in config.capabilities]
        try:
            imported_nodes = git_os.import_plugin_nodes(class_paths, clear_cache=False)
        except ImportError as exc:
            raise ModuleNotFoundError(
                f"Plugin '{name}' could not be imported: {exc}\n"
                f"Its package is not installed in this environment. Provision it "
                f"first, for example:\n"
                f"  uv run provision --pipeline-path <pipeline.yaml> "
                f"--plugins-dir <dir> --apply\n"
                f"or install the plugin directly, e.g. "
                f"uv pip install '<pkg>[extras] @ git+<url>@<tag>'."
            ) from exc
        by_simple_name = {
            node.class_name.rsplit(".", 1)[-1]: node for node in config.capabilities
        }
        for class_name, node_class in imported_nodes.items():
            entry = by_simple_name.get(class_name)
            if entry is not None and self._entry_kind(entry) == "data_module":
                self._register_data_module(name, class_name, node_class, entry)
            else:
                existing = self.loaded_plugin_nodes.get(class_name)
                if existing is not None and existing is not node_class:
                    logger.warning(
                        f"Plugin '{name}' registers {self._class_path(node_class)}, "
                        f"which replaces {self._class_path(existing)} for lookups by "
                        f"the simple name '{class_name}'; full class paths still "
                        f"resolve to each class."
                    )
                self.loaded_plugin_nodes[class_name] = node_class
                self.loaded_plugin_paths[self._class_path(node_class)] = node_class
                if entry is not None:
                    self.loaded_plugin_paths[entry.class_name] = node_class
                providers = [
                    p
                    for p in self.simple_name_providers.get(class_name, [])
                    if p[0] != name
                ]  # a plugin registered again moves to the end
                providers.append(
                    (name, node_class, entry.class_name if entry is not None else None)
                )
                self.simple_name_providers[class_name] = providers
                logger.debug(f"Registered plugin node '{class_name}' from '{name}'")

    def _register_data_module(self, name, class_name, cls, entry) -> None:
        """File a ``kind: data_module`` entry under ``data_modules[DATA_MODULE_NAME]``."""
        from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule

        if not (isinstance(cls, type) and issubclass(cls, BaseCuvisAIDataModule)):
            raise TypeError(
                f"{class_name}: manifest declares kind='data_module' but the class "
                f"is not a BaseCuvisAIDataModule subclass."
            )
        dm_name = entry.data_module_name
        if cls.DATA_MODULE_NAME != dm_name:
            raise ValueError(
                f"{class_name}: manifest data_module_name={dm_name!r} != "
                f"class DATA_MODULE_NAME={cls.DATA_MODULE_NAME!r}."
            )
        existing = self.data_modules.get(dm_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"data module {dm_name!r} is already registered by "
                f"{existing.__module__}.{existing.__name__}; names must be unique."
            )
        self.data_modules[dm_name] = cls
        logger.debug(f"Registered data module '{dm_name}' ({class_name}) from '{name}'")

    def register_plugins_installed(
        self,
        resolved_plugins: Mapping[str, PluginManifest],
    ) -> None:
        """Register + import classes from already-installed plugin packages.

        Records each plugin's config in the catalog and imports its classes via
        a plain ``importlib.import_module`` (no clone / install / ``sys.path``
        step). Used by the in-process file-path front door
        (:meth:`register_plugin`) and the orchestrator child runtime, where every
        plugin in the child's venv is already a real installed package (uv put it
        there during ``compose_env``). The sibling :meth:`register_plugins_catalog`
        records metadata only, without importing.

        Atomic across the set: if any plugin fails to import, every change made to
        ``plugin_catalog`` / ``loaded_plugin_nodes`` / ``data_modules`` during this
        call is rolled back before the error propagates, so a partially-registered
        set never leaks into the registry.
        """
        catalog_snapshot = dict(self.plugin_catalog)
        nodes_snapshot = dict(self.loaded_plugin_nodes)
        paths_snapshot = dict(self.loaded_plugin_paths)
        providers_snapshot = {k: list(v) for k, v in self.simple_name_providers.items()}
        modules_snapshot = dict(self.data_modules)
        try:
            for name, config in resolved_plugins.items():
                self.plugin_catalog[name] = config
                self._register_node_classes(name, config)
                logger.info(
                    f"Loaded preinstalled plugin '{name}' with "
                    f"{len(config.capabilities)} nodes"
                )
        except Exception:
            self.plugin_catalog.clear()
            self.plugin_catalog.update(catalog_snapshot)
            self.loaded_plugin_nodes.clear()
            self.loaded_plugin_nodes.update(nodes_snapshot)
            self.loaded_plugin_paths.clear()
            self.loaded_plugin_paths.update(paths_snapshot)
            self.simple_name_providers.clear()
            self.simple_name_providers.update(providers_snapshot)
            self.data_modules.clear()
            self.data_modules.update(modules_snapshot)
            raise

    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin and remove its nodes from THIS INSTANCE.

        CLI / dev-mode counterpart to :meth:`register_plugin`; the
        orchestrated server path never calls this. The child runtime
        is torn down on ``CloseSession`` and its venv is reused by
        the composer's cache, so no explicit unload step is needed
        on the production path.

        Args:
            name: Plugin identifier to unload

        Note:
            Does not remove the plugin from sys.path (Python limitation).
        """
        if not hasattr(self, "loaded_plugin_nodes"):
            raise RuntimeError(
                "unload_plugin() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        if not self._has_loaded_node(name):
            logger.warning(f"Plugin '{name}' not loaded, nothing to unload")
            return

        # Drop the plugin's classes from loaded_plugin_nodes and its full paths.
        # The catalog entry stays: the plugin is still *known*, just no longer
        # loaded. pop(..., None) is defensive so a partially-loaded plugin still
        # cleans up.
        cfg = self.plugin_catalog[name]
        own: set[type] = set()
        own_paths: set[str] = set()
        for node in cfg.capabilities:
            if self._entry_kind(node) != "node":
                continue
            own_paths.add(node.class_name)
            node_class = self.loaded_plugin_paths.get(node.class_name)
            if node_class is not None:
                own.add(node_class)
                own_paths.add(self._class_path(node_class))
        for class_name in self._provided_class_names(cfg):
            providers = self.simple_name_providers.get(class_name, [])
            holder = self.loaded_plugin_nodes.get(class_name)
            if providers:
                held_by_this_plugin = providers[-1][0] == name
            else:  # registered outside register_plugins_installed
                held_by_this_plugin = holder is not None and (
                    holder in own or self._class_path(holder) in own_paths
                )
            remaining = [p for p in providers if p[0] != name]
            if remaining:
                self.simple_name_providers[class_name] = remaining
            else:
                self.simple_name_providers.pop(class_name, None)
            if not held_by_this_plugin:
                continue  # a namesake of another plugin holds the name and stays
            if remaining:
                fallback = remaining[-1]
                self.loaded_plugin_nodes[class_name] = fallback[1]
                if fallback[1] is not holder:
                    logger.info(
                        f"Simple name '{class_name}' falls back to "
                        f"{self._class_path(fallback[1])} of plugin '{fallback[0]}'"
                    )
            else:
                self.loaded_plugin_nodes.pop(class_name, None)
        # full paths: drop this plugin's, keep those another loaded plugin lists
        still_listed: dict[str, type] = {}
        for providers in self.simple_name_providers.values():
            for _plugin, node_class, manifest_path in providers:
                still_listed[self._class_path(node_class)] = node_class
                if manifest_path is not None:
                    still_listed[manifest_path] = node_class
        for path in own_paths:
            if path in still_listed:
                self.loaded_plugin_paths[path] = still_listed[path]
            else:
                self.loaded_plugin_paths.pop(path, None)
        for dm_name in self._provided_data_module_names(cfg):
            self.data_modules.pop(dm_name, None)

        logger.info(f"Unloaded plugin '{name}'")

    def list_plugins(self) -> list[str]:
        """
        List all loaded plugin names in THIS INSTANCE.

        Returns:
            Sorted list of plugin names
        """
        if not hasattr(self, "loaded_plugin_nodes"):
            return []
        return sorted(name for name in self.plugin_catalog if self._is_loaded(name))

    def clear_plugins(self) -> None:
        """Unload all plugins and clear plugin registries in THIS INSTANCE.

        CLI / dev-mode only — see :meth:`register_plugin` /
        :meth:`unload_plugin` for the rationale. The orchestrated
        server path drops the entire child runtime on
        ``CloseSession`` instead.
        """
        if not hasattr(self, "loaded_plugin_nodes"):
            raise RuntimeError(
                "clear_plugins() requires an instance. "
                "Create instance first: registry = NodeRegistry()"
            )

        self.loaded_plugin_nodes.clear()
        self.loaded_plugin_paths.clear()
        self.simple_name_providers.clear()
        self.data_modules.clear()
        self.plugin_catalog.clear()
        logger.info("Cleared all plugins")

    @classmethod
    def clear(cls):
        """Clear all registered nodes (primarily for testing)."""
        cls._builtin_registry.clear()
