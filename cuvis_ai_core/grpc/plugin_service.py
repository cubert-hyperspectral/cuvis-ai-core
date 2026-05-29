"""Plugin management service component."""

from __future__ import annotations

import inspect
from pathlib import Path

import grpc
import numpy as np
from loguru import logger

from cuvis_ai_core.grpc.error_handling import get_session_or_error, grpc_handler
from cuvis_ai_core.grpc.helpers import dtype_to_proto
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.orchestrator.catalog import (
    CatalogNodeEntry,
    CatalogPortSpec,
    load_catalog_entry,
)
from cuvis_ai_core.utils.icon_helpers import get_node_icon
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import PluginManifest
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.grpc.conversions import (
    node_category_to_proto,
    node_tag_to_proto,
)
from cuvis_ai_schemas.pipeline import PortSpec


def _convert_port_spec_to_proto(spec: PortSpec, name: str) -> cuvis_ai_pb2.PortSpec:
    """Convert Python PortSpec to proto PortSpec message.

    Args:
        spec: Python PortSpec from node class
        name: Port name from INPUT_SPECS/OUTPUT_SPECS dict key

    Returns:
        Proto PortSpec message

    Raises:
        ValueError: If dtype cannot be mapped or shape contains non-int values
    """
    # Map Python dtype to proto DType enum via shared helper so this site
    # cannot drift from the tensor serializers in grpc.helpers.
    proto_dtype = dtype_to_proto(spec.dtype)

    # Convert shape tuple to list of int64
    # Symbolic dimensions (strings like "output_channels") are converted to -1 (dynamic)
    shape_list = []
    for dim in spec.shape:
        if isinstance(dim, int):
            shape_list.append(dim)
        elif isinstance(dim, str):
            # Symbolic dimension referring to a hyperparameter (e.g., "output_channels")
            # Convert to -1 to indicate dynamic/runtime-determined dimension
            shape_list.append(-1)
        else:
            # Invalid dimension type
            raise ValueError(
                f"Port '{name}' has invalid shape dimension type '{type(dim)}'. "
                f"Expected int or str, got: {dim}"
            )

    return cuvis_ai_pb2.PortSpec(
        name=name,
        dtype=proto_dtype,
        shape=shape_list,
        optional=spec.optional,
        description=spec.description,
    )


def _resolve_package_root(node_class: type) -> Path | None:
    """Return the package root that owns ``node_class`` if it carries an
    ``assets/node_icons/`` folder, otherwise ``None``.

    The walk stops after a small bound (8 levels) to avoid pathological
    loops on broken symlinks. Returns ``None`` when ``inspect.getfile``
    can't resolve a real file (e.g. for plugins loaded from frozen modules
    or in-memory exec) — the icon helper falls through to the schemas
    default in that case.
    """
    try:
        source_path = Path(inspect.getfile(node_class)).resolve()
    except (TypeError, OSError):
        return None

    for ancestor in (source_path, *source_path.parents)[:9]:
        candidate = ancestor / "assets" / "node_icons"
        if candidate.is_dir():
            return ancestor
    return None


def _extract_node_metadata(
    node_class: type | None, *, class_name: str
) -> tuple[int, list[int], bytes]:
    """Resolve (category proto-int, sorted tag proto-ints, icon SVG bytes).

    Independently safe: any exception inside ``get_category`` /
    ``get_tags`` / ``get_icon_name`` / icon resolution is logged and
    falls back to ``(NODE_CATEGORY_UNSPECIFIED, [], unspecified.svg)``,
    so a misbehaving plugin never breaks ``list_available_nodes``. ``None``
    for ``node_class`` (lookup failure upstream) takes the same path.
    """
    if node_class is None:
        category = NodeCategory.UNSPECIFIED
        tag_ints: list[int] = []
        icon_svg = get_node_icon(
            class_name=class_name,
            icon_name=None,
            category=category,
            package_root=None,
        )
        return node_category_to_proto(category), tag_ints, icon_svg

    try:
        category = node_class.get_category()
    except Exception as e:
        logger.warning(
            f"Failed to read category for node '{class_name}': {e}",
            exc_info=True,
        )
        category = NodeCategory.UNSPECIFIED

    try:
        tags = node_class.get_tags()
        tag_ints = sorted(node_tag_to_proto(tag) for tag in tags)
    except Exception as e:
        logger.warning(
            f"Failed to read tags for node '{class_name}': {e}",
            exc_info=True,
        )
        tag_ints = []

    try:
        icon_name = node_class.get_icon_name()
    except Exception:
        icon_name = None

    try:
        icon_svg = get_node_icon(
            class_name=class_name,
            icon_name=icon_name,
            category=category,
            package_root=_resolve_package_root(node_class),
        )
    except Exception as e:
        logger.warning(
            f"Failed to resolve icon for node '{class_name}': {e}",
            exc_info=True,
        )
        icon_svg = b""

    return node_category_to_proto(category), tag_ints, icon_svg


def _catalog_port_spec_to_proto(
    port_name: str, spec: CatalogPortSpec
) -> cuvis_ai_pb2.PortSpec:
    """Convert a catalog port spec (string dtype) to its proto form.

    Empty / unknown dtype strings map to ``D_TYPE_UNSPECIFIED`` — the
    same wire value the runtime uses for generic-tensor markers on
    node classes (``torch.Tensor`` as a class).
    """
    if not spec.dtype:
        proto_dtype = cuvis_ai_pb2.D_TYPE_UNSPECIFIED
    else:
        try:
            np_dtype = np.dtype(spec.dtype)
            proto_dtype = dtype_to_proto(np_dtype)
        except (TypeError, ValueError):
            logger.warning(
                f"Catalog port '{port_name}' has unsupported dtype "
                f"{spec.dtype!r}; emitting D_TYPE_UNSPECIFIED"
            )
            proto_dtype = cuvis_ai_pb2.D_TYPE_UNSPECIFIED
    return cuvis_ai_pb2.PortSpec(
        name=port_name,
        dtype=proto_dtype,
        shape=list(spec.shape),
        optional=spec.optional,
        description=spec.description,
    )


def _catalog_specs_map_to_proto(
    specs: dict[str, tuple[CatalogPortSpec, ...]],
) -> dict[str, cuvis_ai_pb2.PortSpecList]:
    """Convert a {port → tuple-of-CatalogPortSpec} map into proto PortSpecList map."""
    out: dict[str, cuvis_ai_pb2.PortSpecList] = {}
    for port_name, specs_tuple in specs.items():
        proto_specs = [_catalog_port_spec_to_proto(port_name, s) for s in specs_tuple]
        out[port_name] = cuvis_ai_pb2.PortSpecList(specs=proto_specs)
    return out


def _catalog_entry_to_node_info(
    entry: CatalogNodeEntry, plugin_name: str
) -> cuvis_ai_pb2.NodeInfo:
    """Build the proto NodeInfo for one catalog node — pure data, no class import."""
    try:
        category = NodeCategory(entry.category)
    except ValueError:
        category = NodeCategory.UNSPECIFIED
    proto_category = node_category_to_proto(category)

    tag_ints: list[int] = []
    for tag_str in entry.tags:
        try:
            tag = NodeTag(tag_str)
        except ValueError:
            continue
        tag_ints.append(node_tag_to_proto(tag))
    tag_ints.sort()

    icon_svg_bytes = entry.icon_svg.encode("utf-8") if entry.icon_svg else b""
    return cuvis_ai_pb2.NodeInfo(
        class_name=entry.class_name,
        full_path=entry.full_path,
        source="plugin",
        plugin_name=plugin_name,
        input_specs=_catalog_specs_map_to_proto(entry.input_specs),
        output_specs=_catalog_specs_map_to_proto(entry.output_specs),
        icon_svg=icon_svg_bytes,
        category=proto_category,
        tags=tag_ints,
    )


class PluginService:
    """gRPC service layer for plugin management operations."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    @grpc_handler("Failed to load plugins")
    def load_plugins(
        self,
        request: cuvis_ai_pb2.LoadPluginsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPluginsResponse:
        """Register manifest entries as catalog metadata in the session.

        This RPC does not install or import plugins. It parses the manifest,
        validates each entry, and registers them via
        ``session.node_registry.register_catalog_entries(...)``. Actual
        materialisation (clone, install, import) happens lazily when
        ``LoadPipeline`` references the registered plugin through the
        pipeline yaml's ``plugins:`` field.

        ``failed_plugins`` reports per-entry Pydantic validation failures;
        install failures surface later in the ``LoadPipeline`` path.
        """
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.LoadPluginsResponse()

        if not request.manifest or not request.manifest.config_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("manifest.config_bytes is required")
            return cuvis_ai_pb2.LoadPluginsResponse()

        # Parse JSON → Pydantic (following existing pattern)
        manifest_json = request.manifest.config_bytes.decode("utf-8")
        manifest = PluginManifest.model_validate_json(manifest_json)

        registered: list[str] = []
        failed: dict[str, str] = {}

        # Register each entry into the session's catalog. No install, no import.
        for plugin_name, config in manifest.plugins.items():
            try:
                session.node_registry.register_catalog_entries({plugin_name: config})
                session.registered_plugins[plugin_name] = config.model_dump()
                registered.append(plugin_name)
                logger.info(
                    f"Registered plugin '{plugin_name}' in session "
                    f"{request.session_id} catalog (not yet installed)"
                )
            except Exception as e:
                failed[plugin_name] = str(e)
                logger.error(
                    f"Failed to register plugin '{plugin_name}' in catalog: {e}"
                )

        return cuvis_ai_pb2.LoadPluginsResponse(
            registered_plugins=registered, failed_plugins=failed
        )

    @grpc_handler("Failed to list loaded plugins")
    def list_loaded_plugins(
        self,
        request: cuvis_ai_pb2.ListLoadedPluginsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ListLoadedPluginsResponse:
        """List plugins loaded in session."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.ListLoadedPluginsResponse()

        plugins = []
        for name, config in session.registered_plugins.items():
            plugin_type = "git" if "repo" in config else "local"
            source = config.get("repo") or config.get("path", "")
            tag = config.get("tag", "")
            provides = config.get("provides", [])

            plugins.append(
                cuvis_ai_pb2.PluginInfo(
                    name=name,
                    type=plugin_type,
                    source=source,
                    tag=tag,
                    provides=provides,
                )
            )

        return cuvis_ai_pb2.ListLoadedPluginsResponse(plugins=plugins)

    @grpc_handler("Failed to get plugin info")
    def get_plugin_info(
        self,
        request: cuvis_ai_pb2.GetPluginInfoRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPluginInfoResponse:
        """Get information about specific plugin."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.GetPluginInfoResponse()

        if request.plugin_name not in session.registered_plugins:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Plugin '{request.plugin_name}' not loaded in session")
            return cuvis_ai_pb2.GetPluginInfoResponse()

        config = session.registered_plugins[request.plugin_name]
        plugin_type = "git" if "repo" in config else "local"
        source = config.get("repo") or config.get("path", "")
        tag = config.get("tag", "")
        provides = config.get("provides", [])

        return cuvis_ai_pb2.GetPluginInfoResponse(
            plugin=cuvis_ai_pb2.PluginInfo(
                name=request.plugin_name,
                type=plugin_type,
                source=source,
                tag=tag,
                provides=provides,
            )
        )

    @grpc_handler("Failed to list available nodes")
    def list_available_nodes(
        self,
        request: cuvis_ai_pb2.ListAvailableNodesRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ListAvailableNodesResponse:
        """List all available nodes (built-in + session plugins).

        Plugin nodes come exclusively from each plugin's static
        ``metadata.json`` (pointed at by ``metadata_path`` in the
        manifest entry). The server never imports plugin code to
        answer this RPC. Plugins whose manifest entry has no
        ``metadata_path`` — or whose metadata file fails to load —
        contribute no nodes; a warning surfaces and the caller should
        ship metadata.json with the plugin.
        """
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.ListAvailableNodesResponse()

        nodes: list[cuvis_ai_pb2.NodeInfo] = []

        # Built-in nodes ship inside cuvis-ai-core itself, so importing
        # them is free and the per-class spec walk stays.
        for class_name in NodeRegistry.list_builtin_nodes():
            try:
                node_class = NodeRegistry.get_builtin_class(class_name)
            except Exception as e:
                logger.warning(
                    f"Failed to resolve builtin node class '{class_name}': {e}",
                    exc_info=True,
                )
                node_class = None

            if node_class is not None:
                try:
                    input_specs, output_specs = self._extract_port_specs(node_class)
                except Exception as e:
                    logger.warning(
                        f"Failed to extract port specs for builtin node '{class_name}': {e}",
                        exc_info=True,
                    )
                    input_specs = {}
                    output_specs = {}
            else:
                input_specs = {}
                output_specs = {}

            category, tags, icon_svg = _extract_node_metadata(
                node_class, class_name=class_name
            )

            nodes.append(
                cuvis_ai_pb2.NodeInfo(
                    class_name=class_name,
                    full_path=class_name,
                    source="builtin",
                    plugin_name="",
                    input_specs=input_specs,
                    output_specs=output_specs,
                    icon_svg=icon_svg,
                    category=category,
                    tags=tags,
                )
            )

        # Plugin nodes via the static catalog. Plugins whose classes
        # are NEVER touched by this RPC.
        plugins_missing_metadata: list[str] = []
        for plugin_name, config in session.registered_plugins.items():
            try:
                entry = load_catalog_entry(plugin_name, config)
            except (ValueError, FileNotFoundError) as exc:
                logger.warning(
                    f"Failed to load catalog for plugin '{plugin_name}': {exc}"
                )
                continue
            if entry is None:
                plugins_missing_metadata.append(plugin_name)
                continue
            for node_entry in entry.nodes:
                try:
                    nodes.append(_catalog_entry_to_node_info(node_entry, plugin_name))
                except Exception as exc:
                    logger.warning(
                        f"Skipping catalog entry '{node_entry.class_name}' from "
                        f"plugin '{plugin_name}': {exc}"
                    )

        if plugins_missing_metadata:
            logger.warning(
                f"Plugins {sorted(plugins_missing_metadata)} ship no metadata.json — "
                "their nodes will not appear in the palette. Add 'metadata_path' "
                "to the plugin manifest and emit metadata via 'tools/emit_metadata.py'."
            )

        return cuvis_ai_pb2.ListAvailableNodesResponse(nodes=nodes)

    def _extract_port_specs(
        self, node_class: type
    ) -> tuple[
        dict[str, cuvis_ai_pb2.PortSpecList], dict[str, cuvis_ai_pb2.PortSpecList]
    ]:
        """Extract INPUT_SPECS and OUTPUT_SPECS from node class and convert to proto.

        Args:
            node_class: Node class to extract specs from

        Returns:
            Tuple of (input_specs_map, output_specs_map) as proto PortSpecList maps
        """
        input_specs_map = {}
        output_specs_map = {}

        # Extract INPUT_SPECS
        if hasattr(node_class, "INPUT_SPECS"):
            input_specs_dict = getattr(node_class, "INPUT_SPECS")
            for port_name, spec in input_specs_dict.items():
                # Handle variadic ports (list of PortSpec)
                if isinstance(spec, list):
                    proto_specs = []
                    for s in spec:
                        proto_specs.append(_convert_port_spec_to_proto(s, port_name))
                    input_specs_map[port_name] = cuvis_ai_pb2.PortSpecList(
                        specs=proto_specs
                    )
                else:
                    # Single PortSpec
                    proto_spec = _convert_port_spec_to_proto(spec, port_name)
                    input_specs_map[port_name] = cuvis_ai_pb2.PortSpecList(
                        specs=[proto_spec]
                    )

        # Extract OUTPUT_SPECS
        if hasattr(node_class, "OUTPUT_SPECS"):
            output_specs_dict = getattr(node_class, "OUTPUT_SPECS")
            for port_name, spec in output_specs_dict.items():
                # Handle variadic ports (list of PortSpec)
                if isinstance(spec, list):
                    proto_specs = []
                    for s in spec:
                        proto_specs.append(_convert_port_spec_to_proto(s, port_name))
                    output_specs_map[port_name] = cuvis_ai_pb2.PortSpecList(
                        specs=proto_specs
                    )
                else:
                    # Single PortSpec
                    proto_spec = _convert_port_spec_to_proto(spec, port_name)
                    output_specs_map[port_name] = cuvis_ai_pb2.PortSpecList(
                        specs=[proto_spec]
                    )

        return input_specs_map, output_specs_map

    @grpc_handler("Failed to clear cache")
    def clear_plugin_cache(
        self,
        request: cuvis_ai_pb2.ClearPluginCacheRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ClearPluginCacheResponse:
        """Clear Git plugin cache."""
        plugin_name = request.plugin_name if request.plugin_name else None

        # Count cleared before clearing
        cache_dir = NodeRegistry._cache_dir

        if plugin_name:
            cleared = (
                len(list(cache_dir.glob(f"{plugin_name}@*")))
                if cache_dir.exists()
                else 0
            )
        else:
            cleared = len(list(cache_dir.glob("*"))) if cache_dir.exists() else 0

        # Clear cache
        NodeRegistry.clear_plugin_cache(plugin_name)

        logger.info(f"Cleared {cleared} cached plugin(s)")
        return cuvis_ai_pb2.ClearPluginCacheResponse(cleared_count=cleared)


__all__ = ["PluginService"]
