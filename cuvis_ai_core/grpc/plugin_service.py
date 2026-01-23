"""Plugin management service component."""

from __future__ import annotations


import grpc
from loguru import logger

from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import PluginManifest


class PluginService:
    """gRPC service layer for plugin management operations."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def load_plugins(
        self,
        request: cuvis_ai_pb2.LoadPluginsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPluginsResponse:
        """Load plugins from JSON manifest into session."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadPluginsResponse()

        if not request.manifest or not request.manifest.config_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("manifest.config_bytes is required")
            return cuvis_ai_pb2.LoadPluginsResponse()

        try:
            # Parse JSON → Pydantic (following existing pattern)
            manifest_json = request.manifest.config_bytes.decode("utf-8")
            manifest = PluginManifest.model_validate_json(manifest_json)

            loaded = []
            failed = {}

            # Load each plugin into session's registry instance
            for plugin_name, config in manifest.plugins.items():
                try:
                    session.node_registry.load_plugin(
                        plugin_name,
                        config.model_dump(),
                    )

                    # Track in session
                    session.loaded_plugins[plugin_name] = config.model_dump()
                    loaded.append(plugin_name)

                    logger.info(
                        f"Loaded plugin '{plugin_name}' in session {request.session_id}"
                    )
                except Exception as e:
                    failed[plugin_name] = str(e)
                    logger.error(f"Failed to load plugin '{plugin_name}': {e}")

            return cuvis_ai_pb2.LoadPluginsResponse(
                loaded_plugins=loaded, failed_plugins=failed
            )

        except Exception as e:
            logger.error(f"LoadPlugins failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to load plugins: {e}")
            return cuvis_ai_pb2.LoadPluginsResponse()

    def list_loaded_plugins(
        self,
        request: cuvis_ai_pb2.ListLoadedPluginsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ListLoadedPluginsResponse:
        """List plugins loaded in session."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.ListLoadedPluginsResponse()

        plugins = []
        for name, config in session.loaded_plugins.items():
            plugin_type = "git" if "repo" in config else "local"
            source = config.get("repo") or config.get("path", "")
            ref = config.get("ref", "")
            provides = config.get("provides", [])

            plugins.append(
                cuvis_ai_pb2.PluginInfo(
                    name=name,
                    type=plugin_type,
                    source=source,
                    ref=ref,
                    provides=provides,
                )
            )

        return cuvis_ai_pb2.ListLoadedPluginsResponse(plugins=plugins)

    def get_plugin_info(
        self,
        request: cuvis_ai_pb2.GetPluginInfoRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPluginInfoResponse:
        """Get information about specific plugin."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetPluginInfoResponse()

        if request.plugin_name not in session.loaded_plugins:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Plugin '{request.plugin_name}' not loaded in session")
            return cuvis_ai_pb2.GetPluginInfoResponse()

        config = session.loaded_plugins[request.plugin_name]
        plugin_type = "git" if "repo" in config else "local"
        source = config.get("repo") or config.get("path", "")
        ref = config.get("ref", "")
        provides = config.get("provides", [])

        return cuvis_ai_pb2.GetPluginInfoResponse(
            plugin=cuvis_ai_pb2.PluginInfo(
                name=request.plugin_name,
                type=plugin_type,
                source=source,
                ref=ref,
                provides=provides,
            )
        )

    def list_available_nodes(
        self,
        request: cuvis_ai_pb2.ListAvailableNodesRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ListAvailableNodesResponse:
        """List all available nodes (built-in + session plugins)."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.ListAvailableNodesResponse()

        nodes = []

        # Built-in nodes
        for class_name in NodeRegistry.list_builtin_nodes():
            nodes.append(
                cuvis_ai_pb2.NodeInfo(
                    class_name=class_name,
                    full_path=class_name,  # Builtin nodes use short names
                    source="builtin",
                    plugin_name="",
                )
            )

        # Session plugin nodes from registry instance
        plugin_nodes = sorted(session.node_registry.plugin_registry.keys())

        for class_name in plugin_nodes:
            # Find which plugin provides this node
            plugin_name = ""
            full_path = class_name
            for pname, config in session.loaded_plugins.items():
                for provided_path in config.get("provides", []):
                    if provided_path.endswith(class_name) or provided_path.endswith(
                        f".{class_name}"
                    ):
                        plugin_name = pname
                        full_path = provided_path
                        break
                if plugin_name:
                    break

            nodes.append(
                cuvis_ai_pb2.NodeInfo(
                    class_name=class_name,
                    full_path=full_path,
                    source="plugin",
                    plugin_name=plugin_name,
                )
            )

        return cuvis_ai_pb2.ListAvailableNodesResponse(nodes=nodes)

    def clear_plugin_cache(
        self,
        request: cuvis_ai_pb2.ClearPluginCacheRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ClearPluginCacheResponse:
        """Clear Git plugin cache."""
        plugin_name = request.plugin_name if request.plugin_name else None

        try:
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

        except Exception as e:
            logger.error(f"ClearPluginCache failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to clear cache: {e}")
            return cuvis_ai_pb2.ClearPluginCacheResponse(cleared_count=0)


__all__ = ["PluginService"]
