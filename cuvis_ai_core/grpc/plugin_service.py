"""Plugin management service component."""

from __future__ import annotations


import grpc
import numpy as np
import torch
from loguru import logger

from cuvis_ai_core.grpc.helpers import DTYPE_NUMPY_TO_PROTO, DTYPE_TORCH_TO_PROTO
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_schemas.pipeline import PortSpec
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import PluginManifest


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
    # Map Python dtype to proto DType enum
    proto_dtype = cuvis_ai_pb2.D_TYPE_UNSPECIFIED

    # Try torch dtype mapping first
    if isinstance(spec.dtype, torch.dtype):
        if spec.dtype in DTYPE_TORCH_TO_PROTO:
            proto_dtype = DTYPE_TORCH_TO_PROTO[spec.dtype]
        else:
            raise ValueError(f"Unsupported torch dtype: {spec.dtype}")
    # Try numpy dtype mapping
    elif hasattr(spec.dtype, "dtype"):
        # Handle numpy scalar types (np.int32, np.float32, etc.)
        np_dtype = np.dtype(spec.dtype)
        if np_dtype in DTYPE_NUMPY_TO_PROTO:
            proto_dtype = DTYPE_NUMPY_TO_PROTO[np_dtype]
        else:
            raise ValueError(f"Unsupported numpy dtype: {spec.dtype}")
    # Handle torch.Tensor as a generic tensor type (use UNSPECIFIED)
    elif spec.dtype is torch.Tensor:
        proto_dtype = cuvis_ai_pb2.D_TYPE_UNSPECIFIED
    # Handle Python built-in types (dict, str, list, etc.) as UNSPECIFIED
    elif isinstance(spec.dtype, type):
        # Python types like dict, str, list, tuple, etc.
        proto_dtype = cuvis_ai_pb2.D_TYPE_UNSPECIFIED
    else:
        raise ValueError(f"Unsupported dtype type: {type(spec.dtype)}")

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
            # Get the node class to extract port specs
            try:
                node_class = NodeRegistry.get_builtin_class(class_name)
                input_specs, output_specs = self._extract_port_specs(node_class)
            except Exception as e:
                logger.warning(
                    f"Failed to extract port specs for builtin node '{class_name}': {e}",
                    exc_info=True,  # Include full traceback
                )
                input_specs = {}
                output_specs = {}

            nodes.append(
                cuvis_ai_pb2.NodeInfo(
                    class_name=class_name,
                    full_path=class_name,  # Builtin nodes use short names
                    source="builtin",
                    plugin_name="",
                    input_specs=input_specs,
                    output_specs=output_specs,
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

            # Get the node class to extract port specs
            try:
                node_class = session.node_registry.plugin_registry[class_name]
                input_specs, output_specs = self._extract_port_specs(node_class)
            except Exception as e:
                logger.warning(
                    f"Failed to extract port specs for plugin node '{class_name}': {e}",
                    exc_info=True,  # Include full traceback
                )
                input_specs = {}
                output_specs = {}

            nodes.append(
                cuvis_ai_pb2.NodeInfo(
                    class_name=class_name,
                    full_path=full_path,
                    source="plugin",
                    plugin_name=plugin_name,
                    input_specs=input_specs,
                    output_specs=output_specs,
                )
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
