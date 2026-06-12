"""Workspace split resolution service (the ``ResolveSplits`` RPC).

Thin delegation layer: the request carries a JSON-serialized
``SplitsResolveConfig``; the named DataModule class is looked up in the
session's registry and its ``resolve_splits`` classmethod owns the strategy
semantics (e.g. ``cu3s_workspace``: anomaly-aware random/stratified with
train = normal frames only). Core stays plugin-agnostic — it never imports a
concrete DataModule.
"""

from __future__ import annotations

import json

import grpc

from .error_handling import get_session_or_error, grpc_handler
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class SplitsService:
    """Handles workspace split resolution."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    @grpc_handler("Failed to resolve splits")
    def resolve_splits(
        self,
        request: cuvis_ai_pb2.ResolveSplitsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ResolveSplitsResponse:
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.ResolveSplitsResponse()

        from cuvis_ai_schemas.training.data import SplitsResolveConfig

        config = SplitsResolveConfig.from_json(request.config_bytes.decode("utf-8"))

        data_modules = getattr(session.node_registry, "data_modules", {})
        cls = data_modules.get(config.data_module)
        if cls is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"no plugin provides data module {config.data_module!r}; "
                f"available: {sorted(data_modules)}"
            )
            return cuvis_ai_pb2.ResolveSplitsResponse()

        hook = getattr(cls, "resolve_splits", None)
        if hook is None:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(
                f"data module {config.data_module!r} does not support split "
                f"resolution (no resolve_splits classmethod)"
            )
            return cuvis_ai_pb2.ResolveSplitsResponse()

        payload, written_path = hook(config.to_dict())
        return cuvis_ai_pb2.ResolveSplitsResponse(
            splits_bytes=json.dumps(payload).encode("utf-8"),
            splits_path=written_path or "",
        )


__all__ = ["SplitsService"]
