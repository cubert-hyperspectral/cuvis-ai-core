"""gRPC health check service implementation."""

from __future__ import annotations

from collections.abc import Generator
from concurrent import futures
from typing import Any

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

# The health port answers Check without reading the request, so the receive cap only guards a
# handler that reads. A client that never reads its answers keeps each call alive after
# maximum_concurrent_rpcs has released it; the stream and age limits bound those per connection.
_LISTENER_OPTIONS = [
    ("grpc.max_receive_message_length", 1024),
    ("grpc.max_concurrent_streams", 2),
    ("grpc.max_connection_age_ms", 60_000),
    ("grpc.max_connection_age_grace_ms", 5_000),
]
_LISTENER_WORKERS = 2
_LISTENER_MAX_CALLS = 8


class HealthService(health_pb2_grpc.HealthServicer):
    """Standard gRPC health checking service.

    Implements the gRPC Health Checking Protocol:
    https://github.com/grpc/grpc/blob/master/doc/health-checking.md
    """

    def __init__(self) -> None:
        self._status = health_pb2.HealthCheckResponse.SERVING

    def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        """Check health status.

        Args:
            request: Health check request
            context: gRPC context

        Returns:
            Health check response with current status
        """
        return health_pb2.HealthCheckResponse(status=self._status)

    def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> Generator[health_pb2.HealthCheckResponse, None, None]:
        """Stream health status updates.

        Args:
            request: Health check request
            context: gRPC context

        Yields:
            Health check responses as status changes
        """
        # For simplicity, just yield current status once
        # A full implementation would stream status changes
        yield health_pb2.HealthCheckResponse(status=self._status)

    def set_not_serving(self) -> None:
        """Mark service as not serving (for graceful shutdown)."""
        self._status = health_pb2.HealthCheckResponse.NOT_SERVING

    def set_serving(self) -> None:
        """Mark service as serving."""
        self._status = health_pb2.HealthCheckResponse.SERVING


def start_health_listener(
    health: HealthService, address: str
) -> tuple[grpc.Server, int]:
    """Serve ``health``'s Check on ``address`` without a token, for probes that hold none.

    Only ``grpc.health.v1.Health/Check`` is served; ``Watch`` and every other method answer
    UNIMPLEMENTED. The answer does not depend on the request, so the handler never reads it: a
    client that opens a call and withholds its request holds no worker thread.

    Returns the started server and the port it bound (useful when ``address`` ends in ``:0``).
    """

    def check(_requests: Any, context: grpc.ServicerContext) -> Any:
        return health.Check(health_pb2.HealthCheckRequest(), context)

    handler = grpc.method_handlers_generic_handler(
        "grpc.health.v1.Health",
        {
            "Check": grpc.stream_unary_rpc_method_handler(
                check,
                request_deserializer=None,
                response_serializer=health_pb2.HealthCheckResponse.SerializeToString,
            )
        },
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=_LISTENER_WORKERS, thread_name_prefix="cuvis-health"
        ),
        handlers=[handler],
        options=_LISTENER_OPTIONS,
        maximum_concurrent_rpcs=_LISTENER_MAX_CALLS,
    )
    port = server.add_insecure_port(address)
    if port == 0:
        raise RuntimeError(f"Failed to bind the health port at {address}")
    server.start()
    return server, port


__all__ = ["HealthService", "start_health_listener"]
