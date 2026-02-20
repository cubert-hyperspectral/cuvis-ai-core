"""Versioned gRPC stubs for cuvis_ai_core.

Proto message definitions come from cuvis-ai-schemas (single source of truth).
The gRPC service stubs are generated locally.
"""

from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

from . import cuvis_ai_core_pb2, cuvis_ai_core_pb2_grpc

# Alias for backward compatibility — all service code imports cuvis_ai_pb2
cuvis_ai_pb2_grpc = cuvis_ai_core_pb2_grpc

__all__ = [
    "cuvis_ai_core_pb2",
    "cuvis_ai_core_pb2_grpc",
    "cuvis_ai_pb2",
    "cuvis_ai_pb2_grpc",
]
