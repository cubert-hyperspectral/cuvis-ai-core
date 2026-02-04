"""Versioned gRPC stubs - now sourced from cuvis-ai-schemas package."""

# Import from external package
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc

# Maintain backward compatibility aliases (deprecated)
cuvis_ai_core_pb2 = cuvis_ai_pb2
cuvis_ai_core_pb2_grpc = cuvis_ai_pb2_grpc

__all__ = [
    "cuvis_ai_core_pb2",  # Deprecated - use cuvis_ai_pb2
    "cuvis_ai_core_pb2_grpc",  # Deprecated - use cuvis_ai_pb2_grpc
    "cuvis_ai_pb2",
    "cuvis_ai_pb2_grpc",
]
