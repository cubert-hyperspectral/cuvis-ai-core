"""Re-export proto definitions from cuvis_ai_schemas (single source of truth).

The proto descriptors are compiled once in cuvis-ai-schemas; core re-exports
them here so that the generated *_grpc.py stubs and all service code continue
to work without registering duplicate symbols.
"""

from cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2 import *  # noqa: F401,F403
from cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2 import DESCRIPTOR  # noqa: F401
