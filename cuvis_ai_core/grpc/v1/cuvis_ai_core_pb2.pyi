"""Type stub mirroring cuvis_ai_core_pb2.py — a re-export of cuvis-ai-schemas.

The proto message descriptors are compiled once in cuvis-ai-schemas (single
source of truth). This stub re-exports them so type checkers resolve every
message (``PortSpec``, ``NodeInfo``, ...) against that one definition instead of
a frozen, drift-prone copy. Keep it in lock-step with ``cuvis_ai_core_pb2.py``.
"""

from cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2 import *  # noqa: F401,F403
from cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2 import DESCRIPTOR as DESCRIPTOR  # noqa: F401
