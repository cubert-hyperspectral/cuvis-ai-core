"""
Port system for typed I/O in cuvis.ai pipelines.

This module re-exports all port-related classes from cuvis-ai-schemas.

All port system functionality has been moved to the cuvis-ai-schemas package:
- PortSpec: Specification for input/output ports with type and shape constraints
- InputPort/OutputPort: Proxy objects representing node ports
- DimensionResolver: Utility for resolving symbolic dimensions
- PortCompatibilityError: Exception for incompatible connections
"""

from __future__ import annotations

from cuvis_ai_schemas.pipeline.exceptions import PortCompatibilityError
from cuvis_ai_schemas.pipeline.ports import (
    DimensionResolver,
    InputPort,
    OutputPort,
    PortSpec,
)

__all__ = [
    "DimensionResolver",
    "InputPort",
    "OutputPort",
    "PortCompatibilityError",
    "PortSpec",
]
