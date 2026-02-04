"""
Port system for typed I/O in cuvis.ai pipelines.

This module provides:
- InputPort/OutputPort: Proxy objects representing node ports
- DimensionResolver: Utility for resolving symbolic dimensions
- PortCompatibilityError: Exception for incompatible connections

Note: PortSpec has been moved to cuvis-ai-schemas package.
Import it from: from cuvis_ai_schemas.pipeline.ports import PortSpec
"""

from __future__ import annotations

from typing import TYPE_CHECKING


# PortSpec has been migrated to cuvis-ai-schemas
from cuvis_ai_schemas.pipeline.ports import PortSpec

if TYPE_CHECKING:
    from cuvis_ai_core.node.node import Node


class PortCompatibilityError(Exception):
    """Raised when attempting to connect incompatible ports."""


class DimensionResolver:
    """Utility class for resolving symbolic dimensions in port shapes."""

    @staticmethod
    def resolve(
        shape: tuple[int | str, ...],
        node: Node | None,
    ) -> tuple[int, ...]:
        """Resolve symbolic dimensions to concrete values.

        Parameters
        ----------
        shape : tuple[int | str, ...]
            Shape specification with flexible (-1), fixed (int), or symbolic (str) dims.
        node : Node | None
            Node instance to resolve symbolic dimensions from.

        Returns
        -------
        tuple[int, ...]
            Resolved shape with concrete integer values.

        Raises
        ------
        AttributeError
            If symbolic dimension references non-existent node attribute.
        """
        resolved: list[int] = []
        for dim in shape:
            if isinstance(dim, int):
                # Flexible (-1) or fixed (int) dimension
                resolved.append(dim)
                continue

            if isinstance(dim, str):
                # Symbolic dimension - resolve from node
                if node is None:
                    raise ValueError(
                        f"Cannot resolve symbolic dimension '{dim}' without node instance"
                    )
                if not hasattr(node, dim):
                    node_label = getattr(node, "id", None) or node
                    raise AttributeError(
                        f"Node {node_label} has no attribute '{dim}' for dimension resolution"
                    )

                value = getattr(node, dim)
                if not isinstance(value, int):
                    raise TypeError(
                        f"Dimension '{dim}' resolved to {type(value)}, expected int"
                    )
                resolved.append(value)
                continue

            raise TypeError(f"Invalid dimension type: {type(dim)}")

        return tuple(resolved)


class OutputPort:
    """Proxy object representing a node's output port."""

    def __init__(self, node: Node, name: str, spec: PortSpec) -> None:
        self.node = node
        self.name = name
        self.spec = spec

    def __repr__(self) -> str:
        node_id = getattr(self.node, "id", None) or self.node
        return f"OutputPort({node_id}.{self.name})"


class InputPort:
    """Proxy object representing a node's input port."""

    def __init__(self, node: Node, name: str, spec: PortSpec) -> None:
        self.node = node
        self.name = name
        self.spec = spec

    def __repr__(self) -> str:
        node_id = getattr(self.node, "id", None) or self.node
        return f"InputPort({node_id}.{self.name})"


__all__ = [
    "DimensionResolver",
    "InputPort",
    "OutputPort",
    "PortCompatibilityError",
]
