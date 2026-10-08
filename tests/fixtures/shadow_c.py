"""Two node classes named Shadowed, one per module, for the registry's exact-path lookup."""

import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


class Shadowed(Node):
    """Passes its input through; only its import path matters to the tests."""

    INPUT_SPECS = {
        "data": PortSpec(dtype=torch.float32, shape=(-1,), description="Input values.")
    }
    OUTPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32, shape=(-1,), description="The input values."
        )
    }

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Return the input unchanged."""
        return {"data": data}
