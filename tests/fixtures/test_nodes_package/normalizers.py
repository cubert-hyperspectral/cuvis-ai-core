"""Mock normalizer nodes for testing auto-registration."""

from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
import torch


class MockMinMaxNormalizer(Node):
    """Mock normalizer node for testing."""

    INPUT_SPECS = {
        "data": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }
    OUTPUT_SPECS = {
        "normalized": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }

    def forward(self, data, **kwargs):
        # Simple mock normalization
        return {"normalized": data}

    def load(self, params, serial_dir):
        pass
