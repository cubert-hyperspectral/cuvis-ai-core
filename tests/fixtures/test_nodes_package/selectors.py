"""Mock selector nodes for testing auto-registration."""

from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline.ports import PortSpec
import torch


class MockSoftChannelSelector(Node):
    """Mock channel selector node for testing."""

    INPUT_SPECS = {
        "data": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }
    OUTPUT_SPECS = {
        "selected": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }

    def forward(self, data, **kwargs):
        return {"selected": data}

    def load(self, params, serial_dir):
        pass
