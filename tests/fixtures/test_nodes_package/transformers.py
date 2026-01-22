"""Mock transformer nodes for testing auto-registration."""

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec
import torch


class MockTrainablePCA(Node):
    """Mock PCA node for testing."""

    INPUT_SPECS = {
        "data": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }
    OUTPUT_SPECS = {
        "transformed": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }

    def forward(self, data, **kwargs):
        return {"transformed": data}

    def load(self, params, serial_dir):
        pass
