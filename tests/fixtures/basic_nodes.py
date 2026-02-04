"""Basic node fixtures for testing without cuvis-ai dependencies."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.pipeline.ports import PortSpec


@pytest.fixture
def simple_input_node():
    """Create a simple input node class for testing.

    Returns a node that outputs a configurable tensor.
    """

    class SimpleInputNode(Node):
        """Simple input node that generates synthetic data."""

        OUTPUT_SPECS = {
            "output": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        }

        def __init__(
            self,
            batch_size: int = 2,
            channels: int = 5,
            height: int = 64,
            width: int = 64,
        ):
            super().__init__()
            self.batch_size = batch_size
            self.channels = channels
            self.height = height
            self.width = width

        def forward(self, **kwargs):
            # Generate random data
            data = torch.randn(self.batch_size, self.height, self.width, self.channels)
            return {"output": data}

    return SimpleInputNode


@pytest.fixture
def simple_transform_node():
    """Create a simple transform node class for testing.

    Returns a node that performs a basic transformation on input data.
    """

    class SimpleTransformNode(Node):
        """Simple transform node that multiplies input by a factor."""

        INPUT_SPECS = {
            "input": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        }
        OUTPUT_SPECS = {
            "output": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        }

        def __init__(self, scale: float = 2.0):
            super().__init__()
            self.scale = scale

        def forward(self, input, **kwargs):
            return {"output": input * self.scale}

    return SimpleTransformNode


@pytest.fixture
def simple_output_node():
    """Create a simple output node class for testing.

    Returns a node that accepts any input and passes it through.
    """

    class SimpleOutputNode(Node):
        """Simple output node that accepts input."""

        INPUT_SPECS = {
            "input": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        }
        OUTPUT_SPECS = {
            "result": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        }

        def forward(self, input, **kwargs):
            return {"result": input}

    return SimpleOutputNode


@pytest.fixture
def trainable_node():
    """Create a trainable node with learnable parameters.

    Returns a node with trainable weights for testing training workflows.
    """

    class TrainableNode(Node):
        """Node with learnable parameters for testing."""

        INPUT_SPECS = {
            "data": PortSpec(dtype=torch.float32, shape=(-1, -1)),
        }
        OUTPUT_SPECS = {
            "transformed": PortSpec(dtype=torch.float32, shape=(-1, -1)),
        }

        def __init__(self, input_dim: int = 10, output_dim: int = 5):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim

            # Learnable linear transformation
            self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
            self.bias = torch.nn.Parameter(torch.zeros(output_dim))

        def forward(self, data, **kwargs):
            # Simple linear transformation
            result = torch.matmul(data, self.weight.t()) + self.bias
            return {"transformed": result}

    return TrainableNode


@pytest.fixture
def statistically_initializable_node():
    """Create a node that can be statistically initialized.

    Returns a node that computes statistics over data batches.
    """

    class StatisticalNode(Node):
        """Node that can be initialized with data statistics."""

        INPUT_SPECS = {
            "data": PortSpec(dtype=torch.float32, shape=(-1, -1)),
        }
        OUTPUT_SPECS = {
            "normalized": PortSpec(dtype=torch.float32, shape=(-1, -1)),
        }

        def __init__(self, num_features: int = 10):
            super().__init__()
            self.num_features = num_features

            # Statistics to be initialized
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_std", torch.ones(num_features))
            self._statistically_initialized = False

        def initialize_statistics(self, data: torch.Tensor):
            """Initialize statistics from data."""
            self.running_mean = data.mean(dim=0)
            self.running_std = data.std(dim=0) + 1e-6
            self._statistically_initialized = True

        def forward(self, data, **kwargs):
            if not self._statistically_initialized:
                # If not initialized, just pass through
                return {"normalized": data}

            # Normalize using statistics
            normalized = (data - self.running_mean) / self.running_std
            return {"normalized": normalized}

    return StatisticalNode


@pytest.fixture
def multi_output_node():
    """Create a node with multiple outputs for testing complex pipelines."""

    class MultiOutputNode(Node):
        """Node that produces multiple outputs."""

        INPUT_SPECS = {
            "data": PortSpec(dtype=torch.float32, shape=(-1, -1)),
        }
        OUTPUT_SPECS = {
            "output_a": PortSpec(dtype=torch.float32, shape=(-1, -1)),
            "output_b": PortSpec(dtype=torch.float32, shape=(-1, -1)),
            "metadata": PortSpec(dtype=torch.float32, shape=(-1,)),
        }

        def forward(self, data, **kwargs):
            return {
                "output_a": data * 2.0,
                "output_b": data + 1.0,
                "metadata": data.mean(dim=1),
            }

    return MultiOutputNode


@pytest.fixture
def data_node():
    """Create a data node that emulates a dataset loader.

    Returns a node that can output batch dictionaries with various data types.
    """

    class DataNode(Node):
        """Node that produces batch dictionaries like a dataset loader."""

        OUTPUT_SPECS = {
            "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
            "wavelengths": PortSpec(dtype=torch.int32, shape=(-1, -1)),
            "mask": PortSpec(dtype=torch.bool, shape=(-1, -1, -1)),
        }

        def __init__(
            self,
            batch_size: int = 2,
            height: int = 64,
            width: int = 64,
            channels: int = 5,
        ):
            super().__init__()
            self.batch_size = batch_size
            self.height = height
            self.width = width
            self.channels = channels

        def forward(self, **kwargs):
            # Simulate hyperspectral cube data
            cube = torch.randn(self.batch_size, self.height, self.width, self.channels)

            # Simulate wavelength metadata
            wavelengths = torch.arange(self.channels, dtype=torch.int32).unsqueeze(0)
            wavelengths = wavelengths.repeat(self.batch_size, 1)

            # Simulate binary mask (e.g., for anomaly detection)
            mask = torch.rand(self.batch_size, self.height, self.width) > 0.5

            return {
                "cube": cube,
                "wavelengths": wavelengths,
                "mask": mask,
            }

    return DataNode
