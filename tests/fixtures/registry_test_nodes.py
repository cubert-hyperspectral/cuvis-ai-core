"""Mock nodes for testing NodeRegistry functionality.

These nodes replace cuvis_ai dependencies in test_node_registry.py.
They provide minimal implementations for testing registry operations.
"""

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import Context, ExecutionStage, Metric
import torch


class MockMinMaxNormalizer(Node):
    """Mock normalizer node for testing."""

    INPUT_SPECS = {
        "data": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }
    OUTPUT_SPECS = {
        "normalized": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }

    def __init__(self, eps: float = 1e-6, use_running_stats: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.use_running_stats = use_running_stats

    def forward(self, data, **kwargs):
        # Simple mock normalization
        return {"normalized": data}

    def load(self, params, serial_dir):
        pass


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


class MockLossNode(Node):
    """Mock loss node for testing - accepts any shape tensors."""

    INPUT_SPECS = {
        "predictions": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "targets": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }
    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=()),
    }

    def forward(self, predictions, targets, **kwargs):
        # Flatten and compute simple MSE loss
        pred_flat = predictions.reshape(-1)
        targ_flat = targets.reshape(-1)
        loss = torch.mean((pred_flat - targ_flat) ** 2)
        return {"loss": loss}

    def load(self, params, serial_dir):
        pass


class MockMetricNode(Node):
    """Mock metric node for testing training workflows.

    Executes only during VAL and TEST stages (like real metric nodes).
    Returns metrics in List[Metric] format for proper logging.
    """

    INPUT_SPECS = {
        "predictions": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "targets": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }
    OUTPUT_SPECS = {
        "metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects"),
    }

    def __init__(
        self,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ):
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            **kwargs,
        )

    def forward(self, predictions, targets, context: Context, **kwargs):
        """Compute simple accuracy-like metric (percentage of close predictions)."""
        pred_flat = predictions.reshape(-1)
        targ_flat = targets.reshape(-1)
        close_predictions = torch.abs(pred_flat - targ_flat) < 0.1
        metric_value = close_predictions.float().mean()

        # Return List[Metric] format for proper trainer logging
        return {
            "metrics": [
                Metric(
                    name="accuracy",
                    value=metric_value.item(),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                )
            ]
        }

    def load(self, params, serial_dir):
        pass


class MockBinaryDecider(Node):
    """Mock binary decider for testing runtime validation.

    Mimics the BinaryDecider from cuvis_ai but properly returns bool dtype.
    """

    INPUT_SPECS = {
        "logits": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1)),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1)),
    }

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def forward(self, logits, **kwargs):
        # Apply threshold to get binary decisions
        decisions = logits > self.threshold
        return {"decisions": decisions}

    def load(self, params, serial_dir):
        pass
