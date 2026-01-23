"""
Integration tests for complete typed I/O pipelines.
Tests end-to-end functionality of the typed I/O system.
"""

import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import ExecutionStage
from tests.fixtures import (
    MinMaxNormalizer,
    MockStatisticalTrainableNode,
    SoftChannelSelector,
)


class TestLinearPipeline:
    """Test simple linear pipelines."""

    def test_normalizer_selector_rx_pipeline(self):
        """Test complete pipeline: normalizer -> selector -> trainable node."""
        pipeline = CuvisPipeline("test_pipeline")

        # Create nodes (selector now accepts input_channels as parameter)
        normalizer = MinMaxNormalizer()
        selector = SoftChannelSelector(n_select=15, input_channels=50)
        trainable = MockStatisticalTrainableNode(input_dim=50, hidden_dim=10)

        # Connect (nodes auto-added)
        pipeline.connect(
            (normalizer.outputs.normalized, selector.inputs.data),
            (selector.outputs.selected, trainable.inputs.data),
        )

        # Initialize trainable node with training data (must match selector output shape: 50 channels)
        train_data = [{"data": torch.randn(4, 10, 10, 50)} for _ in range(5)]
        trainable.statistical_initialization(iter(train_data))

        # Execute with pipeline.forward()
        input_cube = torch.randn(2, 10, 10, 50)
        batch = {"data": input_cube}
        outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

        # Verify outputs exist
        assert (trainable.name, "output") in outputs
        output = outputs[(trainable.name, "output")]
        assert output.shape == (2, 10, 10, 10)  # hidden_dim=10
        assert torch.all(torch.isfinite(output))

    def test_executor_reuse_efficiency(self):
        """Test that executor can be reused for multiple batches."""
        pipeline = CuvisPipeline("test_pipeline")

        normalizer = MinMaxNormalizer()
        selector = SoftChannelSelector(n_select=10, input_channels=50)

        pipeline.connect(normalizer.outputs.normalized, selector.inputs.data)

        # Reuse for multiple batches
        for _ in range(3):
            batch_data = torch.randn(2, 10, 10, 50)
            # Use simple port name for entry point
            outputs = pipeline.forward(
                stage=ExecutionStage.INFERENCE, batch={"data": batch_data}
            )
            assert (selector.name, "selected") in outputs
            assert outputs[(selector.name, "selected")].shape == (2, 10, 10, 50)


class TestComplexDAG:
    """Test complex DAG patterns."""

    def test_multi_output_connections(self):
        """Test node with multiple outputs feeding different nodes (fan-out)."""
        pipeline = CuvisPipeline("multi_output")

        # Selector output feeds two different trainable nodes (fan-out pattern)
        normalizer = MinMaxNormalizer()
        selector = SoftChannelSelector(n_select=15, input_channels=50)
        trainable1 = MockStatisticalTrainableNode(input_dim=50, hidden_dim=10)
        trainable2 = MockStatisticalTrainableNode(input_dim=50, hidden_dim=10)

        # Connect selector output to two different trainable nodes
        pipeline.connect(
            (normalizer.outputs.normalized, selector.inputs.data),
            (selector.outputs.selected, trainable1.inputs.data),
            (selector.outputs.selected, trainable2.inputs.data),
        )

        # Initialize both trainable nodes with training data
        # (must match selector output shape: 50 channels)
        train_data = [{"data": torch.randn(4, 10, 10, 50)} for _ in range(5)]
        trainable1.statistical_initialization(iter(train_data))
        trainable2.statistical_initialization(iter(train_data))

        input_cube = torch.randn(2, 10, 10, 50)
        outputs = pipeline.forward(
            stage=ExecutionStage.INFERENCE, batch={"data": input_cube}
        )

        # Both trainable nodes should have outputs
        assert (trainable1.name, "output") in outputs
        assert (trainable2.name, "output") in outputs

        # Both should have same shape
        assert (
            outputs[(trainable1.name, "output")].shape
            == outputs[(trainable2.name, "output")].shape
        )


class TestGradientFlow:
    """Test gradient flow in complete pipelines."""

    def test_end_to_end_gradient_flow(self):
        """Test gradients flow through entire pipeline."""
        pipeline = CuvisPipeline("gradient_test")

        normalizer = MinMaxNormalizer()
        selector = SoftChannelSelector(n_select=15, input_channels=50)
        trainable = MockStatisticalTrainableNode(input_dim=50, hidden_dim=10)

        pipeline.connect(
            (normalizer.outputs.normalized, selector.inputs.data),
            (selector.outputs.selected, trainable.inputs.data),
        )

        # Initialize trainable node with training data (must match selector output shape: 50 channels)
        train_data = [{"data": torch.randn(4, 10, 10, 50)} for _ in range(5)]
        trainable.statistical_initialization(iter(train_data))

        # Unfreeze both selector and trainable node to enable gradient flow
        selector.unfreeze()
        trainable.unfreeze()

        # Forward pass with gradient tracking
        input_cube = torch.randn(2, 10, 10, 50, requires_grad=True)
        outputs = pipeline.forward(
            stage=ExecutionStage.TRAIN, batch={"data": input_cube}
        )

        # Backward pass
        loss = outputs[(trainable.name, "output")].sum()
        loss.backward()

        # Check gradients flow to input
        assert input_cube.grad is not None
        assert torch.all(torch.isfinite(input_cube.grad))

        # Check gradients flow to selector parameters
        assert selector.channel_logits.grad is not None
        assert torch.all(torch.isfinite(selector.channel_logits.grad))

    def test_no_gradient_detachment(self):
        """Test that executor doesn't detach tensors."""
        pipeline = CuvisPipeline("detach_test")

        normalizer = MinMaxNormalizer()
        selector = SoftChannelSelector(n_select=2, input_channels=3)

        # Connect nodes so they're part of the graph
        pipeline.connect(normalizer.outputs.normalized, selector.inputs.data)

        # Use 4D tensor as expected by normalizer (BHWC format)
        input_data = torch.randn(1, 2, 2, 3, requires_grad=True)
        # Use simple port name for entry point
        outputs = pipeline.forward(
            stage=ExecutionStage.INFERENCE, batch={"data": input_data}
        )

        # Output should still require grad
        output = outputs[(normalizer.name, "normalized")]
        assert output.requires_grad


class TestBatchDistribution:
    """Test batch key distribution to multiple nodes."""

    def test_batch_keys_distributed_to_all_requesting_nodes(self):
        """Test that batch keys are distributed to all nodes that request them."""

        class DataSource(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"features": PortSpec(dtype=torch.float32, shape=(-1, 10))}

            def forward(self, **kwargs):
                return {"features": torch.randn(5, 10)}

        class DataConsumer(Node):
            INPUT_SPECS = {"features": PortSpec(dtype=torch.float32, shape=(-1, 10))}
            OUTPUT_SPECS = {"result": PortSpec(dtype=torch.float32, shape=(-1, 10))}

            def forward(self, features, **kwargs):
                return {"result": features * 2}

        pipeline = CuvisPipeline("batch_dist")
        source = DataSource()
        consumer1 = DataConsumer()
        consumer2 = DataConsumer()

        # Connect source to both consumers (fan-out pattern)
        pipeline.connect(
            (source.outputs.features, consumer1.inputs.features),
            (source.outputs.features, consumer2.inputs.features),
        )

        # Execute the graph
        outputs = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

        # Both consumers should have received the data from source
        assert (consumer1.name, "result") in outputs
        assert (consumer2.name, "result") in outputs


class TestStageAwareExecution:
    """Test execution stage filtering."""

    def test_train_only_nodes_skip_during_inference(self):
        """Test that train-only nodes don't execute during inference."""

        class AlwaysNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {"out": torch.tensor(1.0)}

        class TrainOnlyNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=())}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def __init__(self):
                super().__init__(execution_stages={ExecutionStage.TRAIN})

            def forward(self, x, **kwargs):
                return {"out": x * 2}

        pipeline = CuvisPipeline("stage_test")
        always_node = AlwaysNode()
        train_node = TrainOnlyNode()

        pipeline.connect(always_node.outputs.out, train_node.inputs.x)

        # Inference stage: only always_node executes
        outputs_inf = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)
        assert (always_node.name, "out") in outputs_inf
        assert (train_node.name, "out") not in outputs_inf

        # Train stage: both execute
        outputs_train = pipeline.forward(batch={}, stage=ExecutionStage.TRAIN)
        assert (always_node.name, "out") in outputs_train
        assert (train_node.name, "out") in outputs_train


class TestPortCompatibility:
    """Test port compatibility validation."""

    def test_incompatible_dtype_raises_error(self):
        """Test that connecting incompatible dtypes raises error."""
        from cuvis_ai_core.pipeline.ports import PortCompatibilityError

        class FloatNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {"out": torch.tensor(1.0)}

        class IntNode(Node):
            INPUT_SPECS = {"input": PortSpec(dtype=torch.int64, shape=())}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("compat_test")
        float_node = FloatNode()
        int_node = IntNode()

        with pytest.raises(PortCompatibilityError, match="[Dd]type"):
            pipeline.connect(float_node.outputs.out, int_node.inputs.input)

    def test_compatible_flexible_shapes_succeed(self):
        """Test that flexible dimensions allow connection."""

        class FlexNode1(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1, 10))}

            def forward(self, **inputs):
                return {"out": torch.randn(5, 10)}

        class FlexNode2(Node):
            INPUT_SPECS = {"input": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("flex_test")
        n1 = FlexNode1()
        n2 = FlexNode2()

        # Should not raise
        pipeline.connect(n1.outputs.out, n2.inputs.input)
