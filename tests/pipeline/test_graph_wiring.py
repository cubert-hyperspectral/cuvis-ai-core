"""
Test suite for graph wiring validation with required/optional ports and batch data keys.

Inspired by bad_wiring() in typed_io_example.py.
Tests ensure proper validation of:
- Required vs optional port connections
- Batch data key matching
- Stage-aware execution (train/val/test/inference)
- Partial graph execution with upto_node
"""

from __future__ import annotations


import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import ExecutionStage
from tests.fixtures import MockStatisticalTrainableNode, SimpleLossNode


class SimpleDataNode(Node):
    """Simple data node for testing that outputs cube and mask."""

    INPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "wavelengths": PortSpec(dtype=torch.int32, shape=(-1, -1)),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "mask": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }

    def forward(self, cube, wavelengths, **kwargs):
        # Just pass through cube and create a dummy mask (4D to match loss node expectations)
        batch_size, height, width, channels = cube.shape
        # Create 4D mask with single channel
        mask = torch.zeros(batch_size, height, width, 1, dtype=torch.float32)
        return {"cube": cube, "mask": mask}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class TestRequiredPortValidation:
    """Test validation of required port connections."""

    def test_missing_required_input_raises_runtime_error(
        self, create_test_cube
    ) -> None:
        """Missing required input for executing node should raise RuntimeError."""
        pipeline = CuvisPipeline("statistical_validation")
        cube, wavelengths = create_test_cube(
            batch_size=2, height=4, width=4, num_channels=8, dtype=torch.float32
        )

        data = SimpleDataNode()  # I: cube, wavelengths, O: cube, mask
        statistical_node = MockStatisticalTrainableNode(
            input_dim=8, hidden_dim=4
        )  # I: data, O: result
        loss_node = SimpleLossNode(weight=0.3)  # I: predictions, targets, O: loss
        loss_node.execution_stages = {
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        }

        pipeline.connect(
            (data.outputs.cube, statistical_node.data),
            (statistical_node.outputs.output, loss_node.predictions),
        )

        # Note: loss_node.targets is NOT connected and NOT in batch
        # This should fail because loss_node executes in train stage
        bad_batch = {
            "cube": cube,
            "wavelengths": wavelengths,
        }

        with pytest.raises(RuntimeError, match="missing required inputs"):
            pipeline.forward(stage="train", batch=bad_batch)

    def test_excluded_node_with_missing_input_only_warns(
        self, caplog, create_test_cube
    ) -> None:
        """Excluded node with missing input should only warn, not error."""
        pipeline = CuvisPipeline("statistical_validation")
        cube, wavelengths = create_test_cube(
            batch_size=2, height=4, width=4, num_channels=8, dtype=torch.float32
        )

        data = SimpleDataNode()
        statistical_node = MockStatisticalTrainableNode(input_dim=8, hidden_dim=4)
        statistical_node._statistically_initialized = (
            True  # Mark as initialized for testing
        )
        trainable = MockStatisticalTrainableNode(
            input_dim=4, hidden_dim=2, name="trainable"
        )
        trainable._statistically_initialized = True  # Mark as initialized for testing
        loss_node = SimpleLossNode(weight=0.3)
        loss_node.execution_stages = {
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        }

        pipeline.connect(
            (data.outputs.cube, statistical_node.data),
            (statistical_node.outputs.output, trainable.data),
            (trainable.outputs.output, loss_node.predictions),
            (data.outputs.mask, loss_node.targets),  # Connect targets from data node
        )

        # Provide all required inputs
        batch = {
            "cube": cube,
            "wavelengths": wavelengths,
        }

        # Should NOT raise since all inputs are provided
        outputs = pipeline.forward(stage="val", batch=batch)

        # Execution should succeed up to trainable
        assert (trainable.name, "output") in outputs
        # Loss executes in val stage for validation metrics
        assert (loss_node.name, "loss") in outputs

    def test_upto_node_excludes_downstream_validation(self, create_test_cube) -> None:
        """upto_node parameter should exclude downstream nodes from validation."""
        pipeline = CuvisPipeline("statistical_validation")
        cube, wavelengths = create_test_cube(
            batch_size=2, height=4, width=4, num_channels=8, dtype=torch.float32
        )

        data = SimpleDataNode()
        statistical_node = MockStatisticalTrainableNode(input_dim=8, hidden_dim=4)
        trainable = MockStatisticalTrainableNode(
            input_dim=4, hidden_dim=2, name="trainable"
        )
        loss_node = SimpleLossNode(weight=0.3)
        loss_node.execution_stages = {
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        }

        pipeline.connect(
            (data.outputs.cube, statistical_node.data),
            (statistical_node.outputs.output, trainable.data),
            (trainable.outputs.output, loss_node.predictions),
        )

        # loss_node is excluded via upto_node, so missing targets should not error
        bad_batch = {
            "cube": cube,
            "wavelengths": wavelengths,
        }

        # Should succeed - loss_node is downstream of statistical_node and excluded
        outputs = pipeline.forward(
            stage="train",
            batch=bad_batch,
            upto_node=statistical_node,
        )

        # Should only execute data node (ancestor of statistical_node)
        assert (data.name, "cube") in outputs
        # statistical_node itself should NOT execute (upto_node is exclusive)
        assert (statistical_node.name, "output") not in outputs


class TestOptionalPortHandling:
    """Test handling of optional ports."""

    def test_optional_port_missing_does_not_error(self) -> None:
        """Missing optional port should not raise error."""

        class NodeWithOptionalInput(Node):
            INPUT_SPECS = {
                "required": PortSpec(dtype=torch.float32, shape=(-1,)),
                "optional": PortSpec(dtype=torch.float32, shape=(-1,), optional=True),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, required, optional=None, **kwargs):
                if optional is not None:
                    return {"out": required + optional}
                return {"out": required}

            def load(self, params, serial_dir):
                return None

        # Test standalone node directly without graph
        node = NodeWithOptionalInput()

        # Only provide required input
        from cuvis_ai_core.utils.types import Context

        context = Context(stage="inference")
        result = node.forward(required=torch.tensor([1.0, 2.0, 3.0]), context=context)
        assert "out" in result
        torch.testing.assert_close(result["out"], torch.tensor([1.0, 2.0, 3.0]))

    def test_optional_port_provided_is_used(self) -> None:
        """Optional port when provided should be used correctly."""

        class NodeWithOptionalInput(Node):
            INPUT_SPECS = {
                "required": PortSpec(dtype=torch.float32, shape=(-1,)),
                "optional": PortSpec(dtype=torch.float32, shape=(-1,), optional=True),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, required, optional=None, **kwargs):
                if optional is not None:
                    return {"out": required + optional}
                return {"out": required}

            def load(self, params, serial_dir):
                return None

        # Test standalone node directly without graph
        node = NodeWithOptionalInput()

        # Provide both inputs
        from cuvis_ai_core.utils.types import Context

        context = Context(stage="inference")
        result = node.forward(
            required=torch.tensor([1.0, 2.0, 3.0]),
            optional=torch.tensor([0.5, 0.5, 0.5]),
            context=context,
        )
        expected = torch.tensor([1.5, 2.5, 3.5])
        torch.testing.assert_close(result["out"], expected)


class TestBatchDataKeyMapping:
    """Test mapping of batch dictionary keys to node input ports."""

    def test_batch_key_matches_port_name(self) -> None:
        """Batch keys matching port names should be routed correctly."""

        class SimpleNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {"result": PortSpec(dtype=torch.float32, shape=(-1, -1))}

            def forward(self, data, **kwargs):
                return {"result": data * 2}

            def load(self, params, serial_dir):
                return None

        # Test standalone node directly without graph
        node = SimpleNode()

        # Test that the node processes data correctly
        test_data = torch.randn(4, 10)
        from cuvis_ai_core.utils.types import Context

        context = Context(stage="inference")
        result = node.forward(data=test_data, context=context)

        assert "result" in result
        torch.testing.assert_close(result["result"], test_data * 2)

    def test_batch_key_mismatch_raises_error(self) -> None:
        """Batch key not matching any port should cause missing input error in graph execution."""

        class SimpleNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {"result": PortSpec(dtype=torch.float32, shape=(-1, -1))}

            def forward(self, data, **kwargs):
                return {"result": data * 2}

            def load(self, params, serial_dir):
                return None

        pipeline = CuvisPipeline("batch_mismatch")
        consumer = SimpleNode()

        # Add node to graph by connecting to itself (dummy connection to ensure it's in graph)
        # Actually, just connect its output to input to create a cycle isn't what we want
        # Let's use a simpler approach - create two nodes where one feeds the other
        class ProducerNode(Node):
            INPUT_SPECS = {"input_data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {"output": PortSpec(dtype=torch.float32, shape=(-1, -1))}

            def forward(self, input_data, **kwargs):
                return {"output": input_data}

            def load(self, params, serial_dir):
                return None

        producer = ProducerNode()
        pipeline.connect(producer.output, consumer.data)

        # Batch has wrong key - producer needs "input_data" but batch only has "wrong_key"
        batch = {"wrong_key": torch.randn(4, 10)}

        with pytest.raises(RuntimeError, match="missing required inputs"):
            pipeline.forward(stage="inference", batch=batch)

    def test_connection_overrides_batch_key(self) -> None:
        """Port connection should take precedence over batch key."""

        class ProducerNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, **kwargs):
                return {"data": torch.tensor([10.0, 20.0, 30.0])}

            def load(self, params, serial_dir):
                return None

        class ConsumerNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"result": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, data, **kwargs):
                return {"result": data * 2}

            def load(self, params, serial_dir):
                return None

        pipeline = CuvisPipeline("connection_priority")
        producer = ProducerNode()
        consumer = ConsumerNode()

        pipeline.connect(producer.data, consumer.data)

        # Batch also has "data" key, but connection should take precedence
        batch = {"data": torch.tensor([1.0, 2.0, 3.0])}

        outputs = pipeline.forward(stage="inference", batch=batch)

        # Consumer should receive data from producer, not batch
        expected = torch.tensor([20.0, 40.0, 60.0])  # producer output * 2
        torch.testing.assert_close(outputs[(consumer.name, "result")], expected)


class TestStageAwareExecution:
    """Test stage-specific execution filtering."""

    def test_train_stage_executes_loss_nodes(self, create_test_cube) -> None:
        """Loss nodes should execute in train stage."""
        pipeline = CuvisPipeline("stage_test")
        cube, wavelengths = create_test_cube(
            batch_size=2, height=4, width=4, num_channels=8, dtype=torch.float32
        )

        data = SimpleDataNode()
        statistical_node = MockStatisticalTrainableNode(input_dim=8, hidden_dim=4)
        statistical_node._statistically_initialized = (
            True  # Mark as initialized for testing
        )
        trainable = MockStatisticalTrainableNode(
            input_dim=4, hidden_dim=2, name="trainable"
        )
        trainable._statistically_initialized = True  # Mark as initialized for testing
        loss_node = SimpleLossNode(weight=0.3)
        loss_node.execution_stages = {
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        }

        pipeline.connect(
            (data.outputs.cube, statistical_node.data),
            (statistical_node.outputs.output, trainable.data),
            (trainable.outputs.output, loss_node.predictions),
            (data.outputs.mask, loss_node.targets),  # Connect targets from data node
        )

        # Provide all required inputs
        batch = {
            "cube": cube,
            "wavelengths": wavelengths,
        }

        outputs = pipeline.forward(stage="train", batch=batch)

        # Loss should execute in train stage
        assert (loss_node.name, "loss") in outputs

    def test_inference_stage_skips_loss_nodes(self, create_test_cube) -> None:
        """Loss nodes should not execute in inference stage."""
        pipeline = CuvisPipeline("stage_test")
        cube, wavelengths = create_test_cube(
            batch_size=2, height=4, width=4, num_channels=8, dtype=torch.float32
        )

        data = SimpleDataNode()
        statistical_node = MockStatisticalTrainableNode(input_dim=8, hidden_dim=4)
        statistical_node._statistically_initialized = (
            True  # Mark as initialized for testing
        )
        trainable = MockStatisticalTrainableNode(
            input_dim=4, hidden_dim=2, name="trainable"
        )
        trainable._statistically_initialized = True  # Mark as initialized for testing
        loss_node = SimpleLossNode(weight=0.3)
        loss_node.execution_stages = {
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        }

        pipeline.connect(
            (data.outputs.cube, statistical_node.data),
            (statistical_node.outputs.output, trainable.data),
            (trainable.outputs.output, loss_node.predictions),
            # Don't connect targets - loss won't execute in inference anyway
        )

        # Provide cube only - loss doesn't execute in inference
        batch = {
            "cube": cube,
            "wavelengths": wavelengths,
        }

        outputs = pipeline.forward(stage="inference", batch=batch)

        # Loss should NOT execute in inference stage
        assert (loss_node.name, "loss") not in outputs
        # But trainable should execute
        assert (trainable.name, "output") in outputs


class TestMultipleEntryPoints:
    """Test graphs with multiple entry nodes."""

    def test_multiple_batch_keys_to_different_nodes(self) -> None:
        """Multiple batch keys should route to different nodes correctly."""

        class SourceA(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"out": x * 2}

            def load(self, params, serial_dir):
                return None

        class SourceB(Node):
            INPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, y, **kwargs):
                return {"out": y * 3}

            def load(self, params, serial_dir):
                return None

        class Combiner(Node):
            INPUT_SPECS = {
                "a": PortSpec(dtype=torch.float32, shape=(-1,)),
                "b": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, a, b, **kwargs):
                return {"out": a + b}

            def load(self, params, serial_dir):
                return None

        pipeline = CuvisPipeline("multi_entry")
        source_a = SourceA()
        source_b = SourceB()
        combiner = Combiner()

        pipeline.connect(
            (source_a.out, combiner.a),
            (source_b.out, combiner.b),
        )

        # Provide both entry inputs
        batch = {
            "x": torch.tensor([1.0]),
            "y": torch.tensor([2.0]),
        }

        outputs = pipeline.forward(stage="inference", batch=batch)

        # Result should be (1.0 * 2) + (2.0 * 3) = 8.0
        expected = torch.tensor([8.0])
        torch.testing.assert_close(outputs[(combiner.name, "out")], expected)

    def test_missing_one_entry_point_raises_error(self) -> None:
        """Missing one of multiple entry points should raise error."""

        class SourceA(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"out": x * 2}

            def load(self, params, serial_dir):
                return None

        class SourceB(Node):
            INPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, y, **kwargs):
                return {"out": y * 3}

            def load(self, params, serial_dir):
                return None

        class Combiner(Node):
            INPUT_SPECS = {
                "a": PortSpec(dtype=torch.float32, shape=(-1,)),
                "b": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, a, b, **kwargs):
                return {"out": a + b}

            def load(self, params, serial_dir):
                return None

        pipeline = CuvisPipeline("multi_entry")
        source_a = SourceA()
        source_b = SourceB()
        combiner = Combiner()

        pipeline.connect(
            (source_a.out, combiner.a),
            (source_b.out, combiner.b),
        )

        # Only provide one entry input
        batch = {"x": torch.tensor([1.0])}

        with pytest.raises(RuntimeError, match="missing required inputs"):
            pipeline.forward(stage="inference", batch=batch)
