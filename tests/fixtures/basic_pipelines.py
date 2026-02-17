"""Basic pipeline fixtures for testing without cuvis-ai dependencies."""

from __future__ import annotations

import pytest

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


@pytest.fixture
def simple_two_node_pipeline(simple_input_node, simple_output_node):
    """Create a simple two-node pipeline for testing.

    Structure: Input → Output
    """
    pipeline = CuvisPipeline("test_two_node")

    # Create nodes
    input_node = simple_input_node()
    output_node = simple_output_node()

    # Connect nodes
    pipeline.connect(input_node.outputs.output, output_node.inputs.input)

    return pipeline


@pytest.fixture
def simple_three_node_pipeline(
    simple_input_node, simple_transform_node, simple_output_node
):
    """Create a three-node pipeline for testing.

    Structure: Input → Transform → Output
    """
    pipeline = CuvisPipeline("test_three_node")

    # Create nodes
    input_node = simple_input_node()
    transform_node = simple_transform_node(scale=2.0)
    output_node = simple_output_node()

    # Connect nodes
    pipeline.connect(input_node.outputs.output, transform_node.inputs.input)
    pipeline.connect(transform_node.outputs.output, output_node.inputs.input)

    return pipeline


@pytest.fixture
def pipeline_factory(simple_input_node, simple_transform_node, simple_output_node):
    """Factory fixture for creating customizable pipelines.

    Returns a callable that creates pipelines with configurable parameters.
    """

    def _create_pipeline(
        name: str = "test_pipeline",
        num_nodes: int = 2,
        transform_scale: float = 2.0,
    ):
        """Create a pipeline with specified configuration.

        Args:
            name: Pipeline name
            num_nodes: Number of nodes (2 or 3)
            transform_scale: Scale factor for transform node

        Returns:
            Configured CuvisPipeline instance
        """
        pipeline = CuvisPipeline(name)

        input_node = simple_input_node()
        output_node = simple_output_node()

        if num_nodes == 2:
            pipeline.connect(input_node.outputs.output, output_node.inputs.input)
        elif num_nodes == 3:
            transform_node = simple_transform_node(scale=transform_scale)
            pipeline.connect(input_node.outputs.output, transform_node.inputs.input)
            pipeline.connect(transform_node.outputs.output, output_node.inputs.input)
        else:
            raise ValueError(f"Unsupported num_nodes: {num_nodes}")

        return pipeline

    return _create_pipeline


@pytest.fixture
def trainable_pipeline(simple_input_node, trainable_node, simple_output_node):
    """Create a pipeline with a trainable node for testing training workflows.

    Structure: Input → TrainableNode → Output
    """
    pipeline = CuvisPipeline("test_trainable")

    # Create nodes - input needs to match trainable node's expected input
    _ = simple_input_node(batch_size=4, channels=10, height=1, width=1)
    trainable = trainable_node(input_dim=10, output_dim=5)
    _ = simple_output_node()

    # For trainable node, we need to reshape the input
    # The input is (batch, height, width, channels) = (4, 1, 1, 10)
    # Trainable expects (batch, features) = (4, 10)
    # We'll need a reshape - for now this is a simplified version

    return pipeline, trainable


@pytest.fixture
def multi_branch_pipeline(
    simple_input_node, simple_transform_node, simple_output_node, multi_output_node
):
    """Create a pipeline with multiple branches for testing complex flows.

    Structure:
        Input → MultiOutput → branch_a → Output_A
                           └→ branch_b → Output_B
    """
    pipeline = CuvisPipeline("test_multi_branch")

    _ = simple_input_node(batch_size=2, channels=10, height=1, width=1)
    _ = multi_output_node()
    _ = simple_transform_node(scale=1.5)
    _ = simple_transform_node(scale=0.5)
    _ = simple_output_node()
    _ = simple_output_node()

    # Note: This is a simplified structure
    # Real implementation would need proper port connections

    return pipeline


@pytest.fixture
def data_pipeline(data_node):
    """Create a pipeline with a data node for testing data flow.

    Returns a pipeline that simulates loading and processing hyperspectral data.
    """
    pipeline = CuvisPipeline("test_data_pipeline")

    # Create data node
    node = data_node(batch_size=2, height=64, width=64, channels=5)

    # Add to pipeline (no connections needed for single-node pipeline)
    pipeline.add_node("data", node)

    return pipeline


@pytest.fixture
def pipeline_dict_factory():
    """Factory fixture for creating pipeline configuration dictionaries.

    Returns a callable that creates pipeline dicts with configurable parameters.
    """

    def _create_dict(
        name: str = "test_pipeline",
        num_nodes: int = 2,
        node_class: str = "cuvis_ai_core.node.node.Node",
    ):
        """Create a pipeline configuration dictionary.

        Args:
            name: Pipeline name
            num_nodes: Number of nodes to include
            node_class: Fully qualified node class name

        Returns:
            Pipeline configuration dict
        """
        nodes = {}
        connections = []

        for i in range(num_nodes):
            node_name = f"node_{i}"
            nodes[node_name] = {
                "class_name": node_class,
                "hparams": {},
            }

            # Connect to previous node
            if i > 0:
                connections.append(
                    {
                        "source": f"node_{i - 1}.output",
                        "target": f"{node_name}.input",
                    }
                )

        return {
            "version": "1.0",
            "name": name,
            "nodes": nodes,
            "connections": connections,
        }

    return _create_dict
