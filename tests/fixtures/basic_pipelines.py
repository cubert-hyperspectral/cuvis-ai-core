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
