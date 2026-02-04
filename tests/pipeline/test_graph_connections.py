"""
Test suite for Graph connection API with port-based wiring.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.pipeline.ports import PortCompatibilityError, PortSpec


class TestGraphConnectionBasics:
    """Test basic connection operations."""

    def test_connect_two_nodes_single_syntax(self) -> None:
        """Connecting two nodes with single connection syntax should succeed."""

        class ProducerNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "output": PortSpec(dtype=torch.float32, shape=(-1, 10)),
            }

            def forward(self, **inputs):
                return {"output": torch.zeros(1, 10)}

        class ConsumerNode(Node):
            INPUT_SPECS = {
                "input": PortSpec(dtype=torch.float32, shape=(-1, 10)),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        producer = ProducerNode()
        consumer = ConsumerNode()

        # Nodes are automatically added when connecting
        pipeline.connect(producer.outputs.output, consumer.input)

        assert pipeline._graph.has_edge(producer, consumer)
        edge_dict = pipeline._graph.get_edge_data(producer, consumer)
        assert edge_dict is not None
        edge_data = next(iter(edge_dict.values()))
        assert edge_data["from_port"] == "output"
        assert edge_data["to_port"] == "input"

    def test_connect_multiple_connections_batch_syntax(self) -> None:
        """Batch connection syntax should create multiple edges."""

        class SourceNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "out1": PortSpec(dtype=torch.float32, shape=(-1,)),
                "out2": PortSpec(dtype=torch.float32, shape=(-1,)),
            }

            def forward(self, **inputs):
                return {"out1": torch.zeros(10), "out2": torch.ones(10)}

        class TargetNode(Node):
            INPUT_SPECS = {
                "in1": PortSpec(dtype=torch.float32, shape=(-1,)),
                "in2": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        source = SourceNode()
        target = TargetNode()

        # Nodes are automatically added when connecting
        pipeline.connect(
            (source.outputs.out1, target.in1),
            (source.outputs.out2, target.in2),
        )

        edge_dict = pipeline._graph.get_edge_data(source, target)
        assert edge_dict is not None
        assert len(edge_dict) == 2


class TestConnectionValidation:
    """Test connection validation logic."""

    def test_connecting_incompatible_dtypes_raises_error(self) -> None:
        """Connecting different dtypes should raise PortCompatibilityError."""

        class FloatNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {}

        class IntNode(Node):
            INPUT_SPECS = {"in": PortSpec(dtype=torch.int64, shape=())}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        float_node = FloatNode()
        int_node = IntNode()

        # Connecting incompatible types should raise error
        with pytest.raises(PortCompatibilityError, match=r"(?i)dtype"):
            pipeline.connect(
                float_node.outputs.out, int_node.inputs.__getattribute__("in")
            )

    def test_connecting_incompatible_shapes_raises_error(self) -> None:
        """Connecting mismatched shapes should raise PortCompatibilityError."""

        class Shape10Node(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(10,))}

            def forward(self, **inputs):
                return {}

        class Shape20Node(Node):
            INPUT_SPECS = {"in": PortSpec(dtype=torch.float32, shape=(20,))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        node_10 = Shape10Node()
        node_20 = Shape20Node()

        # Connecting incompatible shapes should raise error
        with pytest.raises(PortCompatibilityError, match=r"(?i)dimension"):
            pipeline.connect(node_10.outputs.out, node_20.inputs.__getattribute__("in"))

    def test_connecting_compatible_flexible_shapes_succeeds(self) -> None:
        """Flexible shapes with -1 should be compatible."""

        class FlexNode1(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1, 10))}

            def forward(self, **inputs):
                return {}

        class FlexNode2(Node):
            INPUT_SPECS = {"in": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        node_1 = FlexNode1()
        node_2 = FlexNode2()

        # Flexible shapes should be compatible
        pipeline.connect(
            node_1.outputs.out, node_2.inputs.__getattribute__("in")
        )  # Should not raise


class TestMultiDiGraphIntegration:
    """Test MultiDiGraph structure and edge attributes."""

    def test_graph_uses_multidigraph(self) -> None:
        """Graph.graph should be a NetworkX MultiDiGraph."""

        pipeline = CuvisPipeline("test")
        assert isinstance(pipeline._graph, nx.MultiDiGraph)

    def test_multiple_connections_between_same_nodes(self) -> None:
        """Multiple connections between same node pair should be supported."""

        class MultiOutNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "out1": PortSpec(dtype=torch.float32, shape=()),
                "out2": PortSpec(dtype=torch.float32, shape=()),
            }

            def forward(self, **inputs):
                return {}

        class MultiInNode(Node):
            INPUT_SPECS = {
                "in1": PortSpec(dtype=torch.float32, shape=()),
                "in2": PortSpec(dtype=torch.float32, shape=()),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        src = MultiOutNode()
        tgt = MultiInNode()

        # Multiple connections between same nodes
        pipeline.connect(
            (src.outputs.out1, tgt.in1),
            (src.outputs.out2, tgt.in2),
        )

        edge_dict = pipeline._graph.get_edge_data(src, tgt)
        assert edge_dict is not None
        assert len(edge_dict) == 2

    def test_edge_iteration_includes_all_connections(self) -> None:
        """Edge iteration should include all multi-edges with data."""

        class Node1(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "a": PortSpec(dtype=torch.float32, shape=()),
                "b": PortSpec(dtype=torch.float32, shape=()),
            }

            def forward(self, **inputs):
                return {}

        class Node2(Node):
            INPUT_SPECS = {
                "x": PortSpec(dtype=torch.float32, shape=()),
                "y": PortSpec(dtype=torch.float32, shape=()),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        node_1 = Node1()
        node_2 = Node2()

        # Connect nodes
        pipeline.connect((node_1.a, node_2.x), (node_1.b, node_2.y))

        edges = list(pipeline._graph.edges(keys=True, data=True))
        assert len(edges) == 2
        for _, _, _, edge_data in edges:
            assert "from_port" in edge_data and "to_port" in edge_data


class TestDuplicateConnectionPrevention:
    """Test connection uniqueness and idempotency using edge keys."""

    def test_duplicate_connection_is_idempotent(self) -> None:
        """Connecting same port pair twice should be idempotent (no duplicate edge)."""

        # Setup: Define simple producer and consumer nodes
        class Producer(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"output": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, **inputs):
                return {"output": torch.zeros(10)}

        class Consumer(Node):
            INPUT_SPECS = {"input": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        producer = Producer()
        consumer = Consumer()

        # First connection succeeds
        pipeline.connect(producer.outputs.output, consumer.input)

        # Verify edge exists with correct key
        edge_dict = pipeline._graph.get_edge_data(producer, consumer)
        assert edge_dict is not None
        assert len(edge_dict) == 1
        assert ("output", "input") in edge_dict  # Key is tuple of port names

        # Second connection (should be idempotent - no error, no duplicate)
        pipeline.connect(producer.outputs.output, consumer.input)  # No error!

        # Verify still only one edge
        edge_dict = pipeline._graph.get_edge_data(producer, consumer)
        assert len(edge_dict) == 1  # Still just one edge
        assert ("output", "input") in edge_dict

    def test_different_port_pairs_create_separate_edges(self) -> None:
        """Multiple different connections between same nodes should succeed."""

        class MultiOutNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "out1": PortSpec(dtype=torch.float32, shape=()),
                "out2": PortSpec(dtype=torch.float32, shape=()),
            }

            def forward(self, **inputs):
                return {"out1": torch.tensor(1.0), "out2": torch.tensor(2.0)}

        class MultiInNode(Node):
            INPUT_SPECS = {
                "in1": PortSpec(dtype=torch.float32, shape=()),
                "in2": PortSpec(dtype=torch.float32, shape=()),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        src = MultiOutNode()
        tgt = MultiInNode()

        # Connect two different port pairs
        pipeline.connect(
            (src.outputs.out1, tgt.in1),
            (src.outputs.out2, tgt.in2),
        )

        # Should have 2 edges with different keys
        edge_dict = pipeline._graph.get_edge_data(src, tgt)
        assert len(edge_dict) == 2
        assert ("out1", "in1") in edge_dict
        assert ("out2", "in2") in edge_dict

    def test_batch_duplicate_in_same_call_is_idempotent(self) -> None:
        """Batch connect with duplicate in same call should be idempotent."""

        class Producer(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"output": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, **inputs):
                return {"output": torch.zeros(10)}

        class Consumer(Node):
            INPUT_SPECS = {"input": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        producer = Producer()
        consumer = Consumer()

        # Batch connect with duplicate - second should update first
        pipeline.connect(
            (producer.outputs.output, consumer.input),
            (producer.outputs.output, consumer.input),  # Duplicate!
        )

        # Should have only 1 edge
        edge_dict = pipeline._graph.get_edge_data(producer, consumer)
        assert len(edge_dict) == 1
        assert ("output", "input") in edge_dict

    def test_edge_key_structure(self) -> None:
        """Edge keys should be (from_port, to_port) tuples."""

        class Producer(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"my_output": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {"my_output": torch.tensor(1.0)}

        class Consumer(Node):
            INPUT_SPECS = {"my_input": PortSpec(dtype=torch.float32, shape=())}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        producer = Producer()
        consumer = Consumer()

        pipeline.connect(producer.outputs.my_output, consumer.my_input)

        # Verify key structure
        edge_dict = pipeline._graph.get_edge_data(producer, consumer)
        keys = list(edge_dict.keys())
        assert len(keys) == 1

        key = keys[0]
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert key == ("my_output", "my_input")

    def test_edge_attributes_preserved(self) -> None:
        """Edge attributes (from_port, to_port) should be preserved."""

        class Producer(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"output": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {"output": torch.tensor(1.0)}

        class Consumer(Node):
            INPUT_SPECS = {"input": PortSpec(dtype=torch.float32, shape=())}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        producer = Producer()
        consumer = Consumer()

        pipeline.connect(producer.outputs.output, consumer.input)

        # Verify edge attributes
        edge_dict = pipeline._graph.get_edge_data(producer, consumer)
        edge_data = edge_dict[("output", "input")]

        assert edge_data["from_port"] == "output"
        assert edge_data["to_port"] == "input"
