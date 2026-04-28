from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.pipeline.visualizer import PipelineVisualizer


class ProducerNode(Node):
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "data": PortSpec(torch.Tensor, (-1, 3), "feature tensor"),
    }

    def forward(self, **inputs):
        return {"data": torch.zeros(1, 3)}


class ConsumerNode(Node):
    INPUT_SPECS = {
        "data": PortSpec(torch.Tensor, (-1, 3)),
    }
    OUTPUT_SPECS: dict[str, PortSpec] = {}

    def forward(self, **inputs):
        return {}


def _build_pipeline() -> CuvisPipeline:
    pipeline = CuvisPipeline("phase3")
    source = ProducerNode(name="source", execution_stages={ExecutionStage.ALWAYS})
    sink = ConsumerNode(name="sink", execution_stages={ExecutionStage.TRAIN})
    pipeline.connect(source.outputs.data, sink.inputs.data)
    return pipeline, source, sink


def test_graphviz_supports_phase3_styling():
    pipeline, source, sink = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot = visualizer.to_graphviz(
        node_type_resolver=lambda node: "producer"
        if isinstance(node, ProducerNode)
        else "consumer",
        node_colors={"producer": "#cfe2ff", "consumer": "#fde2e1"},
        group_by_stage=True,
        stage_labels={ExecutionStage.TRAIN.value: "Training"},
        show_port_types=True,
        graph_attributes={"bgcolor": "transparent"},
        node_attributes={"fontname": "Helvetica"},
        show_execution_stage=True,
    )

    assert 'fillcolor="#cfe2ff"' in dot
    assert 'fillcolor="#fde2e1"' in dot
    assert 'subgraph "cluster_train"' in dot
    assert 'label="Training"' in dot
    assert "data (Tensor [-1, 3]) -> data (Tensor [-1, 3])" in dot
    assert 'bgcolor="transparent"' in dot
    assert 'fontname="Helvetica"' in dot
    assert "Stage: All Stages" in dot
    assert "Stage: Training" in dot


def test_mermaid_supports_phase3_features():
    pipeline, source, sink = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    mermaid = visualizer.to_mermaid(
        direction="TB",
        node_type_resolver=lambda node: "producer"
        if isinstance(node, ProducerNode)
        else "consumer",
        node_colors={"producer": "#cfe2ff"},
        group_by_stage=True,
        stage_labels={ExecutionStage.TRAIN.value: "Training"},
        show_port_types=True,
        show_execution_stage=True,
    )

    assert "subgraph train[Training]" in mermaid
    assert "direction TB" in mermaid
    assert "classDef producer fill:#cfe2ff" in mermaid
    assert "class source" in mermaid
    assert "data (Tensor [-1, 3]) -->" in mermaid
    assert "Stage: All Stages" in mermaid
    assert "Stage: Training" in mermaid


def test_pipeline_visualize_method_proxy():
    pipeline, *_ = _build_pipeline()

    dot = pipeline.visualize(format="graphviz")
    mermaid = pipeline.visualize(format="mermaid")

    assert dot.startswith("digraph")
    assert mermaid.startswith("flowchart")


@pytest.mark.requires_data
def test_pipeline_visualize_can_render_to_file(tmp_path):
    pipeline, *_ = _build_pipeline()
    output_svg = tmp_path / "viz"
    output_md = tmp_path / "viz.md"

    # Rendering requires graphviz dependency; skip gracefully if unavailable
    try:
        path = pipeline.visualize(
            format="render_graphviz",
            output_path=output_svg,
            show_execution_stage=True,
        )
        assert path.is_file()
    except RuntimeError as exc:
        if "graphviz package is required" not in str(exc):
            raise

    path_md = pipeline.visualize(
        format="render_mermaid", output_path=output_md, wrap_markdown=False
    )
    assert path_md.is_file()
    assert path_md.read_text().startswith("flowchart")


def test_normalize_stage_with_string():
    """Cover the string branch of _normalize_stage."""
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    assert visualizer._normalize_stage("TRAIN") == "train"
    assert visualizer._normalize_stage("Inference") == "inference"


def test_normalize_stage_with_enum():
    """Verify enum branch of _normalize_stage still works."""
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    assert (
        visualizer._normalize_stage(ExecutionStage.TRAIN) == ExecutionStage.TRAIN.value
    )


def test_classic_style_unchanged_when_style_omitted():
    """Default behaviour must stay byte-identical to the pre-card-mode output."""
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot_default = visualizer.to_graphviz()
    dot_classic = visualizer.to_graphviz(style="classic")

    assert dot_default == dot_classic
    # Classic output keeps the plain-box shape and quoted labels.
    assert "node [shape=box];" in dot_default
    assert 'label="source' in dot_default
    # No HTML-table scaffolding leaks into classic output.
    assert "<TABLE" not in dot_default
    assert "STYLE=\"ROUNDED\"" not in dot_default


def test_card_style_renders_html_table_label():
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot = visualizer.to_graphviz(style="card")

    # Outer DOT wiring switches to plaintext so the HTML table IS the node.
    assert "node [shape=plaintext];" in dot
    assert 'shape="plaintext"' in dot
    # HTML-label markers are present (angle-bracketed, not quoted).
    assert "label=<<TABLE" in dot
    assert 'STYLE="ROUNDED"' in dot
    # Class name is the card title, not node.name.
    assert "ProducerNode" in dot
    assert "ConsumerNode" in dot
    # A "Node N" index appears for each card.
    assert "Node 1" in dot
    assert "Node 2" in dot


def test_card_style_uses_port_anchors_and_drops_edge_label():
    """Card mode attaches edges to port dots via HTML PORT ids, no edge label."""
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot = visualizer.to_graphviz(style="card")

    # Edge anchors: source output port → target input port (with compass pts).
    assert '"source":"out_data":e -> "sink":"in_data":w' in dot
    # Port names are now shown inside the card, not on the edge.
    assert 'label="data -> data"' not in dot
    assert 'label="data"' not in dot
    # Port name appears inside the HTML label of each card.
    assert ">data</FONT>" in dot


def test_card_style_keeps_port_types_when_requested():
    """show_port_types=True overrides the dedupe and shows the full label."""
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot = visualizer.to_graphviz(style="card", show_port_types=True)

    assert "data (Tensor [-1, 3]) -> data (Tensor [-1, 3])" in dot


def test_card_style_omits_plugin_pill_for_unknown_classes_by_default():
    """Unknown classes (including built-in catalog nodes) must NOT get the
    Plugin pill unless a ``NodeRegistry`` instance lists them as plugins."""
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot = visualizer.to_graphviz(style="card")

    assert "Plugin" not in dot


def test_card_style_marks_plugins_when_registry_lists_them():
    """Passing a NodeRegistry with the class in plugin_registry adds the pill."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    pipeline, *_ = _build_pipeline()
    registry = NodeRegistry()
    registry.plugin_registry["ProducerNode"] = ProducerNode

    dot = PipelineVisualizer(pipeline).to_graphviz(
        style="card", node_registry=registry
    )

    assert "Plugin" in dot


def test_card_style_show_node_name_adds_subtitle():
    pipeline, *_ = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot_without = visualizer.to_graphviz(style="card", show_node_name=False)
    dot_with = visualizer.to_graphviz(style="card", show_node_name=True)

    # Node name "source" only appears under card title when the flag is on.
    # (It still appears as the dot identifier "source" -> "sink" in both.)
    assert dot_with.count("source") > dot_without.count("source")


def test_pipeline_visualize_forwards_card_style():
    """CuvisPipeline.visualize should pass `style` through to to_graphviz."""
    pipeline, *_ = _build_pipeline()

    dot = pipeline.visualize(format="graphviz", style="card")

    assert "label=<<TABLE" in dot


class TripleProducer(Node):
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "alpha": PortSpec(torch.float32, (-1,)),
        "beta":  PortSpec(torch.int64,   (-1,)),
        "gamma": PortSpec(torch.bool,    (-1,)),
    }

    def forward(self, **_):
        return {
            "alpha": torch.zeros(1),
            "beta":  torch.zeros(1, dtype=torch.int64),
            "gamma": torch.zeros(1, dtype=torch.bool),
        }


class TripleConsumer(Node):
    INPUT_SPECS = {
        "alpha": PortSpec(torch.float32, (-1,)),
        "beta":  PortSpec(torch.int64,   (-1,)),
        "gamma": PortSpec(torch.bool,    (-1,)),
    }
    OUTPUT_SPECS: dict[str, PortSpec] = {}

    def forward(self, **_):
        return {}


def test_card_style_emits_one_wire_per_port_with_anchors():
    """Each port gets its own dtype-colored wire anchored to its dot;
    multi-edges between the same (src, dst) are NOT bundled."""
    pipeline = CuvisPipeline("per_port_wires")
    producer = TripleProducer(name="prod")
    consumer = TripleConsumer(name="cons")
    pipeline.connect(producer.outputs.alpha, consumer.inputs.alpha)
    pipeline.connect(producer.outputs.beta,  consumer.inputs.beta)
    pipeline.connect(producer.outputs.gamma, consumer.inputs.gamma)

    visualizer = PipelineVisualizer(pipeline)
    dot = visualizer.to_graphviz(style="card")

    edge_lines = [
        line for line in dot.splitlines() if '"prod"' in line and "->" in line
    ]
    # Three distinct edges, one per port pair.
    assert len(edge_lines) == 3, edge_lines

    # Each edge is anchored to its specific port dot.
    assert any('"prod":"out_alpha":e -> "cons":"in_alpha":w' in line for line in edge_lines)
    assert any('"prod":"out_beta":e -> "cons":"in_beta":w' in line for line in edge_lines)
    assert any('"prod":"out_gamma":e -> "cons":"in_gamma":w' in line for line in edge_lines)

    # Each wire carries its own dtype color, none are merged stripes.
    for line in edge_lines:
        assert 'color="' in line
        # Single hex color, not the "#aaa:#bbb:#ccc" stripe form.
        assert line.count(":") <= 4  # 2x node:port:compass only
