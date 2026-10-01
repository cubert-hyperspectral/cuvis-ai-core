"""The identifier and attribute helpers of :class:`PipelineVisualizer`."""

from __future__ import annotations

import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.pipeline.visualizer import PipelineVisualizer
from cuvis_ai_schemas.pipeline import PortSpec


class _Source(Node):
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"data": PortSpec(torch.Tensor, (-1, 3))}

    def forward(self, **inputs):
        return {"data": torch.zeros(1, 3)}


class _Sink(Node):
    INPUT_SPECS = {"data": PortSpec(torch.Tensor, (-1, 3))}
    OUTPUT_SPECS: dict[str, PortSpec] = {}

    def forward(self, **inputs):
        return {}


def _visualizer() -> PipelineVisualizer:
    pipeline = CuvisPipeline("helpers")
    source, sink = _Source(name="source"), _Sink(name="sink")
    pipeline.connect(source.outputs.data, sink.inputs.data)
    return PipelineVisualizer(pipeline)


def test_sanitize_identifier_keeps_word_characters_and_optionally_dashes():
    sanitize = PipelineVisualizer._sanitize_identifier
    assert sanitize("rx-detector 1") == "rx-detector_1"
    assert sanitize("rx-detector 1", allow_dash=False) == "rx_detector_1"
    assert sanitize("a.b/c") == "a_b_c"
    assert sanitize("") == "node"


def test_compose_attribute_list_joins_the_inline_attributes():
    visualizer = _visualizer()
    attrs = {"penwidth": 2, "color": "red"}
    composed = visualizer._compose_attribute_list(attrs)
    assert composed == ", ".join(visualizer._format_inline_attributes(attrs))
    assert composed.startswith("penwidth=2, color=")


def test_edge_attributes_reach_the_dot_output():
    dot = _visualizer().to_graphviz(edge_attributes={"penwidth": 2})
    assert "penwidth=2" in dot
