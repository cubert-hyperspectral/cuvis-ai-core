"""Tests for sessionless pipeline-config visualization (no node build, no plugin env)."""

from __future__ import annotations

import shutil

import pytest

from cuvis_ai_core.pipeline.config_visualizer import config_to_dot, render_pipeline_config
from cuvis_ai_schemas.pipeline.config import PipelineConfig

_CHAIN_YAML = """
metadata:
  name: Chain_Preview
plugins:
- cuvis_ai_builtin
- sam3
nodes:
- name: sam3_point_expansion
  class_name: cuvis_ai_sam3.node.SAM3PointExpansion
  hparams: {}
- name: mask_cleanup
  class_name: cuvis_ai.node.mask_ops.MaskRobustifier
  hparams:
    min_area: 50
    keep_largest: true
connections:
- source: sam3_point_expansion.outputs.mask
  target: mask_cleanup.inputs.mask
"""

_SINGLE_NODE_YAML = """
metadata:
  name: Single_Node_View
nodes:
- name: sam3_point_expansion
  class_name: cuvis_ai_sam3.node.SAM3PointExpansion
  hparams: {}
connections: []
"""


def _config(yaml_text: str) -> PipelineConfig:
    import yaml

    return PipelineConfig.model_validate(yaml.safe_load(yaml_text))


class TestConfigToDot:
    def test_chain_has_both_nodes_and_the_edge(self) -> None:
        dot = config_to_dot(_config(_CHAIN_YAML))
        assert "digraph" in dot
        # Boxes keyed by node name, labelled with the short class.
        assert '"sam3_point_expansion"' in dot
        assert '"mask_cleanup"' in dot
        assert "SAM3PointExpansion" in dot
        assert "MaskRobustifier" in dot
        # One directed edge for the single connection, labelled with the shared port name.
        assert '"sam3_point_expansion" -> "mask_cleanup"' in dot
        assert dot.count("->") == 1
        assert "mask" in dot

    def test_single_node_has_no_edges(self) -> None:
        dot = config_to_dot(_config(_SINGLE_NODE_YAML))
        assert '"sam3_point_expansion"' in dot
        assert "->" not in dot

    def test_empty_pipeline_renders_placeholder(self) -> None:
        dot = config_to_dot(PipelineConfig())
        assert "empty pipeline" in dot
        assert "->" not in dot

    def test_distinct_ports_are_labelled_directionally(self) -> None:
        yaml_text = """
nodes:
- name: a
  class_name: pkg.A
- name: b
  class_name: pkg.B
connections:
- source: a.outputs.scores
  target: b.inputs.data
"""
        dot = config_to_dot(_config(yaml_text))
        assert "scores → data" in dot


class TestRenderPipelineConfig:
    def test_dot_format_returns_dot_text(self) -> None:
        data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="dot")
        assert fmt == "dot"
        assert b"digraph" in data

    def test_unknown_format_without_graphviz_falls_back_to_dot(self, monkeypatch) -> None:
        # Force the graphviz import to fail so the render path takes the DOT fallback.
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "graphviz":
                raise ImportError("graphviz unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="png")
        assert fmt == "dot"
        assert b"digraph" in data

    @pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz 'dot' binary not installed")
    def test_png_render_returns_png_bytes(self) -> None:
        data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="png")
        assert fmt == "png"
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
