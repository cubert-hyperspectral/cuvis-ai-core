"""Tests for sessionless pipeline-config visualization (no node build, no plugin env)."""

from __future__ import annotations

import shutil

import pytest

from cuvis_ai_core.pipeline.config_visualizer import (
    config_to_dot,
    render_pipeline_config,
)
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

    def test_falls_back_to_dot_when_dot_binary_missing(self, monkeypatch) -> None:
        # dot not on PATH -> subprocess.run raises FileNotFoundError -> DOT fallback.
        import subprocess

        def _no_dot(*args, **kwargs):
            raise FileNotFoundError("dot")

        monkeypatch.setattr(subprocess, "run", _no_dot)
        data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="png")
        assert fmt == "dot"
        assert b"digraph" in data

    @pytest.mark.skipif(
        shutil.which("dot") is None, reason="graphviz 'dot' binary not installed"
    )
    def test_png_render_returns_png_bytes(self) -> None:
        data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="png")
        assert fmt == "png"
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


_DANGLING_YAML = """
metadata:
  name: Dangling
nodes:
- name: mask_cleanup
  class_name: cuvis_ai.node.mask_ops.MaskRobustifier
  hparams: {min_area: 1}
connections:
- source: mask_cleanup.outputs.mask
  target: ghost_sink.inputs.mask
"""


def test_config_to_dot_boxes_endpoint_absent_from_nodes():
    """A connection endpoint missing from nodes[] still gets a minimal box, so a
    partial config renders instead of emitting a dangling edge."""
    dot = config_to_dot(_config(_DANGLING_YAML))

    # ghost_sink is only named in a connection, never declared as a node.
    assert '"ghost_sink" [label="ghost_sink"];' in dot
    assert '"mask_cleanup" -> "ghost_sink"' in dot


def test_render_returns_image_bytes_on_success(monkeypatch):
    """Cover the image-render success path without depending on the ``dot`` binary
    (CI has no graphviz binary, so the real-render test is skipped there)."""
    import subprocess

    fake_png = b"\x89PNG\r\n\x1a\n-fake"

    def _fake_run(cmd, **kwargs):
        assert cmd[0] == "dot" and cmd[1] == "-Tpng"
        return subprocess.CompletedProcess(cmd, 0, stdout=fake_png, stderr=b"")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="png")

    assert fmt == "png"
    assert data == fake_png


def test_render_falls_back_to_dot_on_nonzero_exit(monkeypatch):
    """A nonzero `dot` exit degrades to DOT bytes instead of propagating."""
    import subprocess

    def _fail(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", _fail)

    data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="png")

    assert fmt == "dot"
    assert b"digraph" in data


def test_render_falls_back_to_dot_on_timeout(monkeypatch):
    """A hung `dot` is killed by the timeout and degrades to DOT bytes."""
    import subprocess

    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 10.0))

    monkeypatch.setattr(subprocess, "run", _timeout)

    data, fmt = render_pipeline_config(_CHAIN_YAML, fmt="png")

    assert fmt == "dot"
    assert b"digraph" in data


def test_render_rejects_oversize_config(monkeypatch):
    """An oversize config is a client error (ValueError), not a silent DOT fallback."""
    import cuvis_ai_core.pipeline.config_visualizer as cv

    monkeypatch.setattr(cv, "_MAX_CONFIG_BYTES", 10)
    with pytest.raises(ValueError, match="too large"):
        render_pipeline_config(_CHAIN_YAML, fmt="png")


def test_render_rejects_too_many_elements(monkeypatch):
    """A config over the node+connection cap raises before any render."""
    import cuvis_ai_core.pipeline.config_visualizer as cv

    monkeypatch.setattr(cv, "_MAX_GRAPH_ELEMENTS", 1)
    with pytest.raises(ValueError, match="render cap"):
        render_pipeline_config(_CHAIN_YAML, fmt="png")  # 2 nodes + 1 conn > 1
