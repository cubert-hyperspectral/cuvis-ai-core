"""Tests for the Phase 4 suggest-plugins-fix tool."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from cuvis_ai_core.utils.plugin_fixer import (
    reorder_pipeline_with_plugins,
    suggest_plugins_field,
)
from cuvis_ai_schemas.pipeline import PipelineConfig


def _write(path: Path, body: str) -> None:
    path.write_text(dedent(body), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Standard layout: <root>/configs/plugins/cuvis_ai_builtin.yaml + a pipeline yaml."""
    plugins_dir = tmp_path / "configs" / "plugins"
    plugins_dir.mkdir(parents=True)
    _write(
        plugins_dir / "cuvis_ai_builtin.yaml",
        """
        plugins:
          cuvis_ai_builtin:
            path: "../.."
            provides:
              - cuvis_ai.node.anomaly.rx_detector.RXGlobal
              - cuvis_ai.node.normalization.MinMaxNormalizer
        """,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# reorder_pipeline_with_plugins
# ---------------------------------------------------------------------------


def test_reorder_places_plugins_after_metadata_before_nodes():
    """Canonical leading order: metadata, plugins, nodes, connections."""
    original = {
        "metadata": {"name": "p"},
        "nodes": [{"name": "n1", "class_name": "pkg.N1"}],
        "connections": [],
    }
    rebuilt = reorder_pipeline_with_plugins(original, ["cuvis_ai_builtin"])
    keys = list(rebuilt.keys())
    assert keys == ["metadata", "plugins", "nodes", "connections"]
    assert rebuilt["plugins"] == ["cuvis_ai_builtin"]


def test_reorder_preserves_unknown_top_level_keys():
    """Unknown keys (e.g. a future ``version`` field) must survive the reorder."""
    original = {
        "metadata": {"name": "p"},
        "version": "0.7",
        "nodes": [],
        "connections": [],
        "defaults": [{"dataset": "rx"}],
    }
    rebuilt = reorder_pipeline_with_plugins(original, ["x"])
    keys = list(rebuilt.keys())
    # Known keys come first in canonical order, unknown keys preserved after.
    assert keys[:4] == ["metadata", "plugins", "nodes", "connections"]
    assert set(keys[4:]) == {"version", "defaults"}
    assert rebuilt["version"] == "0.7"
    assert rebuilt["defaults"] == [{"dataset": "rx"}]


# ---------------------------------------------------------------------------
# suggest_plugins_field
# ---------------------------------------------------------------------------


def test_suggest_returns_resolved_names_and_patched_dict(workspace: Path):
    pipeline_yaml = workspace / "p.yaml"
    _write(
        pipeline_yaml,
        """
        metadata:
          name: p
        nodes:
          - name: n
            class_name: cuvis_ai.node.normalization.MinMaxNormalizer
            hparams: {}
        connections: []
        """,
    )
    raw = yaml.safe_load(pipeline_yaml.read_text(encoding="utf-8"))
    pipeline_config = PipelineConfig.load_from_file(pipeline_yaml)

    names, patched = suggest_plugins_field(
        pipeline_config, raw, [workspace / "configs" / "plugins"]
    )
    assert names == ["cuvis_ai_builtin"]
    assert patched["plugins"] == ["cuvis_ai_builtin"]
    assert list(patched.keys()) == ["metadata", "plugins", "nodes", "connections"]


def test_suggest_refuses_when_plugins_already_present(workspace: Path):
    pipeline_yaml = workspace / "p.yaml"
    _write(
        pipeline_yaml,
        """
        metadata:
          name: p
        plugins:
          - cuvis_ai_builtin
        nodes:
          - name: n
            class_name: cuvis_ai.node.normalization.MinMaxNormalizer
        connections: []
        """,
    )
    raw = yaml.safe_load(pipeline_yaml.read_text(encoding="utf-8"))
    pipeline_config = PipelineConfig.load_from_file(pipeline_yaml)

    with pytest.raises(ValueError, match="already declares 'plugins:'"):
        suggest_plugins_field(
            pipeline_config, raw, [workspace / "configs" / "plugins"]
        )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_yaml_output_round_trips(workspace: Path):
    """`suggest-plugins-fix --output yaml` emits a patched yaml on stdout."""
    pipeline_yaml = workspace / "p.yaml"
    _write(
        pipeline_yaml,
        """
        metadata:
          name: p
        nodes:
          - name: n
            class_name: cuvis_ai.node.normalization.MinMaxNormalizer
            hparams: {}
        connections: []
        """,
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "cuvis_ai_core.utils.plugin_fixer",
            "--pipeline-path", str(pipeline_yaml),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    patched = yaml.safe_load(result.stdout)
    assert patched["plugins"] == ["cuvis_ai_builtin"]


def test_cli_idempotent_when_already_declared(workspace: Path):
    pipeline_yaml = workspace / "p.yaml"
    _write(
        pipeline_yaml,
        """
        metadata:
          name: p
        plugins:
          - cuvis_ai_builtin
        nodes: []
        connections: []
        """,
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "cuvis_ai_core.utils.plugin_fixer",
            "--pipeline-path", str(pipeline_yaml),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    # No patched yaml on stdout — only the "nothing to do" log line went to stderr.
    assert result.stdout == ""
