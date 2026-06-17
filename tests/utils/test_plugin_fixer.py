"""Tests for the suggest-plugins-fix tool."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from cuvis_ai_core.utils.plugin_fixer import (
    _discover_plugins_dirs,
    reorder_pipeline_with_plugins,
    suggest_plugins_field,
    suggest_plugins_fix_cli,
)
from cuvis_ai_schemas.pipeline import PipelineConfig

_PIPELINE_BODY = """
metadata:
  name: p
nodes:
  - name: n
    class_name: cuvis_ai.node.normalization.MinMaxNormalizer
    hparams: {}
connections: []
"""


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
        name: cuvis_ai_builtin
        path: "../.."
        capabilities:
          - class_name: cuvis_ai.node.anomaly.rx_detector.RXGlobal
          - class_name: cuvis_ai.node.normalization.MinMaxNormalizer
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
        suggest_plugins_field(pipeline_config, raw, [workspace / "configs" / "plugins"])


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
            sys.executable,
            "-m",
            "cuvis_ai_core.utils.plugin_fixer",
            "--pipeline-path",
            str(pipeline_yaml),
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
            sys.executable,
            "-m",
            "cuvis_ai_core.utils.plugin_fixer",
            "--pipeline-path",
            str(pipeline_yaml),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    # No patched yaml on stdout, only the "nothing to do" log line went to stderr.
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# CLI, called in-process so coverage measures suggest_plugins_fix_cli
# ---------------------------------------------------------------------------


def test_cli_missing_pipeline_file_returns_1(tmp_path):
    rc = suggest_plugins_fix_cli(["--pipeline-path", str(tmp_path / "nope.yaml")])
    assert rc == 1


def test_cli_non_dict_yaml_returns_1(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    rc = suggest_plugins_fix_cli(["--pipeline-path", str(bad)])
    assert rc == 1


def test_cli_yaml_output_emits_patched_yaml(workspace, capsys):
    pipeline = workspace / "p.yaml"
    _write(pipeline, _PIPELINE_BODY)
    rc = suggest_plugins_fix_cli(["--pipeline-path", str(pipeline)])
    assert rc == 0
    patched = yaml.safe_load(capsys.readouterr().out)
    assert patched["plugins"] == ["cuvis_ai_builtin"]
    assert list(patched.keys()) == ["metadata", "plugins", "nodes", "connections"]


def test_cli_diff_output_emits_unified_diff(workspace, capsys):
    pipeline = workspace / "p.yaml"
    _write(pipeline, _PIPELINE_BODY)
    rc = suggest_plugins_fix_cli(["--pipeline-path", str(pipeline), "--output", "diff"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("---") or "@@" in out
    assert "plugins" in out


def test_cli_json_output_emits_envelope(workspace, capsys):
    import json

    pipeline = workspace / "p.yaml"
    _write(pipeline, _PIPELINE_BODY)
    rc = suggest_plugins_fix_cli(["--pipeline-path", str(pipeline), "--output", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plugins"] == ["cuvis_ai_builtin"]
    assert payload["pipeline"].endswith("p.yaml")
    assert "diff" in payload


def test_cli_already_declared_returns_0_without_output(workspace, capsys):
    pipeline = workspace / "p.yaml"
    _write(
        pipeline,
        """
        metadata:
          name: p
        plugins:
          - cuvis_ai_builtin
        nodes: []
        connections: []
        """,
    )
    rc = suggest_plugins_fix_cli(["--pipeline-path", str(pipeline)])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_cli_unresolvable_class_returns_1(workspace):
    pipeline = workspace / "p.yaml"
    _write(
        pipeline,
        """
        metadata:
          name: p
        nodes:
          - name: n
            class_name: totally.unknown.MysteryNode
            hparams: {}
        connections: []
        """,
    )
    rc = suggest_plugins_fix_cli(["--pipeline-path", str(pipeline)])
    assert rc == 1


def test_discover_plugins_dirs_walks_up_and_extends_explicit(workspace):
    nested = workspace / "a" / "b"
    nested.mkdir(parents=True)
    pipeline = nested / "p.yaml"
    pipeline.write_text("metadata:\n  name: p\n", encoding="utf-8")

    explicit = Path("/explicit/plugins")
    dirs = _discover_plugins_dirs(pipeline, [explicit])

    assert (workspace / "configs" / "plugins") in dirs
    assert dirs[-1] == explicit
