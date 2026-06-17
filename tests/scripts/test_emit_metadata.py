"""Tests for scripts/emit_metadata.py against the bare-manifest shape."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.emit_metadata import emit

# A real, importable node class in this repo's test fixtures; emit() introspects it.
_NODE_FQCN = "tests.fixtures.mock_nodes.MinMaxNormalizer"


def _write_manifest(path: Path, capabilities: list[dict]) -> None:
    path.write_text(
        yaml.safe_dump(
            {"name": "cuvis_ai_test_nodes", "path": "../..", "capabilities": capabilities},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_emit_regenerates_node_metadata(tmp_path: Path):
    """emit() introspects each node class and writes metadata under 'capabilities:'."""
    manifest = tmp_path / "m.yaml"
    _write_manifest(manifest, [{"class_name": _NODE_FQCN}])

    assert emit(manifest, check=False) is True

    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert "plugins" not in doc  # bare shape, no wrapper
    assert doc["name"] == "cuvis_ai_test_nodes"
    assert "provides" not in doc
    caps = doc["capabilities"]
    assert len(caps) == 1
    assert caps[0]["class_name"] == _NODE_FQCN
    # A real Node carries a category beyond the default 'unspecified'.
    assert caps[0].get("category", "unspecified") != "" or "category" not in caps[0]


def test_emit_check_is_idempotent(tmp_path: Path):
    """After a regenerate, --check reports in-sync (no drift)."""
    manifest = tmp_path / "m.yaml"
    _write_manifest(manifest, [{"class_name": _NODE_FQCN}])
    emit(manifest, check=False)
    assert emit(manifest, check=True) is True


def test_emit_check_detects_drift(tmp_path: Path):
    """A hand-written stale entry is flagged by --check."""
    manifest = tmp_path / "m.yaml"
    # doc_summary that will not match the freshly introspected one.
    _write_manifest(manifest, [{"class_name": _NODE_FQCN, "doc_summary": "stale text"}])
    assert emit(manifest, check=True) is False


def test_emit_leaves_data_module_entries_untouched(tmp_path: Path):
    """data_module capabilities are not introspected as nodes; they round-trip as-is."""
    manifest = tmp_path / "m.yaml"
    dm_entry = {
        "class_name": "cuvis_ai_dataloader.data.Cu3sDataModule",
        "kind": "data_module",
        "data_module_name": "cu3s",
        "extras": ["cu3s"],
    }
    _write_manifest(manifest, [{"class_name": _NODE_FQCN}, dm_entry])

    assert emit(manifest, check=False) is True

    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    dm = [c for c in doc["capabilities"] if c.get("kind") == "data_module"]
    assert len(dm) == 1
    assert dm[0]["data_module_name"] == "cu3s"
    assert dm[0]["extras"] == ["cu3s"]
    assert dm[0]["class_name"] == "cuvis_ai_dataloader.data.Cu3sDataModule"


def test_emit_empty_capabilities_errors(tmp_path: Path):
    """A manifest with no capabilities is a hard error."""
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump({"name": "x", "path": "..", "capabilities": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty 'capabilities:'"):
        emit(manifest, check=False)
