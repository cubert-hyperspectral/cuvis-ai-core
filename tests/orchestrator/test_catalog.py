"""Tests for the static node catalog loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuvis_ai_core.orchestrator.catalog import (
    CatalogNodeEntry,
    CatalogPluginEntry,
    CatalogPortSpec,
    load_catalog_entry,
)


def _minimal_metadata_dict(*, plugin_name: str = "my_plugin") -> dict:
    return {
        "schema_version": 1,
        "plugin_name": plugin_name,
        "plugin_version": "0.1.0",
        "nodes": [
            {
                "class_name": "MyNode",
                "full_path": "my_plugin.node.MyNode",
                "category": "transform",
                "tags": ["image"],
                "icon_svg": "<svg></svg>",
                "input_specs": {
                    "x": {
                        "dtype": "float32",
                        "shape": [-1, -1, -1, -1],
                        "optional": False,
                        "description": "input cube",
                    }
                },
                "output_specs": {
                    "y": [
                        {
                            "dtype": "float32",
                            "shape": [-1, -1, -1, -1],
                        }
                    ]
                },
                "doc_summary": "Does a thing.",
            }
        ],
    }


def _write_metadata(tmp_path: Path, payload: dict) -> Path:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path


def test_load_catalog_entry_returns_none_when_metadata_path_missing():
    config = {"path": "/wherever", "provides": ["pkg.mod.Foo"]}
    assert load_catalog_entry("my_plugin", config) is None


def test_load_catalog_entry_parses_full_payload(tmp_path):
    metadata_path = _write_metadata(tmp_path, _minimal_metadata_dict())

    entry = load_catalog_entry(
        "my_plugin",
        {
            "path": "/wherever",
            "provides": ["my_plugin.node.MyNode"],
            "metadata_path": str(metadata_path),
        },
    )

    assert isinstance(entry, CatalogPluginEntry)
    assert entry.plugin_name == "my_plugin"
    assert entry.plugin_version == "0.1.0"
    assert entry.schema_version == 1
    assert len(entry.nodes) == 1
    node = entry.nodes[0]
    assert isinstance(node, CatalogNodeEntry)
    assert node.class_name == "MyNode"
    assert node.full_path == "my_plugin.node.MyNode"
    assert node.category == "transform"
    assert node.tags == ["image"]
    assert node.icon_svg == "<svg></svg>"
    assert node.doc_summary == "Does a thing."

    # Both the dict-form and list-form input get normalised to a list.
    assert isinstance(node.input_specs["x"], list)
    assert len(node.input_specs["x"]) == 1
    in_spec = node.input_specs["x"][0]
    assert isinstance(in_spec, CatalogPortSpec)
    assert in_spec.dtype == "float32"
    assert in_spec.shape == [-1, -1, -1, -1]
    assert in_spec.optional is False
    assert in_spec.description == "input cube"

    assert isinstance(node.output_specs["y"], list)
    assert len(node.output_specs["y"]) == 1


def test_load_catalog_entry_rejects_relative_metadata_path():
    with pytest.raises(ValueError, match="must be absolute"):
        load_catalog_entry(
            "my_plugin",
            {
                "path": "/wherever",
                "provides": ["my_plugin.node.MyNode"],
                "metadata_path": "metadata.json",
            },
        )


def test_load_catalog_entry_raises_filenotfound_when_missing(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_catalog_entry(
            "my_plugin",
            {
                "path": "/wherever",
                "provides": ["my_plugin.node.MyNode"],
                "metadata_path": str(missing),
            },
        )


def test_load_catalog_entry_rejects_unsupported_schema_version(tmp_path):
    payload = _minimal_metadata_dict()
    payload["schema_version"] = 999
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_catalog_entry(
            "my_plugin",
            {
                "provides": ["my_plugin.node.MyNode"],
                "metadata_path": str(metadata_path),
            },
        )


def test_load_catalog_entry_rejects_missing_plugin_name(tmp_path):
    payload = _minimal_metadata_dict()
    del payload["plugin_name"]
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(ValueError, match="plugin_name"):
        load_catalog_entry(
            "my_plugin",
            {
                "provides": ["my_plugin.node.MyNode"],
                "metadata_path": str(metadata_path),
            },
        )


def test_load_catalog_entry_rejects_non_int_shape(tmp_path):
    payload = _minimal_metadata_dict()
    payload["nodes"][0]["input_specs"]["x"]["shape"] = ["dynamic", -1]
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(ValueError, match="shape"):
        load_catalog_entry(
            "my_plugin",
            {
                "provides": ["my_plugin.node.MyNode"],
                "metadata_path": str(metadata_path),
            },
        )


def test_load_catalog_entry_allows_empty_dtype(tmp_path):
    """Generic-tensor ports (torch.Tensor as marker) emit dtype='' from
    the release-time emitter; the loader must accept that and the gRPC
    service maps it to D_TYPE_UNSPECIFIED on the wire."""
    payload = _minimal_metadata_dict()
    del payload["nodes"][0]["input_specs"]["x"]["dtype"]
    metadata_path = _write_metadata(tmp_path, payload)

    entry = load_catalog_entry(
        "my_plugin",
        {
            "provides": ["my_plugin.node.MyNode"],
            "metadata_path": str(metadata_path),
        },
    )
    assert entry is not None
    assert entry.nodes[0].input_specs["x"][0].dtype == ""


def test_plugin_manifest_resolves_relative_metadata_path(tmp_path):
    """Relative metadata_path entries become absolute at YAML load time."""
    from cuvis_ai_core.utils.plugin_config import PluginManifest

    manifest_dir = tmp_path / "configs" / "plugins"
    manifest_dir.mkdir(parents=True)
    metadata_path = manifest_dir / "my_plugin.metadata.json"
    metadata_path.write_text(json.dumps(_minimal_metadata_dict()), encoding="utf-8")

    yaml_path = manifest_dir / "my_plugin.yaml"
    yaml_path.write_text(
        "plugins:\n"
        "  my_plugin:\n"
        "    path: /wherever\n"
        "    provides:\n"
        "      - my_plugin.node.MyNode\n"
        f"    metadata_path: {metadata_path.name}\n",
        encoding="utf-8",
    )

    manifest = PluginManifest.from_yaml(yaml_path)
    resolved = manifest.plugins["my_plugin"].metadata_path
    assert Path(resolved).is_absolute()
    assert Path(resolved) == metadata_path.resolve()
