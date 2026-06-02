"""Tests for the static node catalog loader (inline-provides catalog)."""

from __future__ import annotations

import pytest

from cuvis_ai_core.orchestrator.catalog import (
    CatalogNodeEntry,
    CatalogPluginEntry,
    CatalogPortSpec,
    load_catalog_entry,
)


def _node_dict() -> dict:
    return {
        "class_name": "my_plugin.node.MyNode",
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
            "y": {
                "dtype": "float32",
                "shape": [-1, -1, -1, -1],
            }
        },
        "doc_summary": "Does a thing.",
    }


def _config(*, provides: list | None = None, **extra) -> dict:
    """A manifest plugin entry dict, as stored in registered_plugins."""
    cfg = {"repo": "https://github.com/u/r.git", "tag": "v1.0.0"}
    cfg["provides"] = [_node_dict()] if provides is None else provides
    cfg.update(extra)
    return cfg


def test_load_catalog_entry_returns_none_without_provides():
    assert load_catalog_entry("my_plugin", {"path": "/wherever"}) is None
    assert load_catalog_entry("my_plugin", {"path": "/x", "provides": []}) is None


def test_load_catalog_entry_builds_from_inline_provides():
    entry = load_catalog_entry("my_plugin", _config())

    assert isinstance(entry, CatalogPluginEntry)
    assert entry.plugin_name == "my_plugin"
    assert len(entry.nodes) == 1
    node = entry.nodes[0]
    assert isinstance(node, CatalogNodeEntry)
    # class_name is the FQCN; there is no separate full_path.
    assert node.class_name == "my_plugin.node.MyNode"
    assert not hasattr(node, "full_path")
    assert node.category == "transform"
    assert node.tags == ["image"]
    assert node.icon_svg == "<svg></svg>"
    assert node.doc_summary == "Does a thing."

    # Each port maps to exactly one CatalogPortSpec.
    in_spec = node.input_specs["x"]
    assert isinstance(in_spec, CatalogPortSpec)
    assert in_spec.dtype == "float32"
    assert in_spec.shape == [-1, -1, -1, -1]
    assert in_spec.optional is False
    assert in_spec.description == "input cube"
    assert isinstance(node.output_specs["y"], CatalogPortSpec)


def test_load_catalog_entry_minimal_class_name_only():
    """A bare `class_name` entry yields a palette node with defaults."""
    entry = load_catalog_entry(
        "p", _config(provides=[{"class_name": "p.node.Bare"}])
    )
    assert entry is not None
    node = entry.nodes[0]
    assert node.class_name == "p.node.Bare"
    assert node.category == "unspecified"
    assert node.tags == []
    assert node.input_specs == {}


def test_load_catalog_entry_schema_version_defaults_and_validates():
    # Absent → default supported version.
    entry = load_catalog_entry("p", _config())
    assert entry.schema_version in CatalogPluginEntry.SUPPORTED_VERSIONS

    # Explicit unsupported version → rejected.
    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_catalog_entry("p", _config(schema_version=999))


def test_load_catalog_entry_rejects_node_without_class_name():
    with pytest.raises(ValueError, match="class_name"):
        load_catalog_entry("p", _config(provides=[{"category": "model"}]))


def test_load_catalog_entry_rejects_non_int_shape():
    node = _node_dict()
    node["input_specs"]["x"]["shape"] = ["dynamic", -1]
    with pytest.raises(ValueError, match="shape"):
        load_catalog_entry("p", _config(provides=[node]))


def test_load_catalog_entry_allows_empty_dtype():
    """Generic-tensor ports (torch.Tensor marker) carry dtype=''."""
    node = _node_dict()
    del node["input_specs"]["x"]["dtype"]
    entry = load_catalog_entry("p", _config(provides=[node]))
    assert entry is not None
    assert entry.nodes[0].input_specs["x"].dtype == ""
