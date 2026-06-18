"""Tests for the server-side plugin-capabilities loader.

A plugin declares its capabilities in its bare manifest (``name`` + source +
``capabilities`` list). ``load_capabilities`` re-validates a stored manifest
dump and strips the install source down to the palette-facing
:class:`PluginCapabilities`.
"""

from __future__ import annotations

import pytest

from cuvis_ai_core.orchestrator.plugin_capabilities import (
    NodePortSpec,
    PluginCapabilities,
    PluginCapabilityEntry,
    load_capabilities,
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


def _manifest(*, capabilities: list | None = None, **extra) -> dict:
    """A bare git plugin-manifest dump, as stored in registered_plugins."""
    cfg = {
        "name": "my_plugin",
        "repo": "https://github.com/u/r.git",
        "tag": "v1.0.0",
    }
    cfg["capabilities"] = [_node_dict()] if capabilities is None else capabilities
    cfg.update(extra)
    return cfg


def test_load_capabilities_builds_from_manifest_dump():
    caps = load_capabilities(_manifest())

    assert isinstance(caps, PluginCapabilities)
    assert caps.plugin_name == "my_plugin"
    assert len(caps.capabilities) == 1
    entry = caps.capabilities[0]
    assert isinstance(entry, PluginCapabilityEntry)
    # class_name is the FQCN; there is no separate full_path.
    assert entry.class_name == "my_plugin.node.MyNode"
    assert not hasattr(entry, "full_path")
    assert entry.category == "transform"
    assert entry.tags == ["image"]
    assert entry.icon_svg == "<svg></svg>"
    assert entry.doc_summary == "Does a thing."

    # Each port maps to exactly one NodePortSpec.
    in_spec = entry.input_specs["x"]
    assert isinstance(in_spec, NodePortSpec)
    assert in_spec.dtype == "float32"
    assert in_spec.shape == [-1, -1, -1, -1]
    assert in_spec.optional is False
    assert in_spec.description == "input cube"
    assert isinstance(entry.output_specs["y"], NodePortSpec)


def test_load_capabilities_minimal_class_name_only():
    """A bare `class_name` entry yields a palette node with defaults."""
    caps = load_capabilities(_manifest(capabilities=[{"class_name": "p.node.Bare"}]))
    assert caps is not None
    entry = caps.capabilities[0]
    assert entry.class_name == "p.node.Bare"
    assert entry.category == "unspecified"
    assert entry.tags == []
    assert entry.input_specs == {}


def test_from_manifest_builds_capabilities_from_manifest_object():
    """``PluginCapabilities.from_manifest`` takes a parsed manifest object."""
    from cuvis_ai_schemas.plugin import parse_plugin_manifest

    manifest = parse_plugin_manifest(_manifest())
    caps = PluginCapabilities.from_manifest(manifest)
    assert isinstance(caps, PluginCapabilities)
    assert caps.plugin_name == "my_plugin"
    assert caps.capabilities[0].class_name == "my_plugin.node.MyNode"


def test_load_capabilities_rejects_node_without_class_name():
    with pytest.raises(ValueError, match="class_name"):
        load_capabilities(_manifest(capabilities=[{"category": "model"}]))


def test_load_capabilities_rejects_non_int_shape():
    node = _node_dict()
    node["input_specs"]["x"]["shape"] = ["dynamic", -1]
    with pytest.raises(ValueError, match="shape"):
        load_capabilities(_manifest(capabilities=[node]))


def test_load_capabilities_allows_empty_dtype():
    """Generic-tensor ports (torch.Tensor marker) carry dtype=''."""
    node = _node_dict()
    del node["input_specs"]["x"]["dtype"]
    caps = load_capabilities(_manifest(capabilities=[node]))
    assert caps is not None
    assert caps.capabilities[0].input_specs["x"].dtype == ""
