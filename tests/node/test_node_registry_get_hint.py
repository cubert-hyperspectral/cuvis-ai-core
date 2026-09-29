"""The plugin hint in the not-found error of ``NodeRegistry.get`` keys on the bare name."""

import pytest

from cuvis_ai_core.utils.node_registry import NodeRegistry

PLUGIN_HINT = "appears to be an external plugin node"
CUSTOM_HINT = "For custom nodes, provide full import path"


@pytest.mark.parametrize(
    ("identifier", "hint"),
    [
        ("MyPluginDetector", PLUGIN_HINT),
        ("cuvis_ai_widget", PLUGIN_HINT),
        ("AdaCLIPDetector", CUSTOM_HINT),
        ("NoSuchNode", CUSTOM_HINT),
    ],
)
def test_class_lookup_hint_keys_on_generic_plugin_markers(identifier, hint):
    with pytest.raises(KeyError) as excinfo:
        NodeRegistry.get(identifier)
    assert hint in str(excinfo.value)


def test_instance_lookup_without_loaded_plugins_hints_to_load_them():
    with pytest.raises(KeyError) as excinfo:
        NodeRegistry().get("MyPluginDetector")
    assert "no plugins are loaded" in str(excinfo.value)
