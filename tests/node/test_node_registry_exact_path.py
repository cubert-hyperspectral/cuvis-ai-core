"""A full class path resolves to exactly that class, never to a namesake."""

from __future__ import annotations

import pytest
from cuvis_ai_schemas.plugin import LocalPluginSource, PluginCapabilityEntry
from loguru import logger

from cuvis_ai_core.utils.node_registry import NodeRegistry
from tests.fixtures import shadow_a, shadow_b, shadow_c

pytestmark = pytest.mark.unit

A = "tests.fixtures.shadow_a.Shadowed"
B = "tests.fixtures.shadow_b.Shadowed"
C = "tests.fixtures.shadow_c.Shadowed"


def _plugin(name: str, *class_paths: str) -> LocalPluginSource:
    return LocalPluginSource(
        name=name,
        path=".",
        package_name="fake_pkg",
        capabilities=[PluginCapabilityEntry(class_name=p) for p in class_paths],
    )


@pytest.fixture
def builtin_a():
    """shadow_a.Shadowed as a built-in node for the duration of one test."""
    NodeRegistry.register(shadow_a.Shadowed)
    try:
        yield shadow_a.Shadowed
    finally:
        NodeRegistry._builtin_registry.pop("Shadowed", None)


def test_builtin_path_is_not_shadowed_by_a_plugin_namesake(builtin_a):
    reg = NodeRegistry()
    reg.register_plugins_installed({"shadow": _plugin("shadow", B)})

    assert reg.get(A) is shadow_a.Shadowed
    assert reg.get(B) is shadow_b.Shadowed
    # the simple name keeps its old meaning: a loaded plugin first
    assert reg.get("Shadowed") is shadow_b.Shadowed


def test_plugin_namesakes_stay_reachable_by_full_path():
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        reg = NodeRegistry()
        reg.register_plugins_installed(
            {"first": _plugin("first", B), "second": _plugin("second", C)}
        )
    finally:
        logger.remove(handler_id)

    assert reg.get(B) is shadow_b.Shadowed
    assert reg.get(C) is shadow_c.Shadowed
    assert reg.get("Shadowed") is shadow_c.Shadowed
    assert any("replaces tests.fixtures.shadow_b.Shadowed" in m for m in messages)


def test_unload_forgets_the_plugin_paths():
    reg = NodeRegistry()
    reg.register_plugins_installed({"first": _plugin("first", B)})
    assert B in reg.loaded_plugin_paths

    reg.unload_plugin("first")

    assert B not in reg.loaded_plugin_paths
    assert "Shadowed" not in reg.loaded_plugin_nodes


def test_failed_set_rolls_the_paths_back():
    reg = NodeRegistry()
    reg.register_plugins_installed({"first": _plugin("first", B)})
    before = dict(reg.loaded_plugin_paths)

    with pytest.raises(ModuleNotFoundError):
        reg.register_plugins_installed(
            {
                "second": _plugin("second", C),
                "broken": _plugin("broken", "tests.fixtures.no_such_module.Shadowed"),
            }
        )

    assert reg.loaded_plugin_paths == before


def test_clear_plugins_clears_the_paths():
    reg = NodeRegistry()
    reg.register_plugins_installed({"first": _plugin("first", B)})

    reg.clear_plugins()

    assert reg.loaded_plugin_paths == {}
