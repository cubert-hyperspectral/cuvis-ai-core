"""``load_preinstalled_plugins`` registration shape tests."""

from __future__ import annotations

import pytest

from cuvis_ai_core.pipeline.restore_preinstalled import load_preinstalled_plugins
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig


class _DummyClassA:
    pass


class _DummyClassB:
    pass


def test_load_preinstalled_registers_one_entry_per_class():
    registry = NodeRegistry()
    cfg = GitPluginConfig(
        repo="https://example.com/repo.git",
        tag="v1.0",
        provides=[
            {"class_name": f"{__name__}._DummyClassA"},
            {"class_name": f"{__name__}._DummyClassB"},
        ],
    )
    load_preinstalled_plugins(registry, {"fake_plugin": cfg})

    assert "_DummyClassA" in registry.plugin_registry
    assert "_DummyClassB" in registry.plugin_registry
    assert registry.plugin_registry["_DummyClassA"] is _DummyClassA
    assert registry.plugin_registry["_DummyClassB"] is _DummyClassB
    # plugin_configs[name] = config bookkeeping is mirrored from
    # NodeRegistry.load_plugin so downstream lookups by plugin name
    # work the same as in the in-process path.
    assert registry.plugin_configs["fake_plugin"] is cfg


def test_load_preinstalled_local_config_path_resolves(tmp_path):
    registry = NodeRegistry()
    cfg = LocalPluginConfig(
        path=str(tmp_path),
        provides=[{"class_name": f"{__name__}._DummyClassA"}],
    )
    load_preinstalled_plugins(registry, {"local_p": cfg})
    assert registry.plugin_registry["_DummyClassA"] is _DummyClassA


def test_load_preinstalled_no_plugins_is_noop():
    registry = NodeRegistry()
    load_preinstalled_plugins(registry, {})
    assert registry.plugin_registry == {}
    assert registry.plugin_configs == {}


def test_load_preinstalled_unknown_class_raises():
    registry = NodeRegistry()
    cfg = GitPluginConfig(
        repo="https://example.com/repo.git",
        tag="v1.0",
        provides=[{"class_name": "package.that.does.not.exist.Cls"}],
    )
    with pytest.raises((ImportError, AttributeError)):
        load_preinstalled_plugins(registry, {"missing": cfg})
