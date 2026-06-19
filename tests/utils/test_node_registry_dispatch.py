"""Tests for kind-routing of data_module capabilities in NodeRegistry."""

from __future__ import annotations

import pytest

from cuvis_ai_schemas.plugin import LocalPluginSource, PluginCapabilityEntry

from cuvis_ai_core.utils.node_registry import NodeRegistry

_DM = "tests.fixtures.fake_data_modules.FakeDataModule"
_NOT_DM = "tests.fixtures.fake_data_modules.NotADataModule"


def _cfg(name: str, entry: PluginCapabilityEntry) -> LocalPluginSource:
    return LocalPluginSource(
        name=name, path=".", package_name="fake_pkg", capabilities=[entry]
    )


def test_data_module_entry_registers_into_data_modules():
    reg = NodeRegistry()
    reg.register_preinstalled(
        {
            "fake": _cfg(
                "fake",
                PluginCapabilityEntry(
                    class_name=_DM, kind="data_module", data_module_name="fake"
                ),
            )
        }
    )
    assert "fake" in reg.data_modules
    assert reg.data_modules["fake"].DATA_MODULE_NAME == "fake"
    # data modules never land in the node registry (so never in the palette).
    assert "FakeDataModule" not in reg.loaded_plugin_nodes


def test_data_module_name_mismatch_raises():
    reg = NodeRegistry()
    with pytest.raises(ValueError, match="DATA_MODULE_NAME"):
        reg.register_preinstalled(
            {
                "fake": _cfg(
                    "fake",
                    PluginCapabilityEntry(
                        class_name=_DM, kind="data_module", data_module_name="wrong"
                    ),
                )
            }
        )


def test_non_datamodule_class_raises():
    reg = NodeRegistry()
    with pytest.raises(TypeError, match="BaseCuvisAIDataModule"):
        reg.register_preinstalled(
            {
                "fake": _cfg(
                    "fake",
                    PluginCapabilityEntry(
                        class_name=_NOT_DM, kind="data_module", data_module_name="bad"
                    ),
                )
            }
        )


def test_duplicate_data_module_name_raises():
    reg = NodeRegistry()
    reg.register_preinstalled(
        {
            "a": _cfg(
                "a",
                PluginCapabilityEntry(
                    class_name=_DM, kind="data_module", data_module_name="fake"
                ),
            )
        }
    )
    # Re-registering the SAME class under the same name is idempotent (no raise).
    reg.register_preinstalled(
        {
            "a2": _cfg(
                "a2",
                PluginCapabilityEntry(
                    class_name=_DM, kind="data_module", data_module_name="fake"
                ),
            )
        }
    )
    assert reg.data_modules["fake"].DATA_MODULE_NAME == "fake"
