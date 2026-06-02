"""ListAvailableNodes must not import plugin modules.

The catalog is carried inline in each manifest entry's ``provides`` list,
so the parent answers ``ListAvailableNodes`` without importing any plugin
module. The orchestrator deliberately confines plugin code to child
venvs; a plugin module appearing in the parent's ``sys.modules`` during
the call would defeat the purpose of the catalog.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest

from cuvis_ai_core.grpc.plugin_service import PluginService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


@pytest.fixture
def session_manager():
    return SessionManager()


def _inline_node(fqcn: str) -> dict:
    return {
        "class_name": fqcn,
        "category": "transform",
        "tags": ["image"],
        "icon_svg": "",
        "input_specs": {"x": {"dtype": "float32", "shape": [-1, -1]}},
        "output_specs": {"y": {"dtype": "float32", "shape": [-1, -1]}},
        "doc_summary": "Test node.",
    }


def test_list_available_nodes_does_not_import_plugin(tmp_path, session_manager):
    """A plugin named in an inline catalog must not be imported."""
    plugin_name = "uninstalled_plugin"
    class_name = "UninstalledNode"
    fqcn = f"{plugin_name}.node.{class_name}"

    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    session.registered_plugins[plugin_name] = {
        "path": str(tmp_path / plugin_name),
        "provides": [_inline_node(fqcn)],
    }

    plugin_modules_before = {
        name for name in sys.modules if name.startswith(plugin_name)
    }

    plugin_service = PluginService(session_manager)
    response = plugin_service.list_available_nodes(
        cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id),
        Mock(),
    )

    plugin_modules_after = {
        name for name in sys.modules if name.startswith(plugin_name)
    }
    new_modules = plugin_modules_after - plugin_modules_before
    assert not new_modules, f"ListAvailableNodes imported plugin modules: {new_modules}"

    plugin_nodes = [n for n in response.nodes if n.source == "plugin"]
    # The proto carries the short display name; full_path keeps the FQCN.
    assert any(n.class_name == class_name for n in plugin_nodes), (
        "Catalog-provided node missing from response"
    )
    node = next(n for n in plugin_nodes if n.class_name == class_name)
    assert node.plugin_name == plugin_name
    assert node.full_path == fqcn
    assert "x" in node.input_specs
    assert "y" in node.output_specs


def test_list_available_nodes_handles_malformed_catalog(tmp_path, session_manager):
    """A malformed inline catalog entry logs but doesn't break the RPC."""
    plugin_name = "broken_plugin"
    bad_node = {
        "class_name": f"{plugin_name}.node.Broken",
        # Non-int shape entries are rejected by CatalogPortSpec validation.
        "input_specs": {"x": {"dtype": "float32", "shape": ["dynamic", -1]}},
    }

    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    session.registered_plugins[plugin_name] = {
        "path": str(tmp_path / plugin_name),
        "provides": [bad_node],
    }

    plugin_service = PluginService(session_manager)
    response = plugin_service.list_available_nodes(
        cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id),
        Mock(),
    )

    # The handler does not abort the RPC for a single bad plugin —
    # builtins still come back.
    assert response.nodes, "Builtins should still be returned despite bad plugin"
    plugin_nodes = [n for n in response.nodes if n.plugin_name == plugin_name]
    assert plugin_nodes == [], "Broken catalog should contribute no nodes for that plugin"
