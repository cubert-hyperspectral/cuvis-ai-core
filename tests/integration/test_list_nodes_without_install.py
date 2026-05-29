"""ListAvailableNodes must not import plugin modules when the manifest
ships a static ``metadata.json``.

The orchestrator deliberately confines plugin code to child venvs;
``ListAvailableNodes`` is a parent-side RPC, so any plugin module
appearing in the parent's ``sys.modules`` during the call would defeat
the purpose of the catalog.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from cuvis_ai_core.grpc.plugin_service import PluginService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


@pytest.fixture
def session_manager():
    return SessionManager()


def _make_metadata_json(plugin_dir: Path, plugin_name: str, class_name: str) -> Path:
    metadata = {
        "schema_version": 1,
        "plugin_name": plugin_name,
        "plugin_version": "0.1.0",
        "nodes": [
            {
                "class_name": class_name,
                "full_path": f"{plugin_name}.node.{class_name}",
                "category": "transform",
                "tags": ["image"],
                "icon_svg": "",
                "input_specs": {
                    "x": {"dtype": "float32", "shape": [-1, -1]},
                },
                "output_specs": {
                    "y": {"dtype": "float32", "shape": [-1, -1]},
                },
                "doc_summary": "Test node.",
            }
        ],
    }
    metadata_path = plugin_dir / f"{plugin_name}.metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def test_list_available_nodes_does_not_import_plugin(tmp_path, session_manager):
    """A plugin module pointed at by metadata_path must not be imported."""
    plugin_name = "uninstalled_plugin"
    class_name = "UninstalledNode"

    metadata_path = _make_metadata_json(tmp_path, plugin_name, class_name)

    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    session.registered_plugins[plugin_name] = {
        "path": str(tmp_path / plugin_name),
        "provides": [f"{plugin_name}.node.{class_name}"],
        "metadata_path": str(metadata_path),
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
    assert not new_modules, (
        f"ListAvailableNodes imported plugin modules: {new_modules}"
    )

    plugin_nodes = [n for n in response.nodes if n.source == "plugin"]
    assert any(n.class_name == class_name for n in plugin_nodes), (
        "Catalog-provided node missing from response"
    )

    node = next(n for n in plugin_nodes if n.class_name == class_name)
    assert node.plugin_name == plugin_name
    assert node.full_path == f"{plugin_name}.node.{class_name}"
    assert "x" in node.input_specs
    assert "y" in node.output_specs


def test_list_available_nodes_handles_malformed_metadata(tmp_path, session_manager):
    """A malformed metadata.json logs but doesn't break the RPC."""
    plugin_name = "broken_plugin"
    bad_metadata = tmp_path / "broken.metadata.json"
    bad_metadata.write_text("{not valid json", encoding="utf-8")

    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    session.registered_plugins[plugin_name] = {
        "path": str(tmp_path / plugin_name),
        "provides": [f"{plugin_name}.node.Broken"],
        "metadata_path": str(bad_metadata),
    }

    plugin_service = PluginService(session_manager)
    context = Mock()
    response = plugin_service.list_available_nodes(
        cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id),
        context,
    )

    # The handler does not abort the RPC for a single bad plugin —
    # builtins still come back.
    assert response.nodes, "Builtins should still be returned despite bad plugin"
    plugin_nodes = [n for n in response.nodes if n.plugin_name == plugin_name]
    assert plugin_nodes == [], (
        "Broken catalog should contribute no nodes for that plugin"
    )
