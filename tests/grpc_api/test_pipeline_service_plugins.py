"""Tests for pipeline-driven plugin resolution in LoadPipeline.

The resolver now lives in :mod:`cuvis_ai_core.grpc.orchestrator_bridge`;
``PipelineService.load_pipeline`` is the in-process build-and-attach body
the child runs after :func:`InitializeSession` registers plugins. These
tests drive the parent's ``forward_load_pipeline`` through the in-memory
orchestrator fixture and assert on the resolver-error surface that lands
on the gRPC context.

The orchestrator resolves a pipeline's ``plugins:`` against the session's
client-pushed ``registered_plugins`` (populated by ``LoadPlugin``), not a
server-side directory scan, so each test registers the manifests it needs
up front with :func:`_register_plugin`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import grpc

from cuvis_ai_core.grpc import orchestrator_bridge
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.utils.node_registry import NodeRegistry


def _make_pipeline_config_bytes(
    *,
    node_class_name: str,
    plugins: list | None = None,
) -> bytes:
    """Build a minimal pipeline-config JSON payload for LoadPipeline."""
    body = {
        "metadata": {"name": "Plugins test pipeline"},
        "nodes": [
            {"name": "n0", "class_name": node_class_name, "hparams": {}},
        ],
        "connections": [],
    }
    if plugins is not None:
        body["plugins"] = plugins
    return json.dumps(body).encode("utf-8")


def _make_plugin_files(
    *,
    plugin_root: Path,
    plugin_name: str,
    class_name: str,
    create_plugin_pyproject,
) -> str:
    """Write a minimal local plugin and return its fully-qualified class path."""
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "__init__.py").write_text("")
    (plugin_root / "node.py").write_text(
        "from cuvis_ai_core.node import Node\n\n"
        f"class {class_name}(Node):\n"
        "    INPUT_SPECS = {}\n"
        "    OUTPUT_SPECS = {}\n"
        "    def forward(self, **inputs):\n"
        "        return {}\n"
        "    def load(self, params, serial_dir):\n"
        "        pass\n"
    )
    create_plugin_pyproject(plugin_root)
    return f"{plugin_name}.node.{class_name}"


def _register_plugin(
    session_manager: SessionManager,
    session_id: str,
    *,
    plugin_name: str,
    fqcn: str,
    path: str = ".",
) -> None:
    """Register one plugin into the session catalog, exactly as ``LoadPlugin`` does.

    The orchestrator resolves a pipeline's ``plugins:`` against the session's
    client-pushed ``registered_plugins`` (no directory scan), so a test must
    register the manifest first. Resolution reads only the manifest's
    ``capabilities`` class names; the in-memory spawner ignores ``path``.
    """
    session = session_manager.get_session(session_id)
    session.registered_plugins[plugin_name] = {
        "name": plugin_name,
        "path": path,
        "capabilities": [{"class_name": fqcn}],
    }


class TestLoadPipelinePluginResolution:
    """Resolver semantics through the orchestrator's forward_load_pipeline."""

    def setup_method(self):
        NodeRegistry.clear()
        self.session_manager = SessionManager()
        self.context = Mock()

    def teardown_method(self):
        for session_id in list(self.session_manager._sessions.keys()):
            try:
                self.session_manager.close_session(session_id)
            except Exception:
                pass
        NodeRegistry.clear()

    def test_resolves_only_the_plugins_listed_in_the_pipeline(
        self, tmp_path, create_plugin_pyproject
    ):
        """plugins: lists one of two registered plugins → only that one is resolved."""
        wanted_plugin = tmp_path / "wanted_plugin"
        wanted_fqcn = _make_plugin_files(
            plugin_root=wanted_plugin,
            plugin_name="wanted_plugin",
            class_name="WantedNode",
            create_plugin_pyproject=create_plugin_pyproject,
        )

        session_id = self.session_manager.create_session()
        # The client pushed both plugins to the session catalog via LoadPlugin.
        _register_plugin(
            self.session_manager,
            session_id,
            plugin_name="wanted_plugin",
            fqcn=wanted_fqcn,
            path=wanted_plugin.as_posix(),
        )
        _register_plugin(
            self.session_manager,
            session_id,
            plugin_name="other_plugin",
            fqcn="other_plugin.node.OtherNode",
        )

        request = cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=_make_pipeline_config_bytes(
                    node_class_name=wanted_fqcn,
                    plugins=["wanted_plugin"],
                )
            ),
        )

        sys.path.insert(0, str(tmp_path))
        try:
            response = orchestrator_bridge.forward_load_pipeline(
                self.session_manager, request, self.context
            )
        finally:
            sys.path.remove(str(tmp_path))

        assert response.success
        session = self.session_manager.get_session(session_id)
        # Both stay registered (the client pushed both); only the plugin the
        # pipeline listed is resolved/materialised for this run.
        assert "wanted_plugin" in session.registered_plugins
        assert "other_plugin" in session.registered_plugins
        assert "wanted_plugin" in (session.resolved_plugins or {})
        assert "other_plugin" not in (session.resolved_plugins or {})

    def test_missing_plugins_field_returns_invalid_argument(self):
        """A pipeline that omits ``plugins:`` is rejected with INVALID_ARGUMENT
        and a fix-it message pointing at suggest-plugins-fix."""
        fqcn = "auto_plugin.node.AutoNode"
        session_id = self.session_manager.create_session()
        _register_plugin(
            self.session_manager,
            session_id,
            plugin_name="auto_plugin",
            fqcn=fqcn,
        )

        request = cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                # No plugins field — the resolver hard-fails.
                config_bytes=_make_pipeline_config_bytes(node_class_name=fqcn),
            ),
        )

        response = orchestrator_bridge.forward_load_pipeline(
            self.session_manager, request, self.context
        )

        assert not response.success
        self.context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
        details_call = self.context.set_details.call_args[0][0]
        assert "mandatory 'plugins:' field" in details_call
        assert "suggest-plugins-fix" in details_call

    def test_missing_class_in_declared_set_raises_invalid_argument(self):
        """plugins: declares a plugin that does not provide the pipeline's class_name."""
        session_id = self.session_manager.create_session()
        _register_plugin(
            self.session_manager,
            session_id,
            plugin_name="p",
            fqcn="p.node.PNode",
        )

        request = cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=_make_pipeline_config_bytes(
                    node_class_name="not.in.any.plugin.Class",
                    plugins=["p"],
                ),
            ),
        )

        response = orchestrator_bridge.forward_load_pipeline(
            self.session_manager, request, self.context
        )

        assert not response.success
        self.context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
