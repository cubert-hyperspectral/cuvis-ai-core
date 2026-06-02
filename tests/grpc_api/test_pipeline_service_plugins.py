"""Tests for pipeline-driven plugin resolution in LoadPipeline.

The resolver now lives in :mod:`cuvis_ai_core.grpc.orchestrator_bridge`;
``PipelineService.load_pipeline`` is the in-process build-and-attach body
the child runs after :func:`InitializeSession` registers plugins. These
tests drive the parent's ``forward_load_pipeline`` through the in-memory
orchestrator fixture and assert on the resolver-error surface that lands
on the gRPC context.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import grpc
import pytest

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


def _write_catalog_manifest(plugins_dir: Path, plugin_name: str, plugin_dir: Path, fqcn: str) -> None:
    """Write a single-plugin manifest YAML into the catalog directory."""
    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / f"{plugin_name}.yaml").write_text(
        f"plugins:\n"
        f"  {plugin_name}:\n"
        f"    path: {plugin_dir.as_posix()!r}\n"
        f"    provides:\n"
        f"      - class_name: {fqcn}\n"
    )


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

    def _session_with_search_path(self, search_path: Path) -> str:
        session_id = self.session_manager.create_session()
        self.session_manager.set_search_paths(
            session_id,
            [str(search_path)],
            append=False,
        )
        return session_id

    def test_declared_plugins_materialise_only_what_is_listed(
        self, tmp_path, create_plugin_pyproject
    ):
        """plugins: declares one plugin → only that one lands on session.registered_plugins."""
        wanted_plugin = tmp_path / "wanted_plugin"
        wanted_fqcn = _make_plugin_files(
            plugin_root=wanted_plugin,
            plugin_name="wanted_plugin",
            class_name="WantedNode",
            create_plugin_pyproject=create_plugin_pyproject,
        )
        other_plugin = tmp_path / "other_plugin"
        _make_plugin_files(
            plugin_root=other_plugin,
            plugin_name="other_plugin",
            class_name="OtherNode",
            create_plugin_pyproject=create_plugin_pyproject,
        )
        plugins_dir = tmp_path / "configs" / "plugins"
        _write_catalog_manifest(plugins_dir, "wanted_plugin", wanted_plugin, wanted_fqcn)
        _write_catalog_manifest(
            plugins_dir,
            "other_plugin",
            other_plugin,
            "other_plugin.node.OtherNode",
        )

        session_id = self._session_with_search_path(tmp_path / "configs")
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
        # Parent-side mirror reflects exactly what the orchestrator resolved.
        assert "wanted_plugin" in session.registered_plugins
        assert "other_plugin" not in session.registered_plugins
        # The orchestrator stashed the resolved plugin dict alongside the
        # child handle so the same set is forwarded to InitializeSession.
        assert "wanted_plugin" in (session.resolved_plugins or {})
        assert "other_plugin" not in (session.resolved_plugins or {})

    def test_missing_plugins_field_returns_invalid_argument(
        self, tmp_path, create_plugin_pyproject
    ):
        """Pipelines without ``plugins:`` are rejected with INVALID_ARGUMENT
        and a fix-it message pointing at suggest-plugins-fix."""
        plugin_dir = tmp_path / "auto_plugin"
        fqcn = _make_plugin_files(
            plugin_root=plugin_dir,
            plugin_name="auto_plugin",
            class_name="AutoNode",
            create_plugin_pyproject=create_plugin_pyproject,
        )
        plugins_dir = tmp_path / "configs" / "plugins"
        _write_catalog_manifest(plugins_dir, "auto_plugin", plugin_dir, fqcn)

        session_id = self._session_with_search_path(tmp_path / "configs")

        request = cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                # No plugins field — resolver hard-fails.
                config_bytes=_make_pipeline_config_bytes(node_class_name=fqcn),
            ),
        )

        sys.path.insert(0, str(tmp_path))
        try:
            response = orchestrator_bridge.forward_load_pipeline(
                self.session_manager, request, self.context
            )
        finally:
            sys.path.remove(str(tmp_path))

        assert not response.success
        self.context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
        details_call = self.context.set_details.call_args[0][0]
        assert "mandatory 'plugins:' field" in details_call
        assert "suggest-plugins-fix" in details_call

    def test_missing_class_in_declared_set_raises_invalid_argument(
        self, tmp_path, create_plugin_pyproject
    ):
        """plugins: declares a plugin that does not provide the pipeline's class_name."""
        plugin_dir = tmp_path / "p"
        fqcn = _make_plugin_files(
            plugin_root=plugin_dir,
            plugin_name="p",
            class_name="PNode",
            create_plugin_pyproject=create_plugin_pyproject,
        )
        plugins_dir = tmp_path / "configs" / "plugins"
        _write_catalog_manifest(plugins_dir, "p", plugin_dir, fqcn)

        session_id = self._session_with_search_path(tmp_path / "configs")

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
