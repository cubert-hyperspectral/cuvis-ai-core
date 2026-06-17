"""Tests for PluginService gRPC functionality."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import numpy as np
import pytest

from cuvis_ai_core.grpc import plugin_service as plugin_service_mod
from cuvis_ai_core.grpc.plugin_service import (
    PluginService,
    _catalog_entry_to_node_info,
    _catalog_port_spec_to_proto,
    _convert_port_spec_to_proto,
)
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.plugin import NodePortSpec, PluginCapabilityEntry


def _manifest_config_bytes(*manifests: dict) -> bytes:
    """Encode bare single-plugin manifest dicts as the LoadPlugins wire payload.

    ``LoadPlugins`` carries a JSON list of single-plugin manifest dumps in
    ``manifest.config_bytes``; each dict is a bare manifest
    (``name`` + source + ``capabilities``).
    """
    return json.dumps(list(manifests)).encode()


@pytest.mark.slow
class TestPluginService:
    """Test PluginService functionality."""

    def setup_method(self):
        """Setup for each test."""
        NodeRegistry.clear()
        self.session_manager = SessionManager()
        self.plugin_service = PluginService(self.session_manager)
        self.mock_context = Mock()

    def teardown_method(self):
        """Cleanup after each test."""
        # Close all sessions
        for session_id in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(session_id)
        NodeRegistry.clear()

    def test_load_plugins_success(self, tmp_path, create_plugin_pyproject):
        """Test successful plugin loading."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create mock plugin
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class TestPluginNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}

    def forward(self, **inputs):
        return {}

    def load(self, params, serial_dir):
        pass
""")
        create_plugin_pyproject(plugin_dir)

        # Create request
        config_bytes = _manifest_config_bytes(
            {
                "name": "test_plugin",
                "path": str(plugin_dir),
                "capabilities": [{"class_name": "test_plugin.node.TestPluginNode"}],
            }
        )
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )

        # Load plugins
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            response = self.plugin_service.load_plugins(request, self.mock_context)

            # Verify response
            assert len(response.registered_plugins) == 1
            assert "test_plugin" in response.registered_plugins
            assert len(response.failed_plugins) == 0

            # Verify session state updated
            session = self.session_manager.get_session(session_id)
            assert "test_plugin" in session.registered_plugins

            # LoadPlugins registers into the catalog only; nothing is
            # imported into loaded_plugin_nodes until LoadPipeline materialises
            # the plugin.
            assert "test_plugin" in session.node_registry.plugin_catalog
            assert "TestPluginNode" not in session.node_registry.loaded_plugin_nodes
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_plugins_partial_failure(self, tmp_path, create_plugin_pyproject):
        """Catalog registration only runs Pydantic validation on each manifest
        entry; install failures move to ``LoadPipeline``. Both a valid manifest
        entry and a nonexistent-path entry pass Pydantic (LocalPluginManifest
        does NOT exists-check the path), so both register successfully. The
        "partial failure" surface this test originally covered (install
        failures) no longer happens here."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create valid plugin
        valid_plugin_dir = tmp_path / "valid_plugin"
        valid_plugin_dir.mkdir()
        (valid_plugin_dir / "__init__.py").write_text("")
        (valid_plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class ValidNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}

    def forward(self, **inputs):
        return {}

    def load(self, params, serial_dir):
        pass
""")
        create_plugin_pyproject(valid_plugin_dir)

        # Create config with two valid Pydantic entries (one of which
        # points at a nonexistent path that would fail at install time).
        config_bytes = _manifest_config_bytes(
            {
                "name": "valid_plugin",
                "path": str(valid_plugin_dir),
                "capabilities": [{"class_name": "valid_plugin.node.ValidNode"}],
            },
            {
                "name": "unreachable_plugin",
                "path": "/nonexistent/path",
                "capabilities": [
                    {"class_name": "unreachable_plugin.node.UnreachableNode"}
                ],
            },
        )

        # Create request
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )

        # Load plugins
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            response = self.plugin_service.load_plugins(request, self.mock_context)

            # Both entries register cleanly because Pydantic accepts a
            # nonexistent path string. The install failure on
            # "unreachable_plugin" will surface later if a pipeline
            # actually references it.
            assert "valid_plugin" in response.registered_plugins
            assert "unreachable_plugin" in response.registered_plugins
            assert len(response.failed_plugins) == 0
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_plugins_invalid_manifest(self):
        """Test loading plugins with invalid manifest JSON."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create request with invalid JSON
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=b"{ invalid json }"),
        )

        # Service handles error gracefully, returns empty response
        response = self.plugin_service.load_plugins(request, self.mock_context)
        assert len(response.registered_plugins) == 0
        assert len(response.failed_plugins) == 0

    def test_load_plugins_invalid_session(self):
        """Test loading plugins with non-existent session."""
        # Create request with invalid session_id. The session guard fires
        # before the manifest is ever parsed, so an empty payload suffices.
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id="nonexistent_session",
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=_manifest_config_bytes()),
        )

        # Should not raise - returns empty response with context error
        response = self.plugin_service.load_plugins(request, self.mock_context)
        assert len(response.registered_plugins) == 0

    def test_list_loaded_plugins_invalid_session(self):
        """Test listing plugins with non-existent session."""
        request = cuvis_ai_pb2.ListLoadedPluginsRequest(
            session_id="nonexistent_session"
        )
        response = self.plugin_service.list_loaded_plugins(request, self.mock_context)
        assert len(response.plugins) == 0

    def test_get_plugin_info_invalid_session(self):
        """Test getting plugin info with non-existent session."""
        request = cuvis_ai_pb2.GetPluginInfoRequest(
            session_id="nonexistent_session", plugin_name="any"
        )
        response = self.plugin_service.get_plugin_info(request, self.mock_context)
        assert response.plugin.name == ""

    def test_list_available_nodes_invalid_session(self):
        """Test listing available nodes with non-existent session."""
        request = cuvis_ai_pb2.ListAvailableNodesRequest(
            session_id="nonexistent_session"
        )
        response = self.plugin_service.list_available_nodes(request, self.mock_context)
        assert len(response.nodes) == 0

    def test_list_loaded_plugins_empty(self):
        """Test listing plugins when none are loaded."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create request
        request = cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session_id)

        # List plugins
        response = self.plugin_service.list_loaded_plugins(request, self.mock_context)

        # Verify empty list
        assert len(response.plugins) == 0

    def test_list_loaded_plugins_with_plugins(self, tmp_path, create_plugin_pyproject):
        """Test listing loaded plugins."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create mock plugin (unique name to avoid module cache conflicts)
        plugin_dir = tmp_path / "list_loaded_test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class TestNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}

    def forward(self, **inputs):
        return {}

    def load(self, params, serial_dir):
        pass
""")
        create_plugin_pyproject(plugin_dir)

        # Load plugin
        config_bytes = _manifest_config_bytes(
            {
                "name": "list_loaded_test_plugin",
                "path": str(plugin_dir),
                "capabilities": [
                    {"class_name": "list_loaded_test_plugin.node.TestNode"}
                ],
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            self.plugin_service.load_plugins(load_request, self.mock_context)

            # List plugins
            list_request = cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session_id)
            response = self.plugin_service.list_loaded_plugins(
                list_request, self.mock_context
            )

            # Verify response
            assert len(response.plugins) == 1
            plugin_info = response.plugins[0]
            assert plugin_info.name == "list_loaded_test_plugin"
            assert plugin_info.type == "local"
            assert plugin_info.source == str(plugin_dir)
            assert "list_loaded_test_plugin.node.TestNode" in plugin_info.capabilities
        finally:
            sys.path.remove(str(tmp_path))

    def test_get_plugin_info_success(self, tmp_path, create_plugin_pyproject):
        """Test getting info for specific plugin."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create and load plugin (unique name to avoid module cache conflicts)
        plugin_dir = tmp_path / "get_info_test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class TestNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}

    def forward(self, **inputs):
        return {}

    def load(self, params, serial_dir):
        pass
""")
        create_plugin_pyproject(plugin_dir)

        config_bytes = _manifest_config_bytes(
            {
                "name": "get_info_test_plugin",
                "path": str(plugin_dir),
                "capabilities": [{"class_name": "get_info_test_plugin.node.TestNode"}],
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            self.plugin_service.load_plugins(load_request, self.mock_context)

            # Get plugin info
            info_request = cuvis_ai_pb2.GetPluginInfoRequest(
                session_id=session_id, plugin_name="get_info_test_plugin"
            )
            response = self.plugin_service.get_plugin_info(
                info_request, self.mock_context
            )

            # Verify response
            assert response.plugin.name == "get_info_test_plugin"
            assert response.plugin.type == "local"
            assert response.plugin.source == str(plugin_dir)
        finally:
            sys.path.remove(str(tmp_path))

    def test_get_plugin_info_not_found(self):
        """Test getting info for non-existent plugin."""
        # Create session
        session_id = self.session_manager.create_session()

        # Try to get non-existent plugin
        request = cuvis_ai_pb2.GetPluginInfoRequest(
            session_id=session_id, plugin_name="nonexistent_plugin"
        )

        # Should not raise - returns empty response with context error
        response = self.plugin_service.get_plugin_info(request, self.mock_context)
        assert response.plugin.name == ""

    def test_list_available_nodes_builtin_only(self):
        """Test listing nodes when only builtins are available."""
        # Create session
        session_id = self.session_manager.create_session()

        # Register a builtin node
        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class BuiltinTestNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        # List available nodes
        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        # Verify builtin node is in list
        node_names = [node.class_name for node in response.nodes]
        assert "BuiltinTestNode" in node_names

        # Find the builtin node in response
        builtin_node = next(
            node for node in response.nodes if node.class_name == "BuiltinTestNode"
        )
        assert builtin_node.source == "builtin"
        assert builtin_node.plugin_name == ""

        # Every NodeInfo carries metadata fields. Default for unannotated
        # nodes is UNSPECIFIED + empty tags + the bundled unspecified.svg
        # (non-empty bytes).
        assert builtin_node.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert list(builtin_node.tags) == []
        assert len(builtin_node.icon_svg) > 0

    def test_list_available_nodes_with_plugins(self, tmp_path, create_plugin_pyproject):
        """``ListAvailableNodes`` populates plugin nodes from each plugin's
        declared ``capabilities``. No plugin module gets imported on the
        parent side — the manifest entry is the only source.
        """
        session_id = self.session_manager.create_session()

        plugin_dir = tmp_path / "available_nodes_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        create_plugin_pyproject(plugin_dir)

        config_bytes = _manifest_config_bytes(
            {
                "name": "available_nodes_plugin",
                "path": str(plugin_dir),
                "capabilities": [
                    {
                        "class_name": "available_nodes_plugin.nodes.PluginTestNode",
                        "category": "unspecified",
                        "tags": [],
                        "icon_svg": "<svg/>",
                        "input_specs": {},
                        "output_specs": {},
                        "doc_summary": "",
                    }
                ],
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )
        self.plugin_service.load_plugins(load_request, self.mock_context)

        list_request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(
            list_request, self.mock_context
        )

        node_names = [node.class_name for node in response.nodes]
        assert "PluginTestNode" in node_names

        plugin_node = next(
            node for node in response.nodes if node.class_name == "PluginTestNode"
        )
        assert plugin_node.source == "plugin"
        assert plugin_node.plugin_name == "available_nodes_plugin"
        assert plugin_node.full_path == "available_nodes_plugin.nodes.PluginTestNode"
        assert plugin_node.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert list(plugin_node.tags) == []
        assert plugin_node.icon_svg == b"<svg/>"

    def test_clear_plugin_cache_not_implemented(self):
        """Test clear_plugin_cache functionality."""
        # This test verifies the method exists and returns appropriate response
        request = cuvis_ai_pb2.ClearPluginCacheRequest(plugin_name="")

        # Call clear_plugin_cache
        response = self.plugin_service.clear_plugin_cache(request, self.mock_context)

        # Verify response structure (count may be 0 if no cache exists)
        assert isinstance(response.cleared_count, int)
        assert response.cleared_count >= 0

    def test_list_available_nodes_populates_explicit_metadata(self):
        """A node with _category and _tags set surfaces those exact values over the wire."""
        session_id = self.session_manager.create_session()

        from cuvis_ai_core.node import Node
        from cuvis_ai_schemas.enums import NodeCategory, NodeTag

        @NodeRegistry.register
        class AnnotatedNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}
            _category = NodeCategory.MODEL
            _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE, NodeTag.TORCH})

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        annotated = next(n for n in response.nodes if n.class_name == "AnnotatedNode")
        assert annotated.category == cuvis_ai_pb2.NODE_CATEGORY_MODEL

        # Tags arrive sorted by proto int (deterministic wire output).
        from cuvis_ai_schemas.grpc.conversions import (
            node_tag_to_proto,
            proto_to_node_tag,
        )

        wire_ints = list(annotated.tags)
        assert wire_ints == sorted(wire_ints), wire_ints
        assert set(wire_ints) == {
            node_tag_to_proto(NodeTag.HYPERSPECTRAL),
            node_tag_to_proto(NodeTag.LEARNABLE),
            node_tag_to_proto(NodeTag.TORCH),
        }
        # Round-trip back to NodeTag members.
        recovered = {proto_to_node_tag(i) for i in wire_ints}
        assert recovered == {NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE, NodeTag.TORCH}
        # Icon falls back to the schemas-default model.svg (non-empty real SVG).
        assert len(annotated.icon_svg) > 0

    def test_list_available_nodes_survives_broken_get_tags(self, monkeypatch):
        """A node whose get_tags() raises must still appear in the response with empty tags."""
        session_id = self.session_manager.create_session()

        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class BrokenTagsNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            @classmethod
            def get_tags(cls):
                raise RuntimeError("intentional failure")

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        broken = next(n for n in response.nodes if n.class_name == "BrokenTagsNode")
        assert list(broken.tags) == []
        # Other fields still resolve to safe defaults; gRPC handler did not crash.
        assert broken.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert len(broken.icon_svg) > 0

    def test_resolve_package_root_handles_inspect_failure(self, monkeypatch):
        """If inspect.getfile raises, _resolve_package_root falls back to None."""
        from cuvis_ai_core.grpc import plugin_service

        def _raise(_x):
            raise TypeError("not a real module")

        monkeypatch.setattr(plugin_service.inspect, "getfile", _raise)

        from cuvis_ai_core.node import Node

        class NoFile(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        assert plugin_service._resolve_package_root(NoFile) is None

    def test_extract_node_metadata_returns_safe_defaults_for_none(self):
        """Lookup-failure path: None class still produces a valid (cat, tags, icon) triple."""
        from cuvis_ai_core.grpc import plugin_service

        category, tags, icon = plugin_service._extract_node_metadata(
            None, class_name="MissingClass"
        )
        assert category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert tags == []
        # Falls back to the bundled unspecified.svg.
        assert len(icon) > 0

    def test_session_isolation(self, tmp_path, create_plugin_pyproject):
        """Test that plugin loading in one session doesn't affect another."""
        # Create two sessions
        session1 = self.session_manager.create_session()
        session2 = self.session_manager.create_session()

        # Create plugin (unique name to avoid module cache conflicts)
        plugin_dir = tmp_path / "isolation_test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class IsolatedNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}

    def forward(self, **inputs):
        return {}

    def load(self, params, serial_dir):
        pass
""")
        create_plugin_pyproject(plugin_dir)

        # Load plugin only in session1
        config_bytes = _manifest_config_bytes(
            {
                "name": "isolation_test_plugin",
                "path": str(plugin_dir),
                "capabilities": [
                    {"class_name": "isolation_test_plugin.node.IsolatedNode"}
                ],
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session1,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            self.plugin_service.load_plugins(load_request, self.mock_context)

            # Verify session1 has the plugin
            list1_request = cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session1)
            response1 = self.plugin_service.list_loaded_plugins(
                list1_request, self.mock_context
            )
            assert len(response1.plugins) == 1

            # Verify session2 does NOT have the plugin
            list2_request = cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session2)
            response2 = self.plugin_service.list_loaded_plugins(
                list2_request, self.mock_context
            )
            assert len(response2.plugins) == 0
        finally:
            sys.path.remove(str(tmp_path))


class TestPluginServiceMetadata:
    """Fast metadata-extraction tests run in default CI (no `slow` marker).

    `TestPluginService` above is marked `slow` because some tests load real
    plugins from disk. The metadata extraction paths are fast — they only
    register in-memory `Node` subclasses and call `list_available_nodes` /
    helper functions directly. Coverage on `plugin_service.py` depends on
    these tests running in default CI.
    """

    def setup_method(self):
        NodeRegistry.clear()
        self.session_manager = SessionManager()
        self.plugin_service = PluginService(self.session_manager)
        self.mock_context = Mock()

    def teardown_method(self):
        for session_id in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(session_id)
        NodeRegistry.clear()

    def test_list_available_nodes_populates_explicit_metadata(self):
        """A node with _category and _tags set surfaces those exact values over the wire."""
        session_id = self.session_manager.create_session()

        from cuvis_ai_core.node import Node
        from cuvis_ai_schemas.enums import NodeCategory, NodeTag

        @NodeRegistry.register
        class AnnotatedNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}
            _category = NodeCategory.MODEL
            _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE, NodeTag.TORCH})

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        annotated = next(n for n in response.nodes if n.class_name == "AnnotatedNode")
        assert annotated.category == cuvis_ai_pb2.NODE_CATEGORY_MODEL

        from cuvis_ai_schemas.grpc.conversions import (
            node_tag_to_proto,
            proto_to_node_tag,
        )

        wire_ints = list(annotated.tags)
        assert wire_ints == sorted(wire_ints), wire_ints
        assert set(wire_ints) == {
            node_tag_to_proto(NodeTag.HYPERSPECTRAL),
            node_tag_to_proto(NodeTag.LEARNABLE),
            node_tag_to_proto(NodeTag.TORCH),
        }
        recovered = {proto_to_node_tag(i) for i in wire_ints}
        assert recovered == {NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE, NodeTag.TORCH}
        assert len(annotated.icon_svg) > 0

    def test_list_available_nodes_default_metadata_for_unannotated_class(self):
        """Unannotated nodes still surface — UNSPECIFIED category, empty tags, fallback icon."""
        session_id = self.session_manager.create_session()

        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class BareNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        bare = next(n for n in response.nodes if n.class_name == "BareNode")
        assert bare.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert list(bare.tags) == []
        assert len(bare.icon_svg) > 0

    def test_list_available_nodes_survives_broken_get_tags(self):
        """A node whose get_tags() raises must still appear with empty tags."""
        session_id = self.session_manager.create_session()

        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class BrokenTagsNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            @classmethod
            def get_tags(cls):
                raise RuntimeError("intentional failure")

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        broken = next(n for n in response.nodes if n.class_name == "BrokenTagsNode")
        assert list(broken.tags) == []
        assert broken.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert len(broken.icon_svg) > 0

    def test_list_available_nodes_survives_broken_get_category(self):
        """A node whose get_category() raises must still appear with UNSPECIFIED category."""
        session_id = self.session_manager.create_session()

        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class BrokenCategoryNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            @classmethod
            def get_category(cls):
                raise RuntimeError("intentional failure")

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        broken = next(n for n in response.nodes if n.class_name == "BrokenCategoryNode")
        assert broken.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert len(broken.icon_svg) > 0

    def test_list_available_nodes_survives_broken_get_icon_name(self):
        """A node whose get_icon_name() raises must still appear with the schemas-default icon."""
        session_id = self.session_manager.create_session()

        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class BrokenIconNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            @classmethod
            def get_icon_name(cls):
                raise RuntimeError("intentional failure")

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
        response = self.plugin_service.list_available_nodes(request, self.mock_context)

        broken = next(n for n in response.nodes if n.class_name == "BrokenIconNode")
        assert len(broken.icon_svg) > 0

    def test_resolve_package_root_handles_inspect_failure(self, monkeypatch):
        """If inspect.getfile raises, _resolve_package_root falls back to None."""
        from cuvis_ai_core.grpc import plugin_service

        def _raise(_x):
            raise TypeError("not a real module")

        monkeypatch.setattr(plugin_service.inspect, "getfile", _raise)

        from cuvis_ai_core.node import Node

        class NoFile(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        assert plugin_service._resolve_package_root(NoFile) is None

    def test_extract_node_metadata_returns_safe_defaults_for_none(self):
        """Lookup-failure path: None class still produces a valid (cat, tags, icon) triple."""
        from cuvis_ai_core.grpc import plugin_service

        category, tags, icon = plugin_service._extract_node_metadata(
            None, class_name="MissingClass"
        )
        assert category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert tags == []
        assert len(icon) > 0

    def test_extract_node_metadata_falls_back_when_icon_resolution_raises(
        self, monkeypatch
    ):
        """If the icon resolver itself raises, _extract_node_metadata logs the
        error and returns ``b''`` rather than propagating the exception."""
        from cuvis_ai_core.grpc import plugin_service
        from cuvis_ai_core.node import Node

        class IconlessNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        def _broken_icon_resolver(**_kwargs):
            raise RuntimeError("icon disk read failed")

        monkeypatch.setattr(plugin_service, "get_node_icon", _broken_icon_resolver)

        category, tags, icon = plugin_service._extract_node_metadata(
            IconlessNode, class_name="IconlessNode"
        )
        assert category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
        assert tags == []
        assert icon == b""

    def test_resolve_package_root_finds_assets_folder(self, tmp_path, monkeypatch):
        """When a node's source file lives under a package that owns
        ``assets/node_icons/``, _resolve_package_root returns that ancestor."""
        from cuvis_ai_core.grpc import plugin_service
        from cuvis_ai_core.node import Node

        package_root = tmp_path / "fake_pkg"
        (package_root / "assets" / "node_icons").mkdir(parents=True)
        fake_source = package_root / "subpkg" / "module.py"
        fake_source.parent.mkdir(parents=True)
        fake_source.write_text("# fake module")

        class FakeNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        monkeypatch.setattr(
            plugin_service.inspect, "getfile", lambda _cls: str(fake_source)
        )

        resolved = plugin_service._resolve_package_root(FakeNode)
        assert resolved == package_root.resolve()


# ---------------------------------------------------------------------------
# Pure proto-conversion helpers (fast, no session)
# ---------------------------------------------------------------------------


def test_convert_port_spec_rejects_non_int_str_dimension():
    spec = SimpleNamespace(
        dtype=np.dtype("float32"),
        shape=(1, None),
        optional=False,
        description="",
        variadic=False,
    )
    with pytest.raises(ValueError, match="invalid shape dimension"):
        _convert_port_spec_to_proto(spec, "p")


def test_catalog_port_spec_empty_dtype_is_unspecified():
    spec = NodePortSpec(
        dtype="", shape=[1, 2], optional=False, description="", variadic=False
    )
    proto = _catalog_port_spec_to_proto("p", spec)
    assert proto.dtype == cuvis_ai_pb2.D_TYPE_UNSPECIFIED
    assert list(proto.shape) == [1, 2]


def test_catalog_port_spec_unsupported_dtype_falls_back_to_unspecified():
    spec = NodePortSpec(
        dtype="definitely-not-a-dtype",
        shape=[1],
        optional=False,
        description="",
        variadic=False,
    )
    proto = _catalog_port_spec_to_proto("p", spec)
    assert proto.dtype == cuvis_ai_pb2.D_TYPE_UNSPECIFIED


def test_catalog_entry_to_node_info_tolerates_bogus_category_and_tags():
    entry = PluginCapabilityEntry(
        class_name="pkg.mod.SomeNode",
        category="bogus-category",
        tags=["not-a-real-tag", "hyperspectral"],
        input_specs={},
        output_specs={},
    )
    info = _catalog_entry_to_node_info(entry, "my_plugin")
    # Unknown category collapses to UNSPECIFIED; unknown tags are dropped.
    assert info.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
    assert info.class_name == "SomeNode"
    assert info.full_path == "pkg.mod.SomeNode"
    assert info.plugin_name == "my_plugin"


# ---------------------------------------------------------------------------
# Fast handler coverage: load_plugins / list_loaded_plugins / get_plugin_info
# guards + the list_available_nodes error-isolation branches. These mirror
# TestPluginService (which is @slow and deselected in the coverage job).
# ---------------------------------------------------------------------------


class TestPluginServiceFastHandlers:
    """Fast (no-disk) handler tests that run in the default coverage job."""

    def setup_method(self):
        NodeRegistry.clear()
        self.session_manager = SessionManager()
        self.plugin_service = PluginService(self.session_manager)
        self.mock_context = Mock()

    def teardown_method(self):
        for session_id in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(session_id)
        NodeRegistry.clear()

    # -- load_plugins -------------------------------------------------------

    def test_load_plugins_unknown_session_returns_empty(self):
        resp = self.plugin_service.load_plugins(
            cuvis_ai_pb2.LoadPluginsRequest(session_id="nope"), self.mock_context
        )
        assert resp == cuvis_ai_pb2.LoadPluginsResponse()

    def test_load_plugins_missing_config_bytes_is_invalid_argument(self):
        sid = self.session_manager.create_session()
        self.plugin_service.load_plugins(
            cuvis_ai_pb2.LoadPluginsRequest(session_id=sid), self.mock_context
        )
        self.mock_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_load_plugins_registers_local_entry(self):
        sid = self.session_manager.create_session()
        config_bytes = _manifest_config_bytes(
            {
                "name": "p",
                "path": ".",
                "capabilities": [
                    {"class_name": "tests.fixtures.mock_nodes.MinMaxNormalizer"}
                ],
            }
        )
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=sid,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )
        resp = self.plugin_service.load_plugins(request, self.mock_context)
        assert "p" in resp.registered_plugins
        assert "p" in self.session_manager.get_session(sid).registered_plugins

    def test_load_plugins_reports_failed_entry(self):
        sid = self.session_manager.create_session()
        session = self.session_manager.get_session(sid)
        session.node_registry.register_catalog_entries = Mock(
            side_effect=Exception("registration boom")
        )
        config_bytes = _manifest_config_bytes(
            {
                "name": "p",
                "path": ".",
                "capabilities": [
                    {"class_name": "tests.fixtures.mock_nodes.MinMaxNormalizer"}
                ],
            }
        )
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=sid,
            manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
        )
        resp = self.plugin_service.load_plugins(request, self.mock_context)
        assert "p" in resp.failed_plugins
        assert resp.registered_plugins == []

    # -- list_loaded_plugins / get_plugin_info ------------------------------

    def test_list_loaded_plugins_unknown_session_returns_empty(self):
        resp = self.plugin_service.list_loaded_plugins(
            cuvis_ai_pb2.ListLoadedPluginsRequest(session_id="nope"), self.mock_context
        )
        assert resp == cuvis_ai_pb2.ListLoadedPluginsResponse()

    def test_list_loaded_plugins_reports_git_and_local(self):
        sid = self.session_manager.create_session()
        session = self.session_manager.get_session(sid)
        session.registered_plugins["g"] = {
            "name": "g",
            "repo": "https://example.com/p.git",
            "tag": "v1.2.3",
            "capabilities": [{"class_name": "a.B"}],
        }
        session.registered_plugins["l"] = {"name": "l", "path": ".", "capabilities": []}
        resp = self.plugin_service.list_loaded_plugins(
            cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=sid), self.mock_context
        )
        by_name = {p.name: p for p in resp.plugins}
        assert by_name["g"].type == "git"
        assert by_name["g"].source == "https://example.com/p.git"
        assert by_name["g"].tag == "v1.2.3"
        assert list(by_name["g"].capabilities) == ["a.B"]
        assert by_name["l"].type == "local"
        assert by_name["l"].source == "."

    def test_get_plugin_info_unknown_session_returns_empty(self):
        resp = self.plugin_service.get_plugin_info(
            cuvis_ai_pb2.GetPluginInfoRequest(session_id="nope", plugin_name="x"),
            self.mock_context,
        )
        assert resp == cuvis_ai_pb2.GetPluginInfoResponse()

    def test_get_plugin_info_not_found_sets_not_found(self):
        sid = self.session_manager.create_session()
        resp = self.plugin_service.get_plugin_info(
            cuvis_ai_pb2.GetPluginInfoRequest(session_id=sid, plugin_name="missing"),
            self.mock_context,
        )
        assert resp == cuvis_ai_pb2.GetPluginInfoResponse()
        self.mock_context.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_get_plugin_info_returns_registered_plugin(self):
        sid = self.session_manager.create_session()
        session = self.session_manager.get_session(sid)
        session.registered_plugins["g"] = {
            "name": "g",
            "repo": "https://example.com/p.git",
            "tag": "v2",
            "capabilities": [{"class_name": "a.B"}, {"class_name": "a.C"}],
        }
        resp = self.plugin_service.get_plugin_info(
            cuvis_ai_pb2.GetPluginInfoRequest(session_id=sid, plugin_name="g"),
            self.mock_context,
        )
        assert resp.plugin.name == "g"
        assert resp.plugin.type == "git"
        assert list(resp.plugin.capabilities) == ["a.B", "a.C"]

    # -- list_available_nodes error isolation -------------------------------

    def _register_one_builtin(self):
        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class SoloNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        return "SoloNode"

    def test_list_available_nodes_survives_builtin_class_resolution_failure(
        self, monkeypatch
    ):
        sid = self.session_manager.create_session()
        name = self._register_one_builtin()
        monkeypatch.setattr(
            NodeRegistry, "get_builtin_class", Mock(side_effect=Exception("boom"))
        )
        resp = self.plugin_service.list_available_nodes(
            cuvis_ai_pb2.ListAvailableNodesRequest(session_id=sid), self.mock_context
        )
        node = next(n for n in resp.nodes if n.class_name == name)
        assert len(node.input_specs) == 0
        assert node.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED

    def test_list_available_nodes_survives_port_spec_extraction_failure(
        self, monkeypatch
    ):
        sid = self.session_manager.create_session()
        name = self._register_one_builtin()
        monkeypatch.setattr(
            self.plugin_service,
            "_extract_port_specs",
            Mock(side_effect=Exception("spec boom")),
        )
        resp = self.plugin_service.list_available_nodes(
            cuvis_ai_pb2.ListAvailableNodesRequest(session_id=sid), self.mock_context
        )
        node = next(n for n in resp.nodes if n.class_name == name)
        assert len(node.input_specs) == 0
        assert len(node.output_specs) == 0

    def test_list_available_nodes_warns_on_plugin_without_nodes(self):
        sid = self.session_manager.create_session()
        session = self.session_manager.get_session(sid)
        # A registered plugin whose only capability is a data module → no
        # node-kind entries survive the palette filter, so it contributes
        # nothing to the node list.
        session.registered_plugins["empty"] = {
            "name": "empty",
            "path": ".",
            "capabilities": [
                {
                    "class_name": "a.b.SomeDataModule",
                    "kind": "data_module",
                    "data_module_name": "some_dm",
                }
            ],
        }
        resp = self.plugin_service.list_available_nodes(
            cuvis_ai_pb2.ListAvailableNodesRequest(session_id=sid), self.mock_context
        )
        assert all(n.plugin_name != "empty" for n in resp.nodes)

    def test_list_available_nodes_skips_plugin_when_catalog_load_raises(
        self, monkeypatch
    ):
        sid = self.session_manager.create_session()
        session = self.session_manager.get_session(sid)
        session.registered_plugins["p"] = {
            "name": "p",
            "path": ".",
            "capabilities": [{"class_name": "a.B"}],
        }
        monkeypatch.setattr(
            plugin_service_mod,
            "load_capabilities",
            Mock(side_effect=ValueError("bad catalog")),
        )
        resp = self.plugin_service.list_available_nodes(
            cuvis_ai_pb2.ListAvailableNodesRequest(session_id=sid), self.mock_context
        )
        assert all(n.plugin_name != "p" for n in resp.nodes)

    def test_list_available_nodes_skips_unconvertible_catalog_entry(self, monkeypatch):
        sid = self.session_manager.create_session()
        session = self.session_manager.get_session(sid)
        session.registered_plugins["p"] = {
            "name": "p",
            "path": ".",
            "capabilities": [{"class_name": "a.B"}],
        }
        monkeypatch.setattr(
            plugin_service_mod,
            "_catalog_entry_to_node_info",
            Mock(side_effect=Exception("convert boom")),
        )
        resp = self.plugin_service.list_available_nodes(
            cuvis_ai_pb2.ListAvailableNodesRequest(session_id=sid), self.mock_context
        )
        # The plugin had one node, but conversion failed → it is skipped.
        assert all(n.plugin_name != "p" for n in resp.nodes)
