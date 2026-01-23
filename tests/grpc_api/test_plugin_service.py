"""Tests for PluginService gRPC functionality."""

import pytest
from unittest.mock import Mock

from cuvis_ai_core.grpc.plugin_service import PluginService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import PluginManifest, LocalPluginConfig


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

    def test_load_plugins_success(self, tmp_path):
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

        # Create manifest
        manifest = PluginManifest(
            plugins={
                "test_plugin": LocalPluginConfig(
                    path=str(plugin_dir), provides=["test_plugin.node.TestPluginNode"]
                )
            }
        )

        # Create request
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=manifest.model_dump_json().encode()
            ),
        )

        # Load plugins
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            response = self.plugin_service.load_plugins(request, self.mock_context)

            # Verify response
            assert len(response.loaded_plugins) == 1
            assert "test_plugin" in response.loaded_plugins
            assert len(response.failed_plugins) == 0

            # Verify session state updated
            session = self.session_manager.get_session(session_id)
            assert "test_plugin" in session.loaded_plugins

            # Verify plugin registered in session's node registry
            assert "TestPluginNode" in session.node_registry.plugin_registry
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_plugins_partial_failure(self, tmp_path):
        """Test plugin loading with some failures."""
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

        # Create manifest with valid and invalid plugins
        manifest = PluginManifest(
            plugins={
                "valid_plugin": LocalPluginConfig(
                    path=str(valid_plugin_dir), provides=["valid_plugin.node.ValidNode"]
                ),
                "invalid_plugin": LocalPluginConfig(
                    path="/nonexistent/path",
                    provides=["invalid_plugin.node.InvalidNode"],
                ),
            }
        )

        # Create request
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=manifest.model_dump_json().encode()
            ),
        )

        # Load plugins
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            response = self.plugin_service.load_plugins(request, self.mock_context)

            # Verify response
            assert "valid_plugin" in response.loaded_plugins
            assert "invalid_plugin" in response.failed_plugins
            assert len(response.failed_plugins["invalid_plugin"]) > 0
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
        assert len(response.loaded_plugins) == 0
        assert len(response.failed_plugins) == 0

    def test_load_plugins_invalid_session(self):
        """Test loading plugins with non-existent session."""
        # Create manifest
        manifest = PluginManifest(plugins={})

        # Create request with invalid session_id
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id="nonexistent_session",
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=manifest.model_dump_json().encode()
            ),
        )

        # Should not raise - returns empty response with context error
        response = self.plugin_service.load_plugins(request, self.mock_context)
        assert len(response.loaded_plugins) == 0

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

    def test_list_loaded_plugins_with_plugins(self, tmp_path):
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

        # Load plugin
        manifest = PluginManifest(
            plugins={
                "list_loaded_test_plugin": LocalPluginConfig(
                    path=str(plugin_dir),
                    provides=["list_loaded_test_plugin.node.TestNode"],
                )
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=manifest.model_dump_json().encode()
            ),
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
            assert "list_loaded_test_plugin.node.TestNode" in plugin_info.provides
        finally:
            sys.path.remove(str(tmp_path))

    def test_get_plugin_info_success(self, tmp_path):
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

        manifest = PluginManifest(
            plugins={
                "get_info_test_plugin": LocalPluginConfig(
                    path=str(plugin_dir),
                    provides=["get_info_test_plugin.node.TestNode"],
                )
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=manifest.model_dump_json().encode()
            ),
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

    def test_list_available_nodes_with_plugins(self, tmp_path):
        """Test listing nodes includes session plugin nodes."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create and load plugin (use unique plugin name to avoid module caching)
        plugin_dir = tmp_path / "available_nodes_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "nodes.py").write_text("""
from cuvis_ai_core.node import Node

class PluginTestNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        manifest = PluginManifest(
            plugins={
                "available_nodes_plugin": LocalPluginConfig(
                    path=str(plugin_dir),
                    provides=["available_nodes_plugin.nodes.PluginTestNode"],
                )
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=manifest.model_dump_json().encode()
            ),
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            self.plugin_service.load_plugins(load_request, self.mock_context)

            # List available nodes
            list_request = cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
            response = self.plugin_service.list_available_nodes(
                list_request, self.mock_context
            )

            # Verify plugin node is in list
            node_names = [node.class_name for node in response.nodes]
            assert "PluginTestNode" in node_names

            # Find the plugin node in response
            plugin_node = next(
                node for node in response.nodes if node.class_name == "PluginTestNode"
            )
            assert plugin_node.source == "plugin"
            assert plugin_node.plugin_name == "available_nodes_plugin"
            assert (
                plugin_node.full_path == "available_nodes_plugin.nodes.PluginTestNode"
            )
        finally:
            sys.path.remove(str(tmp_path))

    def test_clear_plugin_cache_not_implemented(self):
        """Test clear_plugin_cache functionality."""
        # This test verifies the method exists and returns appropriate response
        request = cuvis_ai_pb2.ClearPluginCacheRequest(plugin_name="")

        # Call clear_plugin_cache
        response = self.plugin_service.clear_plugin_cache(request, self.mock_context)

        # Verify response structure (count may be 0 if no cache exists)
        assert isinstance(response.cleared_count, int)
        assert response.cleared_count >= 0

    def test_session_isolation(self, tmp_path):
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

        # Load plugin only in session1
        manifest = PluginManifest(
            plugins={
                "isolation_test_plugin": LocalPluginConfig(
                    path=str(plugin_dir),
                    provides=["isolation_test_plugin.node.IsolatedNode"],
                )
            }
        )

        load_request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id=session1,
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=manifest.model_dump_json().encode()
            ),
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
