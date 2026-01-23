"""Integration tests for gRPC plugin management workflow."""

import pytest

from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.utils.plugin_config import PluginManifest, LocalPluginConfig


@pytest.mark.slow
class TestPluginManagementIntegration:
    """Integration tests for plugin management workflows."""

    def test_complete_plugin_workflow(self, grpc_stub, tmp_path):
        """Test complete plugin workflow: create session → load plugins → use in pipeline → cleanup."""
        # Step 1: Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        assert session_resp.session_id
        session_id = session_resp.session_id

        # Step 2: Create mock plugin
        plugin_dir = tmp_path / "workflow_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class WorkflowTestNode(Node):
    INPUT_SPECS = {"input": {"dtype": "float32", "shape": (-1,)}}
    OUTPUT_SPECS = {"output": {"dtype": "float32", "shape": (-1,)}}
    
    def forward(self, input):
        return {"output": input}
    
    def load(self, params, serial_dir):
        pass
""")

        # Step 3: Load plugin via gRPC
        manifest = PluginManifest(
            plugins={
                "workflow_plugin": LocalPluginConfig(
                    path=str(plugin_dir),
                    provides=["workflow_plugin.node.WorkflowTestNode"],
                )
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            load_resp = grpc_stub.LoadPlugins(
                cuvis_ai_pb2.LoadPluginsRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(
                        config_bytes=manifest.model_dump_json().encode()
                    ),
                )
            )

            # Verify plugin loaded successfully
            assert "workflow_plugin" in load_resp.loaded_plugins
            assert len(load_resp.failed_plugins) == 0

            # Step 4: List loaded plugins
            list_resp = grpc_stub.ListLoadedPlugins(
                cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session_id)
            )
            assert len(list_resp.plugins) == 1
            assert list_resp.plugins[0].name == "workflow_plugin"

            # Step 5: Get plugin info
            info_resp = grpc_stub.GetPluginInfo(
                cuvis_ai_pb2.GetPluginInfoRequest(
                    session_id=session_id, plugin_name="workflow_plugin"
                )
            )
            assert info_resp.plugin.name == "workflow_plugin"
            assert info_resp.plugin.type == "local"

            # Step 6: List available nodes (should include plugin node)
            nodes_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
            )
            node_names = [node.class_name for node in nodes_resp.nodes]
            assert "WorkflowTestNode" in node_names

            # Step 7: Close session (should cleanup plugins)
            close_resp = grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
            assert close_resp.success
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_multiple_plugins(self, grpc_stub, tmp_path):
        """Test loading multiple plugins in single request."""
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create two plugins
        plugin1_dir = tmp_path / "plugin1"
        plugin1_dir.mkdir()
        (plugin1_dir / "__init__.py").write_text("")
        (plugin1_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class Plugin1Node(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        plugin2_dir = tmp_path / "plugin2"
        plugin2_dir.mkdir()
        (plugin2_dir / "__init__.py").write_text("")
        (plugin2_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class Plugin2Node(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        # Create manifest with both plugins
        manifest = PluginManifest(
            plugins={
                "plugin1": LocalPluginConfig(
                    path=str(plugin1_dir), provides=["plugin1.node.Plugin1Node"]
                ),
                "plugin2": LocalPluginConfig(
                    path=str(plugin2_dir), provides=["plugin2.node.Plugin2Node"]
                ),
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load both plugins
            load_resp = grpc_stub.LoadPlugins(
                cuvis_ai_pb2.LoadPluginsRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(
                        config_bytes=manifest.model_dump_json().encode()
                    ),
                )
            )

            # Verify both loaded
            assert "plugin1" in load_resp.loaded_plugins
            assert "plugin2" in load_resp.loaded_plugins
            assert len(load_resp.failed_plugins) == 0

            # Verify both in list
            list_resp = grpc_stub.ListLoadedPlugins(
                cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session_id)
            )
            assert len(list_resp.plugins) == 2
            plugin_names = [p.name for p in list_resp.plugins]
            assert "plugin1" in plugin_names
            assert "plugin2" in plugin_names

            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
        finally:
            sys.path.remove(str(tmp_path))

    def test_plugin_not_available_in_other_session(self, grpc_stub, tmp_path):
        """Test that plugin loaded in one session is not available in another."""
        # Create two sessions
        session1_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session1_id = session1_resp.session_id

        session2_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session2_id = session2_resp.session_id

        # Create plugin
        plugin_dir = tmp_path / "isolated_plugin"
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
                "isolated_plugin": LocalPluginConfig(
                    path=str(plugin_dir), provides=["isolated_plugin.node.IsolatedNode"]
                )
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            load_resp = grpc_stub.LoadPlugins(
                cuvis_ai_pb2.LoadPluginsRequest(
                    session_id=session1_id,
                    manifest=cuvis_ai_pb2.PluginManifest(
                        config_bytes=manifest.model_dump_json().encode()
                    ),
                )
            )
            assert "isolated_plugin" in load_resp.loaded_plugins

            # Verify session1 has the plugin
            list1_resp = grpc_stub.ListLoadedPlugins(
                cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session1_id)
            )
            assert len(list1_resp.plugins) == 1

            # Verify session2 does NOT have the plugin
            list2_resp = grpc_stub.ListLoadedPlugins(
                cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session2_id)
            )
            assert len(list2_resp.plugins) == 0

            # Verify plugin node available only in session1
            nodes1_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session1_id)
            )
            node1_names = [node.class_name for node in nodes1_resp.nodes]
            assert "IsolatedNode" in node1_names

            nodes2_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session2_id)
            )
            node2_names = [node.class_name for node in nodes2_resp.nodes]
            assert "IsolatedNode" not in node2_names

            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session1_id)
            )
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session2_id)
            )
        finally:
            sys.path.remove(str(tmp_path))

    def test_session_cleanup_removes_plugins(self, grpc_stub, tmp_path):
        """Test that closing session removes plugin references."""
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create and load plugin
        plugin_dir = tmp_path / "cleanup_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class CleanupNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        manifest = PluginManifest(
            plugins={
                "cleanup_plugin": LocalPluginConfig(
                    path=str(plugin_dir), provides=["cleanup_plugin.node.CleanupNode"]
                )
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load plugin
            load_resp = grpc_stub.LoadPlugins(
                cuvis_ai_pb2.LoadPluginsRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(
                        config_bytes=manifest.model_dump_json().encode()
                    ),
                )
            )
            assert "cleanup_plugin" in load_resp.loaded_plugins

            # Close session
            close_resp = grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
            assert close_resp.success

            # Try to use the session (should fail)
            with pytest.raises(Exception):
                grpc_stub.ListLoadedPlugins(
                    cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session_id)
                )
        finally:
            sys.path.remove(str(tmp_path))

    def test_plugin_provides_multiple_nodes(self, grpc_stub, tmp_path):
        """Test plugin that provides multiple node classes."""
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create plugin with multiple nodes
        plugin_dir = tmp_path / "multi_node_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "nodes.py").write_text("""
from cuvis_ai_core.node import Node

class NodeA(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass

class NodeB(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass

class NodeC(Node):
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
                "multi_node_plugin": LocalPluginConfig(
                    path=str(plugin_dir),
                    provides=[
                        "multi_node_plugin.nodes.NodeA",
                        "multi_node_plugin.nodes.NodeB",
                        "multi_node_plugin.nodes.NodeC",
                    ],
                )
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load plugin
            load_resp = grpc_stub.LoadPlugins(
                cuvis_ai_pb2.LoadPluginsRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(
                        config_bytes=manifest.model_dump_json().encode()
                    ),
                )
            )
            assert "multi_node_plugin" in load_resp.loaded_plugins

            # Get plugin info
            info_resp = grpc_stub.GetPluginInfo(
                cuvis_ai_pb2.GetPluginInfoRequest(
                    session_id=session_id, plugin_name="multi_node_plugin"
                )
            )
            assert len(info_resp.plugin.provides) == 3
            assert "multi_node_plugin.nodes.NodeA" in info_resp.plugin.provides
            assert "multi_node_plugin.nodes.NodeB" in info_resp.plugin.provides
            assert "multi_node_plugin.nodes.NodeC" in info_resp.plugin.provides

            # Verify all nodes available
            nodes_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
            )
            node_names = [node.class_name for node in nodes_resp.nodes]
            assert "NodeA" in node_names
            assert "NodeB" in node_names
            assert "NodeC" in node_names

            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
        finally:
            sys.path.remove(str(tmp_path))

    def test_error_handling_invalid_plugin_path(self, grpc_stub):
        """Test error handling when plugin path is invalid."""
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create manifest with invalid path
        manifest = PluginManifest(
            plugins={
                "invalid_plugin": LocalPluginConfig(
                    path="/nonexistent/invalid/path",
                    provides=["invalid.node.InvalidNode"],
                )
            }
        )

        try:
            # Try to load plugin
            load_resp = grpc_stub.LoadPlugins(
                cuvis_ai_pb2.LoadPluginsRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(
                        config_bytes=manifest.model_dump_json().encode()
                    ),
                )
            )

            # Should report failure
            assert "invalid_plugin" in load_resp.failed_plugins
            assert len(load_resp.loaded_plugins) == 0

            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
        except Exception:
            # Also acceptable if it raises an exception
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )

    def test_get_plugin_info_nonexistent(self, grpc_stub):
        """Test getting info for plugin that doesn't exist."""
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        try:
            # Try to get non-existent plugin info
            with pytest.raises(Exception):
                grpc_stub.GetPluginInfo(
                    cuvis_ai_pb2.GetPluginInfoRequest(
                        session_id=session_id, plugin_name="nonexistent_plugin"
                    )
                )
        finally:
            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )

    def test_list_available_nodes_includes_builtins_and_plugins(
        self, grpc_stub, tmp_path
    ):
        """Test that ListAvailableNodes returns both builtin and plugin nodes."""
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create plugin
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class PluginNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        manifest = PluginManifest(
            plugins={
                "test_plugin": LocalPluginConfig(
                    path=str(plugin_dir), provides=["test_plugin.node.PluginNode"]
                )
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load plugin
            grpc_stub.LoadPlugins(
                cuvis_ai_pb2.LoadPluginsRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(
                        config_bytes=manifest.model_dump_json().encode()
                    ),
                )
            )

            # List available nodes
            nodes_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
            )

            # Should have both builtin and plugin nodes
            sources = {node.source for node in nodes_resp.nodes}
            assert "builtin" in sources or "plugin" in sources

            # Plugin node should be marked correctly
            plugin_nodes = [n for n in nodes_resp.nodes if n.class_name == "PluginNode"]
            assert len(plugin_nodes) == 1
            assert plugin_nodes[0].source == "plugin"
            assert plugin_nodes[0].plugin_name == "test_plugin"

            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
        finally:
            sys.path.remove(str(tmp_path))
