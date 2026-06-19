"""Integration tests for gRPC plugin management workflow."""

import pytest

from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from tests.fixtures.grpc import manifest_config_bytes as _manifest_config_bytes


@pytest.mark.slow
class TestPluginManagementIntegration:
    """Integration tests for plugin management workflows."""

    def test_complete_plugin_workflow(
        self, grpc_stub, tmp_path, create_plugin_pyproject
    ):
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
        create_plugin_pyproject(plugin_dir)

        # Step 3: Load plugin via gRPC
        config_bytes = _manifest_config_bytes(
            {
                "name": "workflow_plugin",
                "path": str(plugin_dir),
                "capabilities": [
                    {"class_name": "workflow_plugin.node.WorkflowTestNode"}
                ],
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            load_resp = grpc_stub.LoadPlugin(
                cuvis_ai_pb2.LoadPluginRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
                )
            )

            # Verify plugin loaded successfully
            assert load_resp.registered_plugin == "workflow_plugin"
            assert load_resp.error == ""

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

            # Step 6: List available nodes — the plugin's declared
            # capabilities (its manifest ``capabilities`` list) surface
            # immediately after LoadPlugin; no plugin import is needed. The proto NodeInfo
            # carries the short class name plus the FQCN as full_path.
            nodes_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
            )
            node_names = [node.class_name for node in nodes_resp.nodes]
            assert "WorkflowTestNode" in node_names
            workflow_node = next(
                node
                for node in nodes_resp.nodes
                if node.class_name == "WorkflowTestNode"
            )
            assert workflow_node.source == "plugin"
            assert workflow_node.full_path == "workflow_plugin.node.WorkflowTestNode"

            # Step 7: Close session (should cleanup plugins)
            close_resp = grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
            assert close_resp.success
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_multiple_plugins(self, grpc_stub, tmp_path, create_plugin_pyproject):
        """Test loading multiple plugins via one LoadPlugin call per manifest."""
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
        create_plugin_pyproject(plugin1_dir)

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
        create_plugin_pyproject(plugin2_dir)

        # One manifest per LoadPlugin call.
        manifests = [
            {
                "name": "plugin1",
                "path": str(plugin1_dir),
                "capabilities": [{"class_name": "plugin1.node.Plugin1Node"}],
            },
            {
                "name": "plugin2",
                "path": str(plugin2_dir),
                "capabilities": [{"class_name": "plugin2.node.Plugin2Node"}],
            },
        ]

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load both plugins, one LoadPlugin call each.
            registered = []
            for manifest in manifests:
                load_resp = grpc_stub.LoadPlugin(
                    cuvis_ai_pb2.LoadPluginRequest(
                        session_id=session_id,
                        manifest=cuvis_ai_pb2.PluginManifest(
                            config_bytes=_manifest_config_bytes(manifest)
                        ),
                    )
                )
                assert load_resp.error == ""
                registered.append(load_resp.registered_plugin)

            # Verify both loaded
            assert "plugin1" in registered
            assert "plugin2" in registered

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

    def test_plugin_not_available_in_other_session(
        self, grpc_stub, tmp_path, create_plugin_pyproject
    ):
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
        create_plugin_pyproject(plugin_dir)

        # Load plugin only in session1
        config_bytes = _manifest_config_bytes(
            {
                "name": "isolated_plugin",
                "path": str(plugin_dir),
                "capabilities": [{"class_name": "isolated_plugin.node.IsolatedNode"}],
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            load_resp = grpc_stub.LoadPlugin(
                cuvis_ai_pb2.LoadPluginRequest(
                    session_id=session1_id,
                    manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
                )
            )
            assert load_resp.registered_plugin == "isolated_plugin"

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

            # Session1's inline catalog surfaces the plugin node in
            # ListAvailableNodes immediately after LoadPlugin (no import).
            # Session2 never registered the plugin, so its catalog — and
            # therefore its ListAvailableNodes response — must not include it.
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

    def test_session_cleanup_removes_plugins(
        self, grpc_stub, tmp_path, create_plugin_pyproject
    ):
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
        create_plugin_pyproject(plugin_dir)

        config_bytes = _manifest_config_bytes(
            {
                "name": "cleanup_plugin",
                "path": str(plugin_dir),
                "capabilities": [{"class_name": "cleanup_plugin.node.CleanupNode"}],
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load plugin
            load_resp = grpc_stub.LoadPlugin(
                cuvis_ai_pb2.LoadPluginRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
                )
            )
            assert load_resp.registered_plugin == "cleanup_plugin"

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

    def test_plugin_provides_multiple_nodes(
        self, grpc_stub, tmp_path, create_plugin_pyproject
    ):
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
        create_plugin_pyproject(plugin_dir)

        # Create config
        config_bytes = _manifest_config_bytes(
            {
                "name": "multi_node_plugin",
                "path": str(plugin_dir),
                "capabilities": [
                    {"class_name": "multi_node_plugin.nodes.NodeA"},
                    {"class_name": "multi_node_plugin.nodes.NodeB"},
                    {"class_name": "multi_node_plugin.nodes.NodeC"},
                ],
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load plugin
            load_resp = grpc_stub.LoadPlugin(
                cuvis_ai_pb2.LoadPluginRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
                )
            )
            assert load_resp.registered_plugin == "multi_node_plugin"

            # Get plugin info
            info_resp = grpc_stub.GetPluginInfo(
                cuvis_ai_pb2.GetPluginInfoRequest(
                    session_id=session_id, plugin_name="multi_node_plugin"
                )
            )
            assert len(info_resp.plugin.capabilities) == 3
            assert "multi_node_plugin.nodes.NodeA" in info_resp.plugin.capabilities
            assert "multi_node_plugin.nodes.NodeB" in info_resp.plugin.capabilities
            assert "multi_node_plugin.nodes.NodeC" in info_resp.plugin.capabilities

            # The plugin's inline catalog provides all three classes, so
            # ListAvailableNodes surfaces every one immediately after
            # LoadPlugin (no import). Proto NodeInfo carries the short
            # class name; the FQCN lives in full_path.
            nodes_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
            )
            node_names = [node.class_name for node in nodes_resp.nodes]
            assert "NodeA" in node_names
            assert "NodeB" in node_names
            assert "NodeC" in node_names
            full_paths = {node.full_path for node in nodes_resp.nodes}
            assert "multi_node_plugin.nodes.NodeA" in full_paths
            assert "multi_node_plugin.nodes.NodeB" in full_paths
            assert "multi_node_plugin.nodes.NodeC" in full_paths

            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
        finally:
            sys.path.remove(str(tmp_path))

    def test_error_handling_invalid_manifest_entry(self, grpc_stub):
        """LoadPlugin reports a manifest that fails schema validation.

        LoadPlugin only runs Pydantic validation on the manifest; a
        nonexistent *path* still validates (it is checked at install time
        in the LoadPipeline path), so the failure surface here is a manifest
        that violates the schema. A malformed ``class_name`` (not a
        fully-qualified dotted path) is rejected in-band via the response
        ``error`` field.
        """
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create config whose capability class_name is malformed (no dot).
        config_bytes = _manifest_config_bytes(
            {
                "name": "invalid_plugin",
                "path": "/nonexistent/invalid/path",
                "capabilities": [{"class_name": "NotAFullyQualifiedName"}],
            }
        )

        try:
            # Try to load plugin
            load_resp = grpc_stub.LoadPlugin(
                cuvis_ai_pb2.LoadPluginRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
                )
            )

            # Should report failure in-band.
            assert load_resp.error != ""
            assert "invalid_plugin" in load_resp.error
            assert load_resp.registered_plugin == ""

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
        self, grpc_stub, tmp_path, create_plugin_pyproject
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
        create_plugin_pyproject(plugin_dir)

        config_bytes = _manifest_config_bytes(
            {
                "name": "test_plugin",
                "path": str(plugin_dir),
                "capabilities": [{"class_name": "test_plugin.node.PluginNode"}],
            }
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load plugin
            grpc_stub.LoadPlugin(
                cuvis_ai_pb2.LoadPluginRequest(
                    session_id=session_id,
                    manifest=cuvis_ai_pb2.PluginManifest(config_bytes=config_bytes),
                )
            )

            # ListAvailableNodes returns both built-ins and the plugin's
            # declared-capability nodes. The plugin entry's ``capabilities``
            # list is the catalog, so LoadPlugin alone is enough for the
            # plugin node to surface — no import, no LoadPipeline needed.
            nodes_resp = grpc_stub.ListAvailableNodes(
                cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
            )

            sources = {node.source for node in nodes_resp.nodes}
            assert "builtin" in sources
            assert "plugin" in sources

            plugin_nodes = [n for n in nodes_resp.nodes if n.class_name == "PluginNode"]
            assert len(plugin_nodes) == 1, (
                "The plugin's inline catalog should surface its node in "
                "ListAvailableNodes immediately after LoadPlugin."
            )
            assert plugin_nodes[0].source == "plugin"
            assert plugin_nodes[0].full_path == "test_plugin.node.PluginNode"

            # Cleanup
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
            )
        finally:
            sys.path.remove(str(tmp_path))
