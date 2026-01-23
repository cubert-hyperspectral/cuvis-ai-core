"""Tests for NodeRegistry session-scoped plugin functionality.

OBSOLETE: These tests were written for the old session_id-based API.
The new hybrid class/instance design achieves session isolation through
NodeRegistry instances stored in SessionState, eliminating the need for
explicit session_id parameters and cleanup_session() calls.

For session isolation tests, see integration tests in tests/grpc_api/.
"""

import pytest

from cuvis_ai_core.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry


pytestmark = pytest.mark.skip(
    reason="Obsolete: Tests written for old session_id-based API. New API uses instances for isolation."
)


class TestNodeRegistrySession:
    """Test NodeRegistry session-scoped plugin support."""

    def setup_method(self):
        """Clear registry before each test."""
        NodeRegistry.clear()

    def teardown_method(self):
        """Cleanup after each test."""
        import sys

        # Clear module cache to prevent cross-test contamination
        modules_to_remove = [
            m
            for m in sys.modules
            if m.startswith("test_plugin") or m.startswith("plugin")
        ]
        for module in modules_to_remove:
            del sys.modules[module]

        NodeRegistry.clear()
        NodeRegistry._session_plugins.clear()

    def test_load_plugin_session_scoped(self, tmp_path):
        """Test loading plugin into session-scoped registry."""
        # Create mock plugin module
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class SessionTestNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        # Load plugin with session_id
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": ["test_plugin.test_node.SessionTestNode"],
            }

            NodeRegistry.load_plugin("test_plugin", config, session_id="session_1")

            # Verify plugin registered in session registry
            assert "session_1" in NodeRegistry._session_plugins
            assert "SessionTestNode" in NodeRegistry._session_plugins["session_1"]

            # Verify NOT in global plugin registry
            assert "SessionTestNode" not in NodeRegistry._plugin_registry
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_plugin_global_scoped(self, tmp_path):
        """Test loading plugin into global registry (no session_id)."""
        # Create mock plugin module
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class GlobalTestNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        # Load plugin without session_id
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": ["test_plugin.test_node.GlobalTestNode"],
            }

            NodeRegistry.load_plugin("test_plugin", config, session_id=None)

            # Verify plugin registered in global plugin registry
            assert "GlobalTestNode" in NodeRegistry._plugin_registry

            # Verify NOT in session registry
            assert len(NodeRegistry._session_plugins) == 0
        finally:
            sys.path.remove(str(tmp_path))

    def test_get_session_plugin(self, tmp_path):
        """Test getting node from session-scoped registry."""
        # Create and load mock plugin
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class SessionNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": ["test_plugin.test_node.SessionNode"],
            }

            NodeRegistry.load_plugin("test_plugin", config, session_id="session_1")

            # Get node with session_id
            node_class = NodeRegistry.get("SessionNode", session_id="session_1")
            assert node_class.__name__ == "SessionNode"

            # Try to get without session_id - should fail
            with pytest.raises(KeyError, match="not found"):
                NodeRegistry.get("SessionNode", session_id=None)
        finally:
            sys.path.remove(str(tmp_path))

    def test_get_resolution_order(self, tmp_path):
        """Test node resolution order: session → builtin → global → importlib."""

        # Register builtin node
        @NodeRegistry.register
        class BuiltinNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        # Create session plugin with same name
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class BuiltinNode(Node):
    \"\"\"Session version of BuiltinNode.\"\"\"
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {"session": True}
    
    def load(self, params, serial_dir):
        pass
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": ["test_plugin.test_node.BuiltinNode"],
            }

            NodeRegistry.load_plugin("test_plugin", config, session_id="session_1")

            # Get with session_id - should return session version
            session_cls = NodeRegistry.get("BuiltinNode", session_id="session_1")
            assert session_cls.__doc__ == "Session version of BuiltinNode."

            # Get without session_id - should return builtin version
            builtin_cls = NodeRegistry.get("BuiltinNode", session_id=None)
            assert builtin_cls is BuiltinNode
            assert builtin_cls.__doc__ != "Session version of BuiltinNode."
        finally:
            sys.path.remove(str(tmp_path))

    def test_session_isolation(self, tmp_path):
        """Test that sessions have isolated plugin namespaces."""
        # Create two different plugin modules
        plugin1_dir = tmp_path / "plugin1"
        plugin1_dir.mkdir()
        (plugin1_dir / "__init__.py").write_text("")
        (plugin1_dir / "node1.py").write_text("""
from cuvis_ai_core.node import Node

class TestNode(Node):
    \"\"\"Node from plugin1.\"\"\"
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {"plugin": "plugin1"}
    
    def load(self, params, serial_dir):
        pass
""")

        plugin2_dir = tmp_path / "plugin2"
        plugin2_dir.mkdir()
        (plugin2_dir / "__init__.py").write_text("")
        (plugin2_dir / "node2.py").write_text("""
from cuvis_ai_core.node import Node

class TestNode(Node):
    \"\"\"Node from plugin2.\"\"\"
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {"plugin": "plugin2"}
    
    def load(self, params, serial_dir):
        pass
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load plugin1 into session_1
            config1 = {"path": str(plugin1_dir), "provides": ["plugin1.node1.TestNode"]}
            NodeRegistry.load_plugin("plugin1", config1, session_id="session_1")

            # Load plugin2 into session_2
            config2 = {"path": str(plugin2_dir), "provides": ["plugin2.node2.TestNode"]}
            NodeRegistry.load_plugin("plugin2", config2, session_id="session_2")

            # Verify session_1 sees plugin1 version
            node1_cls = NodeRegistry.get("TestNode", session_id="session_1")
            assert node1_cls.__doc__ == "Node from plugin1."

            # Verify session_2 sees plugin2 version
            node2_cls = NodeRegistry.get("TestNode", session_id="session_2")
            assert node2_cls.__doc__ == "Node from plugin2."

            # Verify they are different classes
            assert node1_cls is not node2_cls
        finally:
            sys.path.remove(str(tmp_path))

    def test_cleanup_session(self, tmp_path):
        """Test cleanup_session removes session plugins."""
        # Create and load mock plugin
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class CleanupTestNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": ["test_plugin.test_node.CleanupTestNode"],
            }

            NodeRegistry.load_plugin("test_plugin", config, session_id="session_1")

            # Verify plugin exists
            assert "session_1" in NodeRegistry._session_plugins
            assert "CleanupTestNode" in NodeRegistry._session_plugins["session_1"]

            # Cleanup session
            NodeRegistry.cleanup_session("session_1")

            # Verify plugin removed
            assert "session_1" not in NodeRegistry._session_plugins

            # Verify get fails
            with pytest.raises(KeyError):
                NodeRegistry.get("CleanupTestNode", session_id="session_1")
        finally:
            sys.path.remove(str(tmp_path))

    def test_cleanup_nonexistent_session(self):
        """Test cleanup_session handles non-existent session gracefully."""
        # Should not raise error
        NodeRegistry.cleanup_session("nonexistent_session")

    def test_list_plugin_nodes_session(self, tmp_path):
        """Test list_plugin_nodes returns session plugins."""
        # Create and load mock plugin
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class SessionPluginNode1(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass

class SessionPluginNode2(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": [
                    "test_plugin.test_node.SessionPluginNode1",
                    "test_plugin.test_node.SessionPluginNode2",
                ],
            }

            NodeRegistry.load_plugin("test_plugin", config, session_id="session_1")

            # List session plugin nodes
            nodes = NodeRegistry.list_plugin_nodes(session_id="session_1")
            assert "SessionPluginNode1" in nodes
            assert "SessionPluginNode2" in nodes
            assert nodes == sorted(nodes)  # Should be sorted
        finally:
            sys.path.remove(str(tmp_path))

    def test_list_plugin_nodes_global(self, tmp_path):
        """Test list_plugin_nodes returns global plugins when no session_id."""
        # Create and load mock plugin (use unique name to avoid cache conflicts)
        plugin_dir = tmp_path / "global_list_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class GlobalPluginNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": ["global_list_plugin.test_node.GlobalPluginNode"],
            }

            NodeRegistry.load_plugin("global_list_plugin", config, session_id=None)

            # List global plugin nodes
            nodes = NodeRegistry.list_plugin_nodes(session_id=None)
            assert "GlobalPluginNode" in nodes
        finally:
            sys.path.remove(str(tmp_path))

    def test_multiple_sessions_independent_cleanup(self, tmp_path):
        """Test cleanup of one session doesn't affect other sessions."""
        # Create mock plugin
        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "test_node.py").write_text("""
from cuvis_ai_core.node import Node

class MultiSessionNode(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "path": str(plugin_dir),
                "provides": ["test_plugin.test_node.MultiSessionNode"],
            }

            # Load into both sessions
            NodeRegistry.load_plugin("test_plugin", config, session_id="session_1")
            NodeRegistry.load_plugin("test_plugin", config, session_id="session_2")

            # Verify both sessions have the plugin
            assert "session_1" in NodeRegistry._session_plugins
            assert "session_2" in NodeRegistry._session_plugins

            # Cleanup session_1
            NodeRegistry.cleanup_session("session_1")

            # Verify session_1 cleaned but session_2 remains
            assert "session_1" not in NodeRegistry._session_plugins
            assert "session_2" in NodeRegistry._session_plugins
            assert "MultiSessionNode" in NodeRegistry._session_plugins["session_2"]

            # Verify session_2 can still get the node
            node_class = NodeRegistry.get("MultiSessionNode", session_id="session_2")
            assert node_class.__name__ == "MultiSessionNode"
        finally:
            sys.path.remove(str(tmp_path))
