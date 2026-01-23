"""Integration tests for plugin session isolation."""

import pytest

from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.utils.node_registry import NodeRegistry


@pytest.mark.slow
class TestPluginSessionIsolation:
    """Test that plugin loading maintains session isolation."""

    def setup_method(self):
        """Setup for each test."""
        NodeRegistry.clear()
        self.session_manager = SessionManager()

    def teardown_method(self):
        """Cleanup after each test."""
        # Close all sessions
        for session_id in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(session_id)
        NodeRegistry.clear()

    def test_session_scoped_pipeline_building(self, tmp_path):
        """Test that pipelines built with session_id use session plugins."""
        # Create two sessions
        session1 = self.session_manager.create_session()
        session2 = self.session_manager.create_session()

        # Create two different plugins with same node name
        plugin1_dir = tmp_path / "plugin1"
        plugin1_dir.mkdir()
        (plugin1_dir / "__init__.py").write_text("")
        (plugin1_dir / "node.py").write_text('''
from cuvis_ai_core.node import Node

class CustomNode(Node):
    """Version from plugin1."""
    INPUT_SPECS = {"input": {"dtype": "float32", "shape": (-1,)}}
    OUTPUT_SPECS = {"output": {"dtype": "float32", "shape": (-1,)}}
    
    def forward(self, input):
        return {"output": input * 1.0}
    
    def load(self, params, serial_dir):
        pass
''')

        plugin2_dir = tmp_path / "plugin2"
        plugin2_dir.mkdir()
        (plugin2_dir / "__init__.py").write_text("")
        (plugin2_dir / "node.py").write_text('''
from cuvis_ai_core.node import Node

class CustomNode(Node):
    """Version from plugin2."""
    INPUT_SPECS = {"input": {"dtype": "float32", "shape": (-1,)}}
    OUTPUT_SPECS = {"output": {"dtype": "float32", "shape": (-1,)}}
    
    def forward(self, input):
        return {"output": input * 2.0}
    
    def load(self, params, serial_dir):
        pass
''')

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Get session states
            state1 = self.session_manager.get_session(session1)
            state2 = self.session_manager.get_session(session2)

            # Load plugin1 into session1's registry instance
            config1 = {
                "path": str(plugin1_dir),
                "provides": ["plugin1.node.CustomNode"],
            }
            state1.node_registry.load_plugin("plugin1", config1)
            state1.loaded_plugins["plugin1"] = config1

            # Load plugin2 into session2's registry instance
            config2 = {
                "path": str(plugin2_dir),
                "provides": ["plugin2.node.CustomNode"],
            }
            state2.node_registry.load_plugin("plugin2", config2)
            state2.loaded_plugins["plugin2"] = config2

            # Build pipeline using session1's registry - should use plugin1 version
            builder1 = PipelineBuilder(node_registry=state1.node_registry)
            pipeline_config = {
                "metadata": {"name": "test_pipeline"},
                "nodes": [{"name": "custom_node", "class": "CustomNode"}],
                "connections": [],
            }
            pipeline1 = builder1.build_from_config(pipeline_config)
            # Access node from pipeline
            node1 = [n for n in pipeline1.nodes if n.name == "custom_node"][0]
            assert node1.__class__.__doc__ == "Version from plugin1."

            # Build pipeline using session2's registry - should use plugin2 version
            builder2 = PipelineBuilder(node_registry=state2.node_registry)
            pipeline2 = builder2.build_from_config(pipeline_config)
            # Access node from pipeline
            node2 = [n for n in pipeline2.nodes if n.name == "custom_node"][0]
            assert node2.__class__.__doc__ == "Version from plugin2."

            # Verify they are different node instances
            assert node1.__class__ is not node2.__class__
        finally:
            sys.path.remove(str(tmp_path))

    def test_session_cleanup_releases_plugins(self, tmp_path):
        """Test that closing session releases plugin references."""
        # Create session
        session_id = self.session_manager.create_session()

        # Create and load plugin
        plugin_dir = tmp_path / "cleanup_test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
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
            # Get session state
            session_state = self.session_manager.get_session(session_id)

            # Load plugin into session's registry instance
            config = {
                "path": str(plugin_dir),
                "provides": ["cleanup_test_plugin.node.CleanupTestNode"],
            }
            session_state.node_registry.load_plugin("cleanup_test_plugin", config)
            session_state.loaded_plugins["cleanup_test_plugin"] = config

            # Verify plugin is registered in session's instance
            assert "CleanupTestNode" in session_state.node_registry.plugin_registry

            # Close session
            self.session_manager.close_session(session_id)

            # Verify session removed from manager (GC handles registry cleanup)
            assert session_id not in self.session_manager._sessions
        finally:
            sys.path.remove(str(tmp_path))

    def test_multiple_sessions_concurrent_different_plugins(self, tmp_path):
        """Test multiple sessions with different plugins running concurrently."""
        # Create three sessions
        sessions = [
            self.session_manager.create_session(),
            self.session_manager.create_session(),
            self.session_manager.create_session(),
        ]

        # Create three different plugins
        plugins = []
        for i in range(3):
            plugin_dir = tmp_path / f"plugin_{i}"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("")
            (plugin_dir / "node.py").write_text(f"""
from cuvis_ai_core.node import Node

class PluginNode{i}(Node):
    INPUT_SPECS = {{}}
    OUTPUT_SPECS = {{}}
    
    def forward(self, **inputs):
        return {{"result": {i}}}
    
    def load(self, params, serial_dir):
        pass
""")
            plugins.append(plugin_dir)

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Load different plugin into each session's registry instance
            session_states = []
            for i, session_id in enumerate(sessions):
                session_state = self.session_manager.get_session(session_id)
                session_states.append(session_state)

                config = {
                    "path": str(plugins[i]),
                    "provides": [f"plugin_{i}.node.PluginNode{i}"],
                }
                session_state.node_registry.load_plugin(f"plugin_{i}", config)
                session_state.loaded_plugins[f"plugin_{i}"] = config

            # Verify each session has only its plugin
            for i, state in enumerate(session_states):
                assert f"PluginNode{i}" in state.node_registry.plugin_registry

                # Verify other plugins not in this session
                for j in range(3):
                    if i != j:
                        assert (
                            f"PluginNode{j}" not in state.node_registry.plugin_registry
                        )

            # Close one session and verify others unaffected
            self.session_manager.close_session(sessions[1])

            # Verify session 1 removed, but 0 and 2 still exist
            assert sessions[0] in self.session_manager._sessions
            assert sessions[1] not in self.session_manager._sessions
            assert sessions[2] in self.session_manager._sessions

            # Verify remaining sessions still have their plugins
            assert "PluginNode0" in session_states[0].node_registry.plugin_registry
            assert "PluginNode2" in session_states[2].node_registry.plugin_registry

            # Cleanup remaining sessions
            self.session_manager.close_session(sessions[0])
            self.session_manager.close_session(sessions[2])
        finally:
            sys.path.remove(str(tmp_path))

    def test_session_without_plugins_uses_builtins(self, tmp_path):
        """Test that session without plugins can still use builtin nodes."""
        # Register a builtin node
        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class BuiltinNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        # Create session1 with plugin
        session1 = self.session_manager.create_session()

        # Create session2 without plugin
        session2 = self.session_manager.create_session()

        # Create plugin for session1
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

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Get session states
            state1 = self.session_manager.get_session(session1)
            state2 = self.session_manager.get_session(session2)

            # Load plugin only in session1's registry instance
            config = {
                "path": str(plugin_dir),
                "provides": ["test_plugin.node.PluginNode"],
            }
            state1.node_registry.load_plugin("test_plugin", config)
            state1.loaded_plugins["test_plugin"] = config

            # Both sessions should be able to use builtin node
            node_class1 = state1.node_registry.get("BuiltinNode")
            node_class2 = state2.node_registry.get("BuiltinNode")
            assert node_class1 is node_class2
            assert node_class1 is BuiltinNode

            # Only session1 should have plugin node
            plugin_node = state1.node_registry.get("PluginNode")
            assert plugin_node.__name__ == "PluginNode"

            # Session2 should NOT have plugin node
            with pytest.raises(KeyError):
                state2.node_registry.get("PluginNode")
        finally:
            sys.path.remove(str(tmp_path))

    def test_plugin_override_builtin_in_session_only(self, tmp_path):
        """Test that plugin can override builtin in session without affecting other sessions."""
        # Register builtin node
        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class SharedNode(Node):
            """Builtin version."""

            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {"version": "builtin"}

            def load(self, params, serial_dir):
                pass

        # Create two sessions
        session1 = self.session_manager.create_session()
        session2 = self.session_manager.create_session()

        # Create plugin that overrides SharedNode
        plugin_dir = tmp_path / "override_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text('''
from cuvis_ai_core.node import Node

class SharedNode(Node):
    """Plugin version."""
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {"version": "plugin"}
    
    def load(self, params, serial_dir):
        pass
''')

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Get session states
            state1 = self.session_manager.get_session(session1)
            state2 = self.session_manager.get_session(session2)

            # Load plugin only in session1's registry instance
            config = {
                "path": str(plugin_dir),
                "provides": ["override_plugin.node.SharedNode"],
            }
            state1.node_registry.load_plugin("override_plugin", config)
            state1.loaded_plugins["override_plugin"] = config

            # Session1 should get plugin version (override)
            node_class1 = state1.node_registry.get("SharedNode")
            assert node_class1.__doc__ == "Plugin version."

            # Session2 should get builtin version
            node_class2 = state2.node_registry.get("SharedNode")
            assert node_class2.__doc__ == "Builtin version."
            assert node_class2 is SharedNode

            # Verify they are different classes
            assert node_class1 is not node_class2
        finally:
            sys.path.remove(str(tmp_path))

    def test_session_state_tracks_loaded_plugins(self, tmp_path):
        """Test that SessionState correctly tracks loaded plugins."""
        # Create session
        session_id = self.session_manager.create_session()
        session_state = self.session_manager.get_session(session_id)

        # Create plugins
        plugin1_dir = tmp_path / "tracked_plugin1"
        plugin1_dir.mkdir()
        (plugin1_dir / "__init__.py").write_text("")
        (plugin1_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class TrackedNode1(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {}
    
    def forward(self, **inputs):
        return {}
    
    def load(self, params, serial_dir):
        pass
""")

        plugin2_dir = tmp_path / "tracked_plugin2"
        plugin2_dir.mkdir()
        (plugin2_dir / "__init__.py").write_text("")
        (plugin2_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class TrackedNode2(Node):
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
            # Initially no plugins
            assert len(session_state.loaded_plugins) == 0

            # Load first plugin into session's registry instance
            config1 = {
                "path": str(plugin1_dir),
                "provides": ["tracked_plugin1.node.TrackedNode1"],
            }
            session_state.node_registry.load_plugin("tracked_plugin1", config1)
            session_state.loaded_plugins["tracked_plugin1"] = config1
            assert len(session_state.loaded_plugins) == 1
            assert "tracked_plugin1" in session_state.loaded_plugins

            # Load second plugin into session's registry instance
            config2 = {
                "path": str(plugin2_dir),
                "provides": ["tracked_plugin2.node.TrackedNode2"],
            }
            session_state.node_registry.load_plugin("tracked_plugin2", config2)
            session_state.loaded_plugins["tracked_plugin2"] = config2
            assert len(session_state.loaded_plugins) == 2
            assert "tracked_plugin1" in session_state.loaded_plugins
            assert "tracked_plugin2" in session_state.loaded_plugins

            # Close session should clear loaded_plugins
            self.session_manager.close_session(session_id)

            # Session should be removed
            assert session_id not in self.session_manager._sessions
        finally:
            sys.path.remove(str(tmp_path))

    def test_pipeline_builder_without_session_id_uses_global_registry(self, tmp_path):
        """Test that PipelineBuilder without session_id uses global registry."""
        # Register builtin node
        from cuvis_ai_core.node import Node

        @NodeRegistry.register
        class GlobalNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        # Create session with plugin
        session_id = self.session_manager.create_session()
        session_state = self.session_manager.get_session(session_id)

        # Create plugin
        plugin_dir = tmp_path / "session_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("")
        (plugin_dir / "node.py").write_text("""
from cuvis_ai_core.node import Node

class SessionOnlyNode(Node):
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
            # Load plugin into session's registry instance
            config = {
                "path": str(plugin_dir),
                "provides": ["session_plugin.node.SessionOnlyNode"],
            }
            session_state.node_registry.load_plugin("session_plugin", config)
            session_state.loaded_plugins["session_plugin"] = config

            # PipelineBuilder with session's registry can use plugin node
            builder_with_session = PipelineBuilder(
                node_registry=session_state.node_registry
            )
            pipeline_config = {
                "metadata": {"name": "test_pipeline"},
                "nodes": [{"name": "session_node", "class": "SessionOnlyNode"}],
                "connections": [],
            }
            pipeline = builder_with_session.build_from_config(pipeline_config)
            # Access node from pipeline
            session_node = [n for n in pipeline.nodes if n.name == "session_node"][0]
            assert session_node is not None

            # PipelineBuilder without registry instance CANNOT use plugin node
            builder_without_session = PipelineBuilder()
            with pytest.raises(Exception):  # KeyError or similar
                builder_without_session.build_from_config(pipeline_config)

            # But can use builtin node
            builtin_config = {
                "metadata": {"name": "test_pipeline"},
                "nodes": [{"name": "global_node", "class": "GlobalNode"}],
                "connections": [],
            }
            pipeline2 = builder_without_session.build_from_config(builtin_config)
            # Access node from pipeline
            global_node = [n for n in pipeline2.nodes if n.name == "global_node"][0]
            assert global_node is not None
        finally:
            sys.path.remove(str(tmp_path))
