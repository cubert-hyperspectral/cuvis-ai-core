"""Tests for the NodeRegistry plugin system (cuvis-ai-core)."""

from __future__ import annotations

from pathlib import Path

import pytest


from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import (
    GitPluginConfig,
    LocalPluginConfig,
    PluginManifest,
)


def _write_local_plugin(plugin_root: Path, create_pyproject_toml) -> Path:
    """Create a minimal test plugin with PEP 621 compliant structure."""
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "__init__.py").write_text("")
    (plugin_root / "simple_node.py").write_text(
        "from cuvis_ai_core.node.node import Node\n"
        "\n"
        "class SimpleTestNode(Node):\n"
        '    def __init__(self, name="SimpleTestNode"):\n'
        "        super().__init__(name)\n"
        '        self.test_value = "Hello from plugin!"\n'
        "\n"
        "    def forward(self, **inputs):\n"
        '        return {"result": self.test_value}\n'
    )
    # Create PEP 621 compliant pyproject.toml
    create_pyproject_toml(plugin_root)
    return plugin_root


def test_plugin_config_validation():
    git_config = GitPluginConfig(
        repo="git@gitlab.cubert.local:cubert/test-plugin.git",
        tag="v1.2.3",
        provides=["test_plugin.TestNode"],
    )
    assert git_config.repo.endswith("test-plugin.git")
    assert git_config.tag == "v1.2.3"

    local_config = LocalPluginConfig(
        path="/path/to/plugin",
        provides=["local_plugin.LocalNode"],
    )
    assert local_config.path == "/path/to/plugin"

    with pytest.raises(Exception):
        GitPluginConfig(repo="invalid-url", tag="v1.0.0", provides=["test.Node"])

    with pytest.raises(Exception):
        GitPluginConfig(
            repo="git@gitlab.com:user/repo.git",
            tag="v1.0.0",
            provides=["InvalidPath"],
        )

    with pytest.raises(Exception):
        GitPluginConfig(
            repo="git@gitlab.com:user/repo.git",
            tag="v1.0.0",
            provides=[],
        )


def test_plugin_manifest_validation(tmp_path: Path):
    manifest_data = {
        "plugins": {
            "test_git": {
                "repo": "git@gitlab.com:user/repo.git",
                "tag": "v1.0.0",
                "provides": ["repo.TestNode"],
            },
            "test_local": {"path": "../my-plugin", "provides": ["my_plugin.MyNode"]},
        }
    }

    manifest = PluginManifest.from_dict(manifest_data)
    assert len(manifest.plugins) == 2

    invalid_data = {
        "plugins": {
            "invalid-name!": {"path": "../my-plugin", "provides": ["my_plugin.MyNode"]}
        }
    }
    with pytest.raises(Exception):
        PluginManifest.from_dict(invalid_data)

    temp_manifest = tmp_path / "test_manifest.yaml"
    manifest.to_yaml(temp_manifest)
    loaded_manifest = PluginManifest.from_yaml(temp_manifest)
    assert "test_git" in loaded_manifest.plugins


def test_local_plugin_loading(tmp_path: Path, create_plugin_pyproject):
    plugin_root = _write_local_plugin(
        tmp_path / "simple_plugin", create_plugin_pyproject
    )
    registry = NodeRegistry()

    registry.load_plugin(
        "simple_test",
        {"path": str(plugin_root), "provides": ["simple_node.SimpleTestNode"]},
    )

    assert "simple_test" in registry.list_plugins()
    assert "SimpleTestNode" in registry.plugin_registry

    node_class = registry.get("SimpleTestNode")
    instance = node_class()
    assert instance.test_value == "Hello from plugin!"


def test_manifest_relative_path_resolution(tmp_path: Path, create_plugin_pyproject):
    plugins_dir = tmp_path / "plugins"
    _write_local_plugin(plugins_dir / "rel_plugin", create_plugin_pyproject)
    manifest_data = {
        "plugins": {
            "rel_test": {
                "path": "plugins/rel_plugin",
                "provides": ["simple_node.SimpleTestNode"],
            }
        }
    }

    manifest_file = tmp_path / "plugins.yaml"
    PluginManifest.from_dict(manifest_data).to_yaml(manifest_file)

    registry = NodeRegistry()
    loaded_count = registry.load_plugins(manifest_file)
    assert loaded_count == 1
    assert "SimpleTestNode" in registry.plugin_registry


def test_pipeline_integration_with_plugin(tmp_path: Path, create_plugin_pyproject):
    plugin_root = _write_local_plugin(
        tmp_path / "simple_plugin", create_plugin_pyproject
    )
    registry = NodeRegistry()

    registry.load_plugin(
        "simple_test",
        {"path": str(plugin_root), "provides": ["simple_node.SimpleTestNode"]},
    )
    config = {
        "metadata": {"name": "plugin_pipeline"},
        "nodes": [
            {"name": "plugin_node", "class_name": "SimpleTestNode", "hparams": {}}
        ],
        "connections": [],
    }

    builder = PipelineBuilder(node_registry=registry)
    pipeline = builder.build_from_config(config)

    assert pipeline.name == "plugin_pipeline"
    assert any(node.__class__.__name__ == "SimpleTestNode" for node in pipeline.nodes)


# this doesnt work in windows apparently
# @pytest.mark.skipif(git is None, reason="gitpython not installed")
# def test_git_cache_clearing(tmp_path: Path):
#     repo_root, _, commit_2 = _write_git_plugin(tmp_path / "git_plugin")
#     cache_dir = tmp_path / "cache"
#     NodeRegistry.set_cache_dir(cache_dir)

#     NodeRegistry.clear_plugins()
#     try:
#         NodeRegistry.load_plugin(
#             "git_test",
#             {
#                 "repo": f"file://{repo_root}",
#                 "tag": commit_2,
#                 "provides": ["node_impl.GitTestNode"],
#             },
#         )
#         cache_entry = cache_dir / f"git_test@{commit_2}"
#         assert cache_entry.exists()

#         NodeRegistry.clear_plugin_cache("git_test")
#         assert not cache_entry.exists()
#     finally:
#         NodeRegistry.clear_plugins()
