"""Tests for the NodeRegistry plugin system (cuvis-ai-core)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.plugin import (
    GitPluginSource,
    LocalPluginSource,
    parse_plugin_manifest,
    write_plugin_manifest,
)


@pytest.fixture
def provision_local(monkeypatch):
    """Make a bare local plugin dir importable (simulates provisioning).

    Registration is import-only now, so a test plugin must be on ``sys.path``
    before it can be registered. ``monkeypatch.syspath_prepend`` restores
    ``sys.path``; the finalizer drops any modules imported from the provisioned
    roots so module names don't leak across tests.
    """
    roots: list[Path] = []

    def _add(plugin_root: Path) -> None:
        monkeypatch.syspath_prepend(str(plugin_root))
        roots.append(Path(plugin_root).resolve())

    yield _add

    for name in list(sys.modules):
        mod = sys.modules.get(name)
        mod_file = getattr(mod, "__file__", None)
        if mod_file and any(str(r) in str(Path(mod_file).resolve()) for r in roots):
            del sys.modules[name]


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
    git_config = GitPluginSource(
        name="test_plugin",
        repo="git@gitlab.cubert.local:cubert/test-plugin.git",
        tag="v1.2.3",
        capabilities=[{"class_name": "test_plugin.TestNode"}],
    )
    assert git_config.repo.endswith("test-plugin.git")
    assert git_config.tag == "v1.2.3"

    local_config = LocalPluginSource(
        name="local_plugin",
        path="/path/to/plugin",
        capabilities=[{"class_name": "local_plugin.LocalNode"}],
    )
    assert local_config.path == "/path/to/plugin"

    with pytest.raises(Exception):
        GitPluginSource(
            name="test_plugin",
            repo="invalid-url",
            tag="v1.0.0",
            capabilities=[{"class_name": "test.Node"}],
        )

    with pytest.raises(Exception):
        GitPluginSource(
            name="repo",
            repo="git@gitlab.com:user/repo.git",
            tag="v1.0.0",
            capabilities=[{"class_name": "InvalidPath"}],
        )

    with pytest.raises(Exception):
        GitPluginSource(
            name="repo",
            repo="git@gitlab.com:user/repo.git",
            tag="v1.0.0",
            capabilities=[],
        )


def test_plugin_manifest_validation(tmp_path: Path):
    """A bare manifest (one yaml = one plugin) parses, validates ``name``, and
    round-trips through write/load."""
    git_data = {
        "name": "test_git",
        "repo": "git@gitlab.com:user/repo.git",
        "tag": "v1.0.0",
        "capabilities": [{"class_name": "repo.TestNode"}],
    }
    git_manifest = parse_plugin_manifest(git_data)
    assert isinstance(git_manifest, GitPluginSource)
    assert git_manifest.name == "test_git"

    local_data = {
        "name": "test_local",
        "path": "../my-plugin",
        "capabilities": [{"class_name": "my_plugin.MyNode"}],
    }
    local_manifest = parse_plugin_manifest(local_data)
    assert isinstance(local_manifest, LocalPluginSource)

    # A non-identifier plugin name is rejected.
    with pytest.raises(Exception):
        parse_plugin_manifest(
            {
                "name": "invalid-name!",
                "path": "../my-plugin",
                "capabilities": [{"class_name": "my_plugin.MyNode"}],
            }
        )

    # Round-trip a bare manifest through write + load.
    from cuvis_ai_schemas.plugin import load_plugin_manifest

    temp_manifest = tmp_path / "test_manifest.yaml"
    write_plugin_manifest(git_manifest, temp_manifest)
    loaded_manifest = load_plugin_manifest(temp_manifest)
    assert loaded_manifest.name == "test_git"


def test_local_plugin_loading(tmp_path: Path, create_plugin_pyproject, provision_local):
    plugin_root = _write_local_plugin(
        tmp_path / "simple_plugin", create_plugin_pyproject
    )
    provision_local(plugin_root)
    registry = NodeRegistry()

    registry.register_plugin(
        "simple_test",
        {
            "path": str(plugin_root),
            "capabilities": [{"class_name": "simple_node.SimpleTestNode"}],
        },
    )

    assert "simple_test" in registry.list_plugins()
    assert "SimpleTestNode" in registry.loaded_plugin_nodes

    node_class = registry.get("SimpleTestNode")
    instance = node_class()
    assert instance.test_value == "Hello from plugin!"


def test_manifest_relative_path_resolution(
    tmp_path: Path, create_plugin_pyproject, provision_local
):
    plugins_dir = tmp_path / "plugins"
    plugin_root = _write_local_plugin(
        plugins_dir / "rel_plugin", create_plugin_pyproject
    )
    provision_local(plugin_root)
    manifest = LocalPluginSource(
        name="rel_test",
        path="plugins/rel_plugin",
        capabilities=[{"class_name": "simple_node.SimpleTestNode"}],
    )

    manifest_file = tmp_path / "rel_test.yaml"
    write_plugin_manifest(manifest, manifest_file)

    registry = NodeRegistry()
    loaded_count = registry.register_plugins(manifest_file)
    assert loaded_count == 1
    assert "SimpleTestNode" in registry.loaded_plugin_nodes


def test_register_plugins_is_import_only(
    tmp_path: Path, create_plugin_pyproject, provision_local
):
    """Regression: register_plugins imports a preinstalled plugin and never
    clones / installs (the dropped in-process path) or mutates sys.path itself."""
    plugin_root = _write_local_plugin(tmp_path / "io_plugin", create_plugin_pyproject)
    provision_local(plugin_root)
    manifest = LocalPluginSource(
        name="io",
        path=str(plugin_root),
        capabilities=[{"class_name": "simple_node.SimpleTestNode"}],
    )
    manifest_file = tmp_path / "io.yaml"
    write_plugin_manifest(manifest, manifest_file)

    registry = NodeRegistry()
    sys_path_before = list(sys.path)
    registry.register_plugins(manifest_file)

    assert "SimpleTestNode" in registry.loaded_plugin_nodes
    # Import-only: registration adds nothing to sys.path on its own.
    assert sys.path == sys_path_before
    # The clone/install path is gone entirely.
    assert not hasattr(registry, "load_plugin")
    assert not hasattr(NodeRegistry, "_install_plugin_dependencies")


def test_register_plugins_missing_plugin_hints_provision(tmp_path: Path):
    """An un-provisioned plugin raises a guided error pointing at provision."""
    manifest = LocalPluginSource(
        name="absent",
        path=str(tmp_path),
        capabilities=[{"class_name": "definitely_absent_pkg.Node"}],
    )
    manifest_file = tmp_path / "absent.yaml"
    write_plugin_manifest(manifest, manifest_file)

    registry = NodeRegistry()
    with pytest.raises(ModuleNotFoundError, match="provision"):
        registry.register_plugins(manifest_file)


def test_pipeline_integration_with_plugin(
    tmp_path: Path, create_plugin_pyproject, provision_local
):
    plugin_root = _write_local_plugin(
        tmp_path / "simple_plugin", create_plugin_pyproject
    )
    provision_local(plugin_root)
    registry = NodeRegistry()

    registry.register_plugin(
        "simple_test",
        {
            "path": str(plugin_root),
            "capabilities": [{"class_name": "simple_node.SimpleTestNode"}],
        },
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
