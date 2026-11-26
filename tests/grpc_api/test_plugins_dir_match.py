"""LoadPlugin against the launcher's plugins directory (CUVIS_PLUGINS_DIR).

Set, a client may register only a plugin whose manifest sits in that directory, with the same
source (repo and tag, or local path); the server registers its own copy. Unset, LoadPlugin
accepts any well-formed manifest, as before.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc.plugin_service import PLUGINS_DIR_ENV, PluginService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.utils.node_registry import NodeRegistry

GIT = {
    "name": "sam3",
    "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git",
    "tag": "v0.6.1",
    "capabilities": [{"class_name": "cuvis_ai_sam3.node.SAM3TrackerInference"}],
}

LOCAL_CAPABILITIES = [{"class_name": "local_plug.node.Thing"}]


@pytest.fixture
def plugins_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "plugins"
    directory.mkdir()
    (tmp_path / "local_src").mkdir()
    (directory / "sam3.yaml").write_text(yaml.safe_dump(GIT), encoding="utf-8")
    local = {
        "name": "local_plug",
        "path": "../local_src",
        "capabilities": LOCAL_CAPABILITIES,
    }
    (directory / "local_plug.yaml").write_text(yaml.safe_dump(local), encoding="utf-8")
    monkeypatch.setenv(PLUGINS_DIR_ENV, str(directory))
    return directory


@pytest.fixture
def service():
    NodeRegistry.clear()
    manager = SessionManager()
    session_id = manager.create_session()
    yield PluginService(manager), manager, session_id
    manager.close_session(session_id)
    NodeRegistry.clear()


def _load(service, manifest: dict):
    plugin_service, manager, session_id = service
    context = Mock()
    response = plugin_service.load_plugin(
        cuvis_ai_pb2.LoadPluginRequest(
            session_id=session_id,
            manifest=cuvis_ai_pb2.PluginManifest(
                config_bytes=json.dumps(manifest).encode("utf-8")
            ),
        ),
        context,
    )
    return response, context, manager.get_session(session_id).registered_plugins


def _refused(context) -> bool:
    return any(
        call.args == (grpc.StatusCode.PERMISSION_DENIED,)
        for call in context.set_code.call_args_list
    )


def test_an_exact_copy_of_a_git_manifest_is_registered(plugins_dir, service):
    response, context, registered = _load(service, GIT)

    assert response.registered_plugin == "sam3"
    assert not _refused(context)
    assert registered["sam3"]["repo"] == GIT["repo"]


@pytest.mark.parametrize(
    "change",
    [
        {"repo": "https://github.com/attacker/sam3.git"},
        {"tag": "v0.6.2"},
    ],
    ids=["other-repo", "other-tag"],
)
def test_a_git_manifest_with_another_source_is_refused(plugins_dir, service, change):
    response, context, registered = _load(service, {**GIT, **change})

    assert _refused(context)
    assert not response.registered_plugin
    assert "sam3" in response.error
    assert "sam3" not in registered


def test_an_unknown_name_is_refused(plugins_dir, service):
    response, context, registered = _load(service, {**GIT, "name": "not_in_dir"})

    assert _refused(context)
    assert "not_in_dir" not in registered


def test_a_local_manifest_as_cuvisnext_sends_it_is_registered(plugins_dir, service):
    # CuvisNEXT makes a relative path absolute against the manifest folder, with '/', and
    # turns a bare capability name into a {class_name: ...} entry.
    sent_path = (plugins_dir.parent / "local_src").resolve().as_posix()
    response, context, registered = _load(
        service,
        {"name": "local_plug", "path": sent_path, "capabilities": LOCAL_CAPABILITIES},
    )

    assert response.registered_plugin == "local_plug"
    assert not _refused(context)
    assert (
        Path(registered["local_plug"]["path"])
        == (plugins_dir.parent / "local_src").resolve()
    )


def test_a_local_manifest_pointing_elsewhere_is_refused(plugins_dir, service, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    response, context, registered = _load(
        service,
        {
            "name": "local_plug",
            "path": str(elsewhere),
            "capabilities": LOCAL_CAPABILITIES,
        },
    )

    assert _refused(context)
    assert "local_plug" not in registered


def test_a_git_manifest_under_a_local_plugins_name_is_refused(plugins_dir, service):
    response, context, registered = _load(service, {**GIT, "name": "local_plug"})

    assert _refused(context)
    assert "local_plug" not in registered


def test_the_registered_manifest_is_the_one_on_disk(plugins_dir, service):
    sent = {**GIT, "capabilities": [{"class_name": "evil.module.Payload"}]}

    _response, _context, registered = _load(service, sent)

    assert [c["class_name"] for c in registered["sam3"]["capabilities"]] == [
        "cuvis_ai_sam3.node.SAM3TrackerInference"
    ]


def test_a_missing_plugins_dir_refuses_everything(tmp_path, monkeypatch, service):
    monkeypatch.setenv(PLUGINS_DIR_ENV, str(tmp_path / "does-not-exist"))

    response, context, registered = _load(service, GIT)

    assert _refused(context)
    assert "sam3" not in registered


def test_without_a_plugins_dir_any_manifest_is_registered(monkeypatch, service):
    monkeypatch.delenv(PLUGINS_DIR_ENV, raising=False)
    anything = {**GIT, "repo": "https://example.com/some/other.git", "tag": "v9"}

    response, context, registered = _load(service, anything)

    assert response.registered_plugin == "sam3"
    assert registered["sam3"]["repo"] == anything["repo"]


def test_a_manifest_that_vanishes_while_the_folder_is_read_is_the_servers_problem(
    plugins_dir, service, monkeypatch
):
    # The launcher replaced its manifests between the folder listing and the read.
    from cuvis_ai_core.utils import plugin_resolver

    def vanished(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(plugin_resolver, "load_plugin_manifest", vanished)

    response, context, registered = _load(service, GIT)

    context.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)
    assert "sam3" not in registered


@pytest.mark.parametrize(
    "broken",
    [yaml.safe_dump(GIT), "name: [unclosed\n"],
    ids=["duplicate-name", "yaml-syntax-error"],
)
def test_a_broken_plugins_dir_is_the_servers_problem(plugins_dir, service, broken):
    (plugins_dir / "broken.yaml").write_text(broken, encoding="utf-8")

    response, context, registered = _load(service, GIT)

    context.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)
    assert str(plugins_dir) in context.set_details.call_args.args[0]
    assert "sam3" not in registered
