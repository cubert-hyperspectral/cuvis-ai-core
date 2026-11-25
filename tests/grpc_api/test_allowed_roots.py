"""Path confinement under CUVIS_ALLOWED_ROOTS.

Unset, every path behaves as before. Set, a path that reaches the filesystem has to resolve
inside one of the roots, and remote forms (UNC, device paths) are refused before any
filesystem call, so Windows never opens an SMB session for them.
"""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
from cuvis_ai_core.utils import config_helpers
from cuvis_ai_core.utils.path_confinement import (
    ALLOWED_ROOTS_ENV,
    PathNotAllowedError,
    allowed_roots,
    confine_path,
    confine_resolved,
)

REMOTE_FORMS = [
    r"\\fileserver\share\weights.pt",
    "//fileserver/share/weights.pt",
    r"\\?\C:\weights.pt",
    r"\\?\UNC\fileserver\share\weights.pt",
    r"\\.\pipe\weights",
]


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """One allowed root; ``tmp_path / "outside"`` lies outside it."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "outside").mkdir()
    monkeypatch.setenv(ALLOWED_ROOTS_ENV, str(allowed))
    return allowed


@pytest.fixture
def no_remote_filesystem_calls(monkeypatch):
    """Fail the test when anything stats or resolves a remote-looking path."""

    def guard(name, original):
        def wrapped(self, *args, **kwargs):
            if str(self).startswith(("\\\\", "//")):
                raise AssertionError(f"{name} called on {self}")
            return original(self, *args, **kwargs)

        return wrapped

    for name in ("resolve", "exists", "stat", "is_dir", "is_file"):
        monkeypatch.setattr(
            pathlib.Path, name, guard(name, getattr(pathlib.Path, name))
        )


# ---------------------------------------------------------------------------
# allowed_roots / confine_path / confine_resolved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "  ", os.pathsep])
def test_no_roots_when_the_variable_is_unset_or_empty(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ALLOWED_ROOTS_ENV, raising=False)
    else:
        monkeypatch.setenv(ALLOWED_ROOTS_ENV, value)

    assert allowed_roots() is None


def test_roots_are_split_on_the_path_separator(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv(ALLOWED_ROOTS_ENV, f"{first}{os.pathsep} {second} ")

    assert allowed_roots() == [first.resolve(), second.resolve()]


def test_without_roots_every_path_passes_unchanged(
    monkeypatch, no_remote_filesystem_calls
):
    monkeypatch.delenv(ALLOWED_ROOTS_ENV, raising=False)

    for raw in [*REMOTE_FORMS, "../up/there.pt", "/anywhere/at/all.pt"]:
        assert confine_path(raw, field="weights_path") == raw


def test_an_absolute_path_inside_a_root_comes_back_resolved(root):
    inside = root / "runs" / "model.pt"

    assert confine_path(str(inside), field="weights_path") == str(inside.resolve())


def test_an_absolute_path_outside_the_roots_is_refused(root, tmp_path):
    with pytest.raises(PathNotAllowedError, match="weights_path"):
        confine_path(str(tmp_path / "outside" / "model.pt"), field="weights_path")


def test_a_sibling_whose_name_starts_like_the_root_is_outside(root):
    sibling = root.parent / f"{root.name}-other" / "model.pt"

    with pytest.raises(PathNotAllowedError):
        confine_path(str(sibling), field="weights_path")


@pytest.mark.parametrize("raw", REMOTE_FORMS)
def test_remote_forms_are_refused_before_any_filesystem_call(
    root, no_remote_filesystem_calls, raw
):
    with pytest.raises(PathNotAllowedError):
        confine_path(raw, field="weights_path")


@pytest.mark.parametrize(
    "raw", ["../model.pt", "runs/../../model.pt", r"runs\..\..\x.pt"]
)
def test_parent_segments_are_refused(root, raw):
    with pytest.raises(PathNotAllowedError):
        confine_path(raw, field="weights_path")


def test_a_plain_relative_path_passes_for_its_resolver_to_check(root):
    assert confine_path("runs/model.pt", field="weights_path") == "runs/model.pt"


@pytest.mark.skipif(
    sys.platform != "win32", reason="drive-relative paths are a Windows form"
)
def test_a_drive_relative_path_is_refused(root):
    with pytest.raises(PathNotAllowedError):
        confine_path("C:model.pt", field="weights_path")


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows paths compare without case"
)
def test_case_does_not_matter_on_windows(root):
    shouted = str(root / "model.pt").upper()

    assert (
        Path(confine_path(shouted, field="weights_path"))
        == (root / "model.pt").resolve()
    )


def test_a_symlink_that_leaves_the_root_is_refused(root, tmp_path):
    link = root / "escape"
    try:
        link.symlink_to(tmp_path / "outside", target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is not permitted here")

    with pytest.raises(PathNotAllowedError):
        confine_path(str(link / "model.pt"), field="weights_path")


def test_confine_resolved_checks_a_path_some_resolver_built(root, tmp_path):
    assert (
        confine_resolved(root / "x.yaml", what="pipeline")
        == (root / "x.yaml").resolve()
    )
    with pytest.raises(PathNotAllowedError, match="pipeline"):
        confine_resolved(tmp_path / "outside" / "x.yaml", what="pipeline")


def test_confine_resolved_without_roots_returns_the_path(monkeypatch, tmp_path):
    monkeypatch.delenv(ALLOWED_ROOTS_ENV, raising=False)
    path = tmp_path / "x.yaml"

    assert confine_resolved(path, what="pipeline") == path


# ---------------------------------------------------------------------------
# Every path-bearing request field is screened
# ---------------------------------------------------------------------------

_PATH_WORDS = ("path", "dir", "file", "root", "folder")


def _path_fields() -> set[tuple[str, str]]:
    service = cuvis_ai_pb2.DESCRIPTOR.services_by_name["CuvisAIService"]
    found = set()
    for method in service.methods:
        for field in method.input_type.fields:
            if field.type == field.TYPE_STRING and any(
                word in field.name.lower() for word in _PATH_WORDS
            ):
                found.add((method.name, field.name))
    return found


def test_every_path_field_of_the_api_has_a_confinement():
    covered = {
        (rpc, field)
        for rpc in CuvisAIService.__dict__
        for field in getattr(getattr(CuvisAIService, rpc), "path_fields", ())
    }
    # SetSessionSearchPaths checks each entry when it stores it (a rejected entry is
    # reported back, not an error), see test_search_paths_outside_the_roots_are_rejected.
    covered.add(("SetSessionSearchPaths", "search_paths"))

    assert _path_fields() == covered


_SCREENED_REQUESTS = [
    ("LoadPipelineWeights", "weights_path"),
    ("SavePipeline", "pipeline_path"),
    ("SaveTrainRun", "trainrun_path"),
    ("RestoreTrainRun", "trainrun_path"),
    ("RestoreTrainRun", "weights_path"),
    ("ResolveConfig", "path"),
]


@pytest.mark.parametrize(("rpc", "field"), _SCREENED_REQUESTS)
def test_a_request_path_outside_the_roots_answers_permission_denied(
    root, tmp_path, rpc, field
):
    request = getattr(cuvis_ai_pb2, f"{rpc}Request")()
    setattr(request, field, str(tmp_path / "outside" / "file.yaml"))
    context = MagicMock()

    response = getattr(CuvisAIService(), rpc)(request, context)

    context.set_code.assert_called_once_with(grpc.StatusCode.PERMISSION_DENIED)
    assert isinstance(response, getattr(cuvis_ai_pb2, f"{rpc}Response"))


@pytest.mark.parametrize(("rpc", "field"), _SCREENED_REQUESTS)
def test_a_unc_request_path_is_refused_without_touching_it(
    root, no_remote_filesystem_calls, rpc, field
):
    request = getattr(cuvis_ai_pb2, f"{rpc}Request")()
    setattr(request, field, r"\\fileserver\share\file.yaml")
    context = MagicMock()

    getattr(CuvisAIService(), rpc)(request, context)

    context.set_code.assert_called_once_with(grpc.StatusCode.PERMISSION_DENIED)


def test_a_screened_absolute_path_is_forwarded_resolved(root, monkeypatch):
    seen = {}

    def forward(session_manager, request, context):
        seen["path"] = request.pipeline_path
        return cuvis_ai_pb2.SavePipelineResponse(success=True)

    monkeypatch.setattr(
        "cuvis_ai_core.grpc.service.orchestrator_bridge.forward_save_pipeline", forward
    )
    raw = os.path.join(str(root), "runs", ".", "model.yaml")

    CuvisAIService().SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(session_id="s", pipeline_path=raw), MagicMock()
    )

    assert seen["path"] == str((root / "runs" / "model.yaml").resolve())


# ---------------------------------------------------------------------------
# Resolution points: where a relative path becomes a location
# ---------------------------------------------------------------------------


def test_search_paths_outside_the_roots_are_rejected(
    root, tmp_path, no_remote_filesystem_calls
):
    manager = SessionManager()
    session_id = manager.create_session()
    inside = root / "plugins"
    inside.mkdir()

    accepted, rejected = manager.set_search_paths(
        session_id,
        [str(inside), str(tmp_path / "outside"), r"\\fileserver\share\plugins"],
        append=False,
    )

    assert accepted == [str(inside.resolve())]
    assert rejected == [str(tmp_path / "outside"), r"\\fileserver\share\plugins"]


def test_search_paths_without_roots_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOWED_ROOTS_ENV, raising=False)
    manager = SessionManager()
    session_id = manager.create_session()

    accepted, rejected = manager.set_search_paths(
        session_id, [str(tmp_path)], append=False
    )

    assert accepted == [str(tmp_path.resolve())]
    assert rejected == []


def test_a_relative_save_path_that_lands_outside_the_roots_is_refused(
    root, tmp_path, monkeypatch
):
    monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path / "outside"))

    with pytest.raises(PathNotAllowedError):
        helpers.resolve_pipeline_save_path("saved.yaml")


def test_a_relative_save_path_inside_the_roots_resolves(root, monkeypatch):
    monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(root))

    assert (
        helpers.resolve_pipeline_save_path("saved.yaml")
        == (root / "pipeline" / "saved.yaml").resolve()
    )


def test_weights_found_through_a_search_path_outside_the_roots_are_refused(
    root, tmp_path
):
    outside = tmp_path / "outside"
    (outside / "model.pt").write_bytes(b"")

    with pytest.raises(PathNotAllowedError):
        helpers.find_weights_file("model.pt", [str(outside)])


def test_a_config_found_outside_the_roots_is_refused(root, tmp_path):
    outside = tmp_path / "outside"
    (outside / "trainrun.yaml").write_text("{}", encoding="utf-8")

    with pytest.raises(PathNotAllowedError):
        config_helpers._find_config_file("trainrun.yaml", [str(outside)])


def _write_trainrun(directory: Path, pipeline_ref: str, experiment: dict) -> Path:
    path = directory / "trainrun.yaml"
    path.write_text(
        yaml.safe_dump({**experiment, "pipeline": pipeline_ref}), encoding="utf-8"
    )
    return path


def test_a_trainrun_outside_the_roots_is_refused(root, tmp_path, mock_experiment_dict):
    trainrun = _write_trainrun(
        tmp_path / "outside", "pipeline.yaml", mock_experiment_dict
    )

    with pytest.raises(PathNotAllowedError):
        TrainRunService.parse_trainrun_yaml(trainrun)


def test_a_trainrun_whose_pipeline_lies_outside_the_roots_is_refused(
    root, tmp_path, mock_experiment_dict
):
    outside_pipeline = tmp_path / "outside" / "pipeline.yaml"
    outside_pipeline.write_text("nodes: []\n", encoding="utf-8")
    trainrun = _write_trainrun(root, outside_pipeline.as_posix(), mock_experiment_dict)

    with pytest.raises(PathNotAllowedError, match="pipeline"):
        TrainRunService.parse_trainrun_yaml(trainrun)


def test_a_trainrun_naming_a_unc_pipeline_is_refused_without_touching_it(
    root, no_remote_filesystem_calls, mock_experiment_dict
):
    trainrun = _write_trainrun(
        root, "//fileserver/share/pipeline.yaml", mock_experiment_dict
    )

    with pytest.raises(PathNotAllowedError):
        TrainRunService.parse_trainrun_yaml(trainrun)


def test_a_trainrun_may_climb_to_a_pipeline_inside_the_roots(
    root, mock_experiment_dict
):
    (root / "pipelines").mkdir()
    pipeline = root / "pipelines" / "pipeline.yaml"
    pipeline.write_text("nodes: []\n", encoding="utf-8")
    (root / "runs").mkdir()
    trainrun = _write_trainrun(
        root / "runs", "../pipelines/pipeline.yaml", mock_experiment_dict
    )

    _config, resolved = TrainRunService.parse_trainrun_yaml(trainrun)

    assert resolved == pipeline.resolve()


def test_save_train_run_outside_the_roots_answers_permission_denied(
    root, tmp_path, monkeypatch
):
    # A relative trainrun_path resolves against the working directory.
    monkeypatch.chdir(tmp_path / "outside")
    context = MagicMock()

    response = TrainRunService(MagicMock()).save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id="s", trainrun_path="runs/trainrun.yaml"
        ),
        context,
    )

    context.set_code.assert_called_with(grpc.StatusCode.PERMISSION_DENIED)
    assert not response.success
    assert not (tmp_path / "outside" / "runs").exists()
