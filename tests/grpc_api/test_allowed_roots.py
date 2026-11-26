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
import torch
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.training_service import confined_output_dirs
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import (
    CallbacksConfig,
    ModelCheckpointConfig,
    TrainingConfig,
)
from cuvis_ai_core.utils import config_helpers
from cuvis_ai_core.utils.path_confinement import (
    ALLOWED_ROOTS_ENV,
    PathNotAllowedError,
    allowed_roots,
    check_not_remote,
    confine_path,
    confine_resolved,
)

REMOTE_FORMS = [
    r"\\fileserver\share\weights.pt",
    "//fileserver/share/weights.pt",
    r"\\?\C:\weights.pt",
    r"\\?\UNC\fileserver\share\weights.pt",
    r"\\.\pipe\weights",
    r"\??\C:\weights.pt",
    r"\??\UNC\fileserver\share\weights.pt",
    "/??/C:/weights.pt",
]

# The leading text of every spelling above, as str(Path(...)) prints it.
REMOTE_PREFIXES = ("\\\\", "//", "\\??\\", "/??/")


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """One allowed root; ``tmp_path / "outside"`` lies outside it."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "outside").mkdir()
    monkeypatch.setenv(ALLOWED_ROOTS_ENV, str(allowed))
    return allowed


@pytest.fixture
def no_remote_filesystem_calls(monkeypatch) -> list[str]:
    """Fail the test when anything stats or resolves a remote-looking path.

    The calls are recorded as well, for code that catches the failure and carries on.
    """
    touched: list[str] = []

    def guard(name, original):
        def wrapped(self, *args, **kwargs):
            if str(self).startswith(REMOTE_PREFIXES):
                touched.append(f"{name}({self})")
                raise AssertionError(f"{name} called on {self}")
            return original(self, *args, **kwargs)

        return wrapped

    for name in ("resolve", "exists", "stat", "is_dir", "is_file"):
        monkeypatch.setattr(
            pathlib.Path, name, guard(name, getattr(pathlib.Path, name))
        )
    return touched


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


@pytest.mark.parametrize("raw", REMOTE_FORMS)
def test_confine_resolved_refuses_remote_forms_before_any_filesystem_call(
    root, no_remote_filesystem_calls, raw
):
    with pytest.raises(PathNotAllowedError):
        confine_resolved(Path(raw), what="weights_path")
    assert no_remote_filesystem_calls == []


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


DEVICE_NAMES = ["NUL", "con.yaml", "runs/COM1.txt", "Aux .pt", "lpt9", "CONIN$"]


@pytest.mark.skipif(
    sys.platform != "win32", reason="DOS device names are a Windows form"
)
@pytest.mark.parametrize("name", DEVICE_NAMES)
def test_a_device_name_is_refused_on_its_text(root, name):
    # The message says why: the text check refused it, before anything resolved it.
    with pytest.raises(PathNotAllowedError, match="device"):
        confine_path(name, field="weights_path")
    with pytest.raises(PathNotAllowedError, match="device"):
        confine_path(str(root / name), field="weights_path")
    with pytest.raises(PathNotAllowedError, match="device"):
        confine_resolved(root / name, what="weights_path")


@pytest.mark.parametrize("name", ["console.yaml", "auxiliary.pt", "COM10.txt", "nuls"])
def test_a_name_that_only_starts_like_a_device_passes(root, name):
    assert Path(confine_path(str(root / name), field="weights_path")) == (
        (root / name).resolve()
    )


@pytest.mark.skipif(
    sys.platform != "win32", reason="DOS device names are a Windows form"
)
@pytest.mark.parametrize("ref", ["C:NUL", "C:nul.yaml", "CON:", "runs/aux:x"])
def test_a_device_behind_a_drive_or_a_colon_is_refused_on_its_text(root, ref):
    # A trainrun's pipeline reference only gets this check before the resolver probes it.
    with pytest.raises(PathNotAllowedError, match="device"):
        check_not_remote(ref, what="trainrun pipeline")


@pytest.mark.parametrize(
    "raw",
    [
        "http://fileserver/ckpt",
        "s3://bucket/ckpt",
        "file:///elsewhere/ckpt",
        "~/ckpt",
        "~other/ckpt",
        r"~\ckpt",
    ],
)
def test_a_url_or_a_home_path_is_refused_on_its_text(root, raw):
    # Lightning hands a URL to fsspec and expands '~'; neither is a folder under the roots.
    with pytest.raises(PathNotAllowedError):
        confine_path(raw, field="training.default_root_dir")


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


@pytest.mark.parametrize(
    "remote", [r"\\fileserver\share\file.yaml", r"\??\UNC\fileserver\share\file.yaml"]
)
@pytest.mark.parametrize(("rpc", "field"), _SCREENED_REQUESTS)
def test_a_unc_request_path_is_refused_without_touching_it(
    root, no_remote_filesystem_calls, rpc, field, remote
):
    request = getattr(cuvis_ai_pb2, f"{rpc}Request")()
    setattr(request, field, remote)
    context = MagicMock()

    getattr(CuvisAIService(), rpc)(request, context)

    context.set_code.assert_called_once_with(grpc.StatusCode.PERMISSION_DENIED)
    assert no_remote_filesystem_calls == []


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

    remote = [r"\\fileserver\share\plugins", r"\??\UNC\fileserver\share\plugins"]

    accepted, rejected = manager.set_search_paths(
        session_id, [str(inside), str(tmp_path / "outside"), *remote], append=False
    )

    assert accepted == [str(inside.resolve())]
    assert rejected == [str(tmp_path / "outside"), *remote]
    # The validator logs and rejects on any error, so the guard's own failure is not enough.
    assert no_remote_filesystem_calls == []


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


def test_a_trainrun_whose_pipeline_lies_outside_the_roots_is_not_found(
    root, tmp_path, mock_experiment_dict
):
    outside_pipeline = tmp_path / "outside" / "pipeline.yaml"
    outside_pipeline.write_text("nodes: []\n", encoding="utf-8")
    trainrun = _write_trainrun(root, outside_pipeline.as_posix(), mock_experiment_dict)

    with pytest.raises(FileNotFoundError):
        TrainRunService.parse_trainrun_yaml(trainrun)


def test_a_trainrun_pipeline_outside_the_roots_answers_like_a_missing_one(
    root, tmp_path, monkeypatch, mock_experiment_dict
):
    # Otherwise the answer tells a client whether a file outside the roots exists.
    (tmp_path / "outside" / "exists.yaml").write_text("nodes: []\n", encoding="utf-8")
    probed: list[str] = []
    real_is_file = pathlib.Path.is_file

    def spy(self):
        probed.append(str(self))
        return real_is_file(self)

    monkeypatch.setattr(pathlib.Path, "is_file", spy)

    for name in ("exists", "missing"):
        trainrun = _write_trainrun(root, f"../outside/{name}", mock_experiment_dict)
        with pytest.raises(FileNotFoundError):
            TrainRunService.parse_trainrun_yaml(trainrun)
    assert [path for path in probed if "outside" in path] == []


@pytest.mark.parametrize(
    "pipeline_ref",
    ["//fileserver/share/pipeline.yaml", r"\??\UNC\fileserver\share\pipeline.yaml"],
)
def test_a_trainrun_naming_a_unc_pipeline_is_refused_without_touching_it(
    root, no_remote_filesystem_calls, mock_experiment_dict, pipeline_ref
):
    trainrun = _write_trainrun(root, pipeline_ref, mock_experiment_dict)

    with pytest.raises(PathNotAllowedError):
        TrainRunService.parse_trainrun_yaml(trainrun)
    assert no_remote_filesystem_calls == []


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


# ---------------------------------------------------------------------------
# Review round 1: derived writes, the sibling fallback, Hydra search paths
# ---------------------------------------------------------------------------


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("creating symlinks is not permitted here")


def test_weights_climbing_out_of_a_search_path_are_refused(root, tmp_path):
    (tmp_path / "outside" / "model.pt").write_bytes(b"")
    (root / "weights").mkdir()

    with pytest.raises(PathNotAllowedError):
        helpers.find_weights_file("../../outside/model.pt", [str(root / "weights")])


def test_a_save_whose_checkpoint_links_outside_the_roots_is_refused(root, tmp_path):
    victim = tmp_path / "outside" / "model.pt"
    victim.write_bytes(b"keep me")
    _symlink_or_skip(root / "model.pt", victim)

    with pytest.raises(PathNotAllowedError):
        helpers.resolve_pipeline_save_path(str(root / "model.yaml"))
    assert victim.read_bytes() == b"keep me"


def test_a_save_through_a_linked_yaml_checks_the_checkpoint_beside_its_target(
    root, tmp_path
):
    # save_to_file writes the .pt beside the yaml it resolved to, not beside the link.
    victim = tmp_path / "outside" / "real.pt"
    victim.write_bytes(b"keep me")
    (root / "sub").mkdir()
    (root / "sub" / "real.yaml").write_text("nodes: []\n", encoding="utf-8")
    _symlink_or_skip(root / "model.yaml", root / "sub" / "real.yaml")
    _symlink_or_skip(root / "sub" / "real.pt", victim)

    with pytest.raises(PathNotAllowedError):
        helpers.resolve_pipeline_save_path(str(root / "model.yaml"))
    assert victim.read_bytes() == b"keep me"


@pytest.mark.parametrize("linked", ["model.yaml", "model.pt"])
def test_saving_a_pipeline_replaces_a_link_put_there_after_the_check(
    root, tmp_path, mock_statistical_trainable_node, linked
):
    # SavePipeline checked the target; a link swapped in before the write must not be followed.
    victim = tmp_path / "outside" / "victim"
    victim.write_text("keep", encoding="utf-8")
    _symlink_or_skip(root / linked, victim)
    pipeline = CuvisPipeline("staged_save")
    node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
    pipeline._graph.add_node(node)
    node.statistical_initialization(iter([{"data": torch.randn(1, 4, 4, 3)}]))

    pipeline.save_to_file(root / "model.yaml")

    assert victim.read_text(encoding="utf-8") == "keep"
    assert not (root / linked).is_symlink()
    assert (root / "model.yaml").is_file() and (root / "model.pt").is_file()


def test_save_train_run_whose_pipeline_sibling_links_outside_is_refused(root, tmp_path):
    victim = tmp_path / "outside" / "run_pipeline.yaml"
    victim.write_text("keep: me\n", encoding="utf-8")
    _symlink_or_skip(root / "run_pipeline.yaml", victim)
    context = MagicMock()

    response = TrainRunService(MagicMock()).save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id="s", trainrun_path=str(root / "run.yaml"), save_weights=True
        ),
        context,
    )

    context.set_code.assert_called_with(grpc.StatusCode.PERMISSION_DENIED)
    assert not response.success
    assert victim.read_text(encoding="utf-8") == "keep: me\n"


def test_save_train_run_names_a_linked_sibling_by_its_own_name(root):
    kept = root / "pipelines" / "kept.yaml"
    kept.parent.mkdir()
    kept.write_text("nodes: []\n", encoding="utf-8")
    _symlink_or_skip(root / "run_pipeline.yaml", kept)
    manager = MagicMock()
    session = manager.get_session.return_value
    session.has_live_pipeline = True
    session.trainrun_config = None
    session.data_config = None
    session.training_config = None
    session.pipeline.name = "run"
    session.pipeline_config.save_to_file = lambda path: Path(path).write_text(
        "nodes: []\n", encoding="utf-8"
    )

    response = TrainRunService(manager).save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id="s", trainrun_path=str(root / "run.yaml")
        ),
        MagicMock(),
    )

    assert response.success
    saved = yaml.safe_load((root / "run.yaml").read_text(encoding="utf-8"))
    # Resolved beside run.yaml, the link name finds the pipeline; the target's name does not.
    assert saved["pipeline"] == "run_pipeline.yaml"


def test_a_sibling_pipeline_that_links_outside_the_roots_is_refused(
    root, tmp_path, mock_experiment_dict
):
    outside_pipeline = tmp_path / "outside" / "pipeline.yaml"
    outside_pipeline.write_text("nodes: []\n", encoding="utf-8")
    cfg = {k: v for k, v in mock_experiment_dict.items() if k != "pipeline"}
    trainrun = root / "run.yaml"
    trainrun.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _symlink_or_skip(root / "run_pipeline.yaml", outside_pipeline)

    with pytest.raises(PathNotAllowedError):
        TrainRunService.parse_trainrun_yaml(trainrun)


@pytest.mark.parametrize(
    "override",
    [
        "hydra.searchpath=[file:////fileserver/share]",
        "+hydra.searchpath=[file:///elsewhere]",
        "++hydra.searchpath=[pkg://somewhere]",
        "hydra/job_logging=disabled",
    ],
)
def test_hydra_settings_in_overrides_are_refused(
    root, no_remote_filesystem_calls, override
):
    (root / "trainrun.yaml").write_text("name: run\n", encoding="utf-8")

    with pytest.raises(PathNotAllowedError, match="hydra"):
        config_helpers.resolve_config_with_hydra(
            "trainrun", "trainrun.yaml", [str(root)], [override]
        )


def test_a_config_that_sets_its_own_hydra_search_path_is_refused(root):
    (root / "trainrun.yaml").write_text(
        "name: run\nhydra:\n  searchpath:\n    - file:////fileserver/share\n",
        encoding="utf-8",
    )

    with pytest.raises(PathNotAllowedError, match="hydra"):
        config_helpers.resolve_config_with_hydra(
            "trainrun", "trainrun.yaml", [str(root)], []
        )


def test_plain_overrides_still_resolve_with_roots(root, monkeypatch):
    seen = {}

    def compose(config_name, overrides):
        seen["overrides"] = overrides
        return {"name": "run"}

    monkeypatch.setattr("hydra.compose", compose)
    (root / "trainrun.yaml").write_text("name: run\n", encoding="utf-8")

    with pytest.raises(
        Exception
    ):  # the stub config is no valid trainrun; only the screen matters
        config_helpers.resolve_config_with_hydra(
            "trainrun", "trainrun.yaml", [str(root)], ["name=other"]
        )
    assert seen["overrides"] == ["name=other"]


# ---------------------------------------------------------------------------
# Review round 2: the folders gradient training writes to
# ---------------------------------------------------------------------------


def _training_config(**fields) -> TrainingConfig:
    checkpoint = fields.pop("checkpoint", None)
    filename = fields.pop("filename", None)
    callbacks = (
        CallbacksConfig(
            checkpoint=ModelCheckpointConfig(dirpath=checkpoint, filename=filename)
        )
        if checkpoint is not None
        else None
    )
    return TrainingConfig(callbacks=callbacks, **fields)


def test_training_output_dirs_inside_the_roots_come_back_resolved(root):
    config = _training_config(
        default_root_dir=str(root / "runs"), checkpoint=str(root / "ckpt")
    )

    confined = confined_output_dirs(config)

    assert confined.default_root_dir == str((root / "runs").resolve())
    assert confined.callbacks.checkpoint.dirpath == str((root / "ckpt").resolve())


def test_relative_training_output_dirs_inside_the_roots_stay_relative(
    root, monkeypatch
):
    # A saved trainrun keeps them relative; they land under the working directory.
    monkeypatch.chdir(root)
    config = _training_config(default_root_dir="runs", checkpoint="checkpoints")

    confined = confined_output_dirs(config)

    assert confined.default_root_dir == "runs"
    assert confined.callbacks.checkpoint.dirpath == "checkpoints"


@pytest.mark.parametrize(
    "fields",
    [
        {"default_root_dir": "runs"},
        {"checkpoint": "checkpoints"},
        {},
    ],
    ids=["relative-root-dir", "relative-checkpoint-dir", "no-root-dir"],
)
def test_training_output_under_a_working_directory_outside_the_roots_is_refused(
    root, tmp_path, monkeypatch, fields
):
    # Lightning writes a relative folder, or everything when default_root_dir is
    # missing, under the working directory.
    monkeypatch.chdir(tmp_path / "outside")
    if "checkpoint" in fields:  # only the checkpoint folder is relative here
        fields = {**fields, "default_root_dir": str(root / "runs")}

    with pytest.raises(PathNotAllowedError):
        confined_output_dirs(_training_config(**fields))


def test_a_missing_root_dir_passes_with_the_working_directory_inside_the_roots(
    root, monkeypatch
):
    monkeypatch.chdir(root)

    assert confined_output_dirs(_training_config()).default_root_dir is None


@pytest.mark.parametrize(
    "filename",
    [
        "../x",
        "runs/../../escape-{epoch}",
        r"C:\elsewhere\x",
        "/elsewhere/x",
        r"\\fileserver\share\x",
        r"\??\UNC\fileserver\share\x",
    ],
)
def test_a_checkpoint_file_name_that_leaves_its_folder_is_refused(
    root, no_remote_filesystem_calls, filename
):
    config = _training_config(checkpoint=str(root / "ckpt"), filename=filename)

    with pytest.raises(PathNotAllowedError):
        confined_output_dirs(config)
    assert no_remote_filesystem_calls == []


@pytest.mark.parametrize("filename", ["{epoch:02d}", "runs/{epoch}-{step}"])
def test_a_checkpoint_file_name_inside_its_folder_passes_unchanged(root, filename):
    config = _training_config(
        default_root_dir=str(root / "runs"),
        checkpoint=str(root / "ckpt"),
        filename=filename,
    )

    confined = confined_output_dirs(config)

    assert confined.callbacks.checkpoint.filename == filename


def test_a_checkpoint_file_name_through_a_link_out_of_the_roots_is_refused(
    root, tmp_path
):
    (root / "ckpt").mkdir()
    _symlink_or_skip(root / "ckpt" / "escape", tmp_path / "outside")
    config = _training_config(
        default_root_dir=str(root / "runs"),
        checkpoint=str(root / "ckpt"),
        filename="escape/{epoch}",
    )

    with pytest.raises(PathNotAllowedError):
        confined_output_dirs(config)


@pytest.mark.parametrize("field", ["default_root_dir", "checkpoint"])
@pytest.mark.parametrize("raw", ["~/runs", "http://fileserver/runs"])
def test_a_training_folder_lightning_would_expand_or_fetch_is_refused(
    root, monkeypatch, field, raw
):
    monkeypatch.chdir(root)
    fields = {"default_root_dir": str(root / "runs"), field: raw}

    with pytest.raises(PathNotAllowedError):
        confined_output_dirs(_training_config(**fields))


def test_loading_weights_found_outside_the_roots_answers_permission_denied(
    root, tmp_path
):
    from cuvis_ai_core.grpc import orchestrator_bridge

    (tmp_path / "outside" / "model.pt").write_bytes(b"")
    manager = SessionManager()
    session_id = manager.create_session()
    manager.get_session(session_id).search_paths = [str(tmp_path / "outside")]
    context = MagicMock()

    response = orchestrator_bridge.forward_load_pipeline_weights(
        manager,
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id, weights_path="model.pt"
        ),
        context,
    )
    manager.close_session(session_id)

    context.set_code.assert_called_once_with(grpc.StatusCode.PERMISSION_DENIED)
    assert not response.success


def test_restoring_a_trainrun_outside_the_roots_answers_permission_denied(
    root, tmp_path, mock_experiment_dict
):
    # Called on the bridge directly, past the parent's request screen.
    from cuvis_ai_core.grpc import orchestrator_bridge

    trainrun = _write_trainrun(
        tmp_path / "outside", "pipeline.yaml", mock_experiment_dict
    )
    context = MagicMock()

    response = orchestrator_bridge.forward_restore_train_run(
        SessionManager(),
        cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=str(trainrun)),
        context,
    )

    context.set_code.assert_called_once_with(grpc.StatusCode.PERMISSION_DENIED)
    assert response == cuvis_ai_pb2.RestoreTrainRunResponse()


@pytest.mark.parametrize("where", ["network", "outside"])
def test_training_with_a_splits_file_outside_the_roots_answers_permission_denied(
    root, tmp_path, no_remote_filesystem_calls, where
):
    from cuvis_ai_schemas.training.data import DataSplitConfig

    from tests.grpc_api.test_train_cancellation import _make_session

    if where == "network":
        splits_path = "//fileserver/share/splits.json"
    else:
        splits_path = str(tmp_path / "outside" / "splits.json")
        Path(splits_path).write_text("{}", encoding="utf-8")
    manager = SessionManager()
    session_id, service = _make_session(manager)
    session = manager.get_session(session_id)
    session.data_config = session.data_config.model_copy(
        update={"splits": DataSplitConfig(splits_path=splits_path)}
    )
    context = MagicMock()

    responses = list(
        service.train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            ),
            context,
        )
    )
    manager.close_session(session_id)

    context.set_code.assert_called_with(grpc.StatusCode.PERMISSION_DENIED)
    assert responses == []
    assert no_remote_filesystem_calls == []


def test_training_output_dirs_without_roots_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv(ALLOWED_ROOTS_ENV, raising=False)
    config = _training_config(default_root_dir=str(tmp_path), checkpoint="../anywhere")

    assert confined_output_dirs(config) == config


@pytest.mark.parametrize(
    "dirpath",
    ["../escape", r"\??\UNC\fileserver\share\ckpt", "//fileserver/share/ckpt"],
)
def test_a_checkpoint_dir_that_leaves_the_roots_by_its_text_is_refused(
    root, no_remote_filesystem_calls, dirpath
):
    with pytest.raises(PathNotAllowedError):
        confined_output_dirs(_training_config(checkpoint=dirpath))
    assert no_remote_filesystem_calls == []


@pytest.mark.parametrize("field", ["default_root_dir", "checkpoint"])
def test_gradient_training_into_a_folder_outside_the_roots_answers_permission_denied(
    root, tmp_path, field
):
    from tests.grpc_api.test_train_cancellation import _make_session

    outside = tmp_path / "outside" / "runs"
    manager = SessionManager()
    session_id, service = _make_session(manager)
    session = manager.get_session(session_id)
    update = _training_config(**{field: str(outside)})
    session.training_config = session.training_config.model_copy(
        update={
            "default_root_dir": update.default_root_dir,
            "callbacks": update.callbacks,
        }
    )
    context = MagicMock()

    responses = list(
        service.train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT
            ),
            context,
        )
    )
    manager.close_session(session_id)

    context.set_code.assert_called_with(grpc.StatusCode.PERMISSION_DENIED)
    assert responses == []
    assert not outside.exists()


# ---------------------------------------------------------------------------
# Review round 2: files Hydra pulls in while composing
# ---------------------------------------------------------------------------


@pytest.fixture
def outside_extra(tmp_path) -> Path:
    """A config outside the roots that a compose must not read."""
    extra = tmp_path / "outside" / "extra.yaml"
    extra.parent.mkdir(exist_ok=True)
    extra.write_text("seed: 666\n", encoding="utf-8")
    return extra


def _resolve_training(
    root: Path, body: str, overrides: list[str] | None = None
) -> dict:
    (root / "training.yaml").write_text(body, encoding="utf-8")
    return config_helpers.resolve_config_with_hydra(
        "training", "training.yaml", [str(root)], overrides or []
    )


def test_a_defaults_entry_that_climbs_out_of_the_roots_is_refused(root, outside_extra):
    with pytest.raises(PathNotAllowedError):
        _resolve_training(
            root, "defaults:\n  - ../outside/extra@_global_\n  - _self_\n"
        )


def test_a_group_override_that_climbs_out_of_the_roots_is_refused(root, outside_extra):
    # A folder makes "extra" a config group, so the override names a file to load.
    (root / "extra").mkdir()
    (root / "extra" / "inside.yaml").write_text("seed: 3\n", encoding="utf-8")

    with pytest.raises(PathNotAllowedError):
        _resolve_training(root, "seed: 1\n", ["+extra=../../outside/extra"])


def test_a_defaults_entry_that_links_out_of_the_roots_is_refused(root, outside_extra):
    _symlink_or_skip(root / "linked.yaml", outside_extra)

    with pytest.raises(PathNotAllowedError):
        _resolve_training(root, "defaults:\n  - linked@_global_\n  - _self_\n")


def test_defaults_inside_the_roots_still_compose(root):
    (root / "extra.yaml").write_text("seed: 7\n", encoding="utf-8")

    resolved = _resolve_training(root, "defaults:\n  - extra@_global_\n  - _self_\n")

    assert resolved["seed"] == 7


def test_defaults_without_roots_compose_as_before(tmp_path, monkeypatch, outside_extra):
    monkeypatch.delenv(ALLOWED_ROOTS_ENV, raising=False)
    configs = tmp_path / "configs"
    configs.mkdir()

    resolved = _resolve_training(
        configs, "defaults:\n  - ../outside/extra@_global_\n  - _self_\n"
    )

    assert resolved["seed"] == 666


def test_a_search_path_above_the_roots_still_composes_a_config_inside_them(
    tmp_path, monkeypatch
):
    configs = tmp_path / "configs"
    training = configs / "training"
    training.mkdir(parents=True)
    (training / "extra.yaml").write_text("seed: 5\n", encoding="utf-8")
    (training / "default.yaml").write_text(
        "defaults:\n  - extra@_global_\n  - _self_\n", encoding="utf-8"
    )
    monkeypatch.setenv(ALLOWED_ROOTS_ENV, str(training))

    resolved = config_helpers.resolve_config_with_hydra(
        "training", "training/default.yaml", [str(configs)], []
    )

    assert resolved["seed"] == 5


# ---------------------------------------------------------------------------
# Review round 3: the server's pipeline library is readable, not writable
# ---------------------------------------------------------------------------


@pytest.fixture
def library(tmp_path, monkeypatch) -> Path:
    """The server's pipeline library (CUVIS_CONFIGS_DIR), outside the roots."""
    configs = tmp_path / "library"
    (configs / "pipeline").mkdir(parents=True)
    monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(configs))
    return configs


def test_a_config_in_the_pipeline_library_resolves_with_its_includes(root, library):
    (library / "extra.yaml").write_text("seed: 9\n", encoding="utf-8")
    (library / "training.yaml").write_text(
        "defaults:\n  - extra@_global_\n  - _self_\n", encoding="utf-8"
    )

    resolved = config_helpers.resolve_config_with_hydra(
        "training", "training.yaml", [str(library)], []
    )

    assert resolved["seed"] == 9


def test_weights_in_the_pipeline_library_are_found(root, library):
    weights = library / "pipeline" / "model.pt"
    weights.write_bytes(b"")

    assert helpers.find_weights_file("pipeline/model.pt", [str(library)]) == (
        weights.resolve()
    )
    assert helpers.find_weights_file(str(weights), []) == weights.resolve()


def test_the_pipeline_library_is_accepted_as_a_search_path(root, library):
    manager = SessionManager()
    session_id = manager.create_session()

    accepted, rejected = manager.set_search_paths(
        session_id, [str(library)], append=False
    )
    manager.close_session(session_id)

    assert accepted == [str(library.resolve())]
    assert rejected == []


def test_a_trainrun_in_the_pipeline_library_is_read(
    root, library, mock_experiment_dict
):
    (library / "pipeline" / "run.yaml").write_text("nodes: []\n", encoding="utf-8")
    trainrun = _write_trainrun(library, "pipeline/run.yaml", mock_experiment_dict)

    _config, pipeline = TrainRunService.parse_trainrun_yaml(trainrun)

    assert pipeline == (library / "pipeline" / "run.yaml").resolve()


@pytest.mark.parametrize(
    ("rpc", "fields"),
    [
        ("ResolveConfig", ["path"]),
        ("LoadPipelineWeights", ["weights_path"]),
        ("RestoreTrainRun", ["trainrun_path", "weights_path"]),
        ("GetPipelineInfo", ["pipeline_path"]),
    ],
)
def test_a_read_from_the_pipeline_library_passes_the_screen(root, library, rpc, fields):
    request = getattr(cuvis_ai_pb2, f"{rpc}Request")()
    for field in fields:
        setattr(request, field, str(library / "pipeline" / f"{field}.yaml"))
    context = MagicMock()

    getattr(CuvisAIService(), rpc)(request, context)

    refused = [
        call.args
        for call in context.set_code.call_args_list
        if call.args == (grpc.StatusCode.PERMISSION_DENIED,)
    ]
    assert refused == []


@pytest.mark.parametrize(
    ("rpc", "field"),
    [("SavePipeline", "pipeline_path"), ("SaveTrainRun", "trainrun_path")],
)
def test_a_write_into_the_pipeline_library_is_refused(root, library, rpc, field):
    request = getattr(cuvis_ai_pb2, f"{rpc}Request")()
    setattr(request, field, str(library / "pipeline" / "mine.yaml"))
    context = MagicMock()

    getattr(CuvisAIService(), rpc)(request, context)

    context.set_code.assert_called_once_with(grpc.StatusCode.PERMISSION_DENIED)


def test_a_relative_save_into_the_pipeline_library_is_refused(root, library):
    with pytest.raises(PathNotAllowedError):
        helpers.resolve_pipeline_save_path("mine.yaml")


def test_training_cannot_write_into_the_pipeline_library(root, library):
    config = _training_config(default_root_dir=str(library / "runs"))

    with pytest.raises(PathNotAllowedError):
        confined_output_dirs(config)
