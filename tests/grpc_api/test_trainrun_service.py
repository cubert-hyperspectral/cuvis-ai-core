"""Fast TrainRunService tests.

`tests/grpc_api/test_experiment_management.py` exercises save/restore through
the full gRPC server and is marked ``slow`` (deselected in the coverage job).
These tests drive ``TrainRunService`` directly with mock nodes so the
save-guard, yaml-parse, weights-isolation, and restore branches are covered in
the default (fast) run.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock, Mock

import grpc
import pytest
import torch
import yaml

from cuvis_ai_core.grpc import training_service as training_service_mod
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import TrainRunConfig
from cuvis_ai_schemas.training import CalibrationConfig


def _service():
    sm = SessionManager()
    return sm, TrainRunService(sm)


# ---------------------------------------------------------------------------
# save_train_run
# ---------------------------------------------------------------------------


def test_save_train_run_unknown_session_fails():
    _sm, service = _service()
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(session_id="nope", trainrun_path="x.yaml"),
        Mock(),
    )
    assert resp.success is False


def test_save_train_run_requires_trainrun_path():
    sm, service = _service()
    sid = sm.create_session()
    ctx = Mock()
    resp = service.save_train_run(cuvis_ai_pb2.SaveTrainRunRequest(session_id=sid), ctx)
    assert resp.success is False
    ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)


def test_save_train_run_writes_existing_config(tmp_path, mock_experiment_dict):
    sm, service = _service()
    sid = sm.create_session()
    sm.get_session(sid).trainrun_config = TrainRunConfig.from_dict(mock_experiment_dict)
    out = tmp_path / "tr.yaml"
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=False
        ),
        Mock(),
    )
    assert resp.success is True
    assert out.exists()
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["pipeline"]


def test_save_train_run_isolates_weights_save_failure(tmp_path, mock_experiment_dict):
    sm, service = _service()
    sid = sm.create_session()
    session = sm.get_session(sid)
    session.trainrun_config = TrainRunConfig.from_dict(mock_experiment_dict)

    bad_node = MagicMock()
    bad_node.name = "n"
    bad_node.state_dict.side_effect = RuntimeError("state_dict blew up")
    session.pipeline = MagicMock()
    session.pipeline.nodes.return_value = [bad_node]

    ctx = Mock()
    out = tmp_path / "tr.yaml"
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=True
        ),
        ctx,
    )
    # The yaml still saved; only the optional weights dump failed, and the answer
    # says so: no weights_path, nothing left beside the yaml.
    assert resp.success is True
    assert out.exists()
    assert resp.weights_path == ""
    assert not list(tmp_path.glob("*.pt")) and not list(tmp_path.glob("*.tmp"))
    assert any(
        "weights save failed" in str(c.args[0]) for c in ctx.set_details.mock_calls
    )


def _session_with_pipeline(sm, mock_experiment_dict, mock_pipeline_dict):
    import torch
    from cuvis_ai_schemas.pipeline.config import PipelineConfig

    sid = sm.create_session()
    session = sm.get_session(sid)
    session.trainrun_config = TrainRunConfig.from_dict(mock_experiment_dict)
    session._pipeline_config = PipelineConfig.from_dict(mock_pipeline_dict)
    node = MagicMock()
    node.name = "n"
    node.state_dict.return_value = {"w": torch.zeros(2)}
    session.pipeline = MagicMock()
    session.pipeline.nodes.return_value = [node]
    return sid


@pytest.mark.parametrize("failing", ["torch.save", "os.replace"])
def test_save_train_run_keeps_an_older_checkpoint_intact_and_names_it(
    tmp_path, mock_experiment_dict, mock_pipeline_dict, monkeypatch, failing
):
    """A failed weights save (the write or the move into place) leaves the previous
    <stem>.pt byte-identical, no temp file beside it, an empty weights_path, and details
    that name the failure and the older checkpoint a loader would otherwise pick up."""
    import os

    import torch

    sm, service = _service()
    sid = _session_with_pipeline(sm, mock_experiment_dict, mock_pipeline_dict)
    weights = tmp_path / "tr_pipeline.pt"
    weights.write_bytes(b"older checkpoint")

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    real_replace = os.replace

    def fail_weights_move(src, dst, *args, **kwargs):
        # The yaml writes are staged through os.replace as well; only the
        # checkpoint's move into place fails here.
        if str(dst).endswith(".pt"):
            raise OSError("disk full")
        return real_replace(src, dst, *args, **kwargs)

    if failing == "torch.save":
        monkeypatch.setattr(torch, "save", fail)
    else:
        monkeypatch.setattr(os, "replace", fail_weights_move)

    ctx = Mock()
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(tmp_path / "tr.yaml"), save_weights=True
        ),
        ctx,
    )
    assert resp.success is True
    assert resp.weights_path == ""
    assert (tmp_path / "tr.yaml").exists()
    assert weights.read_bytes() == b"older checkpoint"
    assert not list(tmp_path.glob("*.tmp"))
    detail = ctx.set_details.call_args.args[0]
    assert "weights save failed" in detail and "disk full" in detail
    assert str(weights) in detail


def test_save_train_run_reanchors_reference_to_sibling(
    tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    """A restored run saved elsewhere re-emits a self-contained sibling pipeline."""
    from cuvis_ai_schemas.pipeline.config import PipelineConfig

    sm, service = _service()
    sid = sm.create_session()
    session = sm.get_session(sid)
    # Restored-style: the config carries a reference relative to the *source* dir,
    # and the session holds the live pipeline.
    session.trainrun_config = TrainRunConfig.from_dict(
        {**mock_experiment_dict, "pipeline": "../elsewhere/p.yaml"}
    )
    session._pipeline_config = PipelineConfig.from_dict(mock_pipeline_dict)

    out = tmp_path / "dest" / "tr.yaml"
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=False
        ),
        Mock(),
    )
    assert resp.success is True
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert saved["pipeline"] == "tr_pipeline.yaml"  # re-anchored to the new dir
    sibling = out.parent / "tr_pipeline.yaml"
    assert sibling.is_file()
    assert PipelineConfig.load_from_file(sibling).nodes  # self-contained + loadable
    assert resp.pipeline_path == str(sibling)  # reported in the response


def test_save_train_run_names_weights_after_the_pipeline_sibling(
    tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    """The checkpoint is `<pipeline yaml stem>.pt`, so every loader finds it by default.

    `CuvisPipeline.load_pipeline`, the CLI restore and the CuvisNEXT pickers derive the
    weights path from the pipeline yaml; a checkpoint named after the trainrun yaml
    (`tr.pt` beside `tr_pipeline.yaml`) was invisible to all of them.
    """
    import torch
    from cuvis_ai_schemas.pipeline.config import PipelineConfig

    sm, service = _service()
    sid = sm.create_session()
    session = sm.get_session(sid)
    session.trainrun_config = TrainRunConfig.from_dict(mock_experiment_dict)
    session._pipeline_config = PipelineConfig.from_dict(mock_pipeline_dict)
    node = MagicMock()
    node.name = "n"
    node.state_dict.return_value = {"w": torch.zeros(2)}
    session.pipeline = MagicMock()
    session.pipeline.nodes.return_value = [node]

    out = tmp_path / "tr.yaml"
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=True
        ),
        Mock(),
    )
    assert resp.success is True
    sibling = out.parent / "tr_pipeline.yaml"
    weights = sibling.with_suffix(".pt")
    assert weights.is_file(), sorted(p.name for p in tmp_path.iterdir())
    assert not (tmp_path / "tr.pt").exists()
    assert resp.weights_path == str(weights)
    assert resp.pipeline_path == str(sibling)
    checkpoint = torch.load(weights, weights_only=False)
    assert set(checkpoint["state_dict"]) == {"n"}


# ---------------------------------------------------------------------------
# parse_trainrun_yaml
# ---------------------------------------------------------------------------


def test_parse_trainrun_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainRunService.parse_trainrun_yaml(tmp_path / "absent.yaml")


def test_parse_trainrun_yaml_rejects_hydra_defaults(tmp_path):
    p = tmp_path / "tr.yaml"
    p.write_text("defaults:\n  - base\nname: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Hydra defaults"):
        TrainRunService.parse_trainrun_yaml(p)


def test_parse_trainrun_yaml_missing_pipeline_section_raises(tmp_path):
    p = tmp_path / "tr.yaml"
    p.write_text(
        "name: x\ndata:\n  data_module: cu3s\n  params:\n    cu3s_file_path: ''\n"
        "training:\n  seed: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pipeline"):
        TrainRunService.parse_trainrun_yaml(p)


def test_parse_trainrun_yaml_finds_sibling_pipeline_file(
    tmp_path, mock_experiment_dict
):
    # No explicit reference -> fall back to the <name>_pipeline.yaml sibling.
    cfg = {k: v for k, v in mock_experiment_dict.items() if k != "pipeline"}
    tr = tmp_path / "run.yaml"
    tr.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    sibling = tmp_path / "run_pipeline.yaml"
    sibling.write_text("nodes: []\n", encoding="utf-8")

    _config, pipeline_path = TrainRunService.parse_trainrun_yaml(tr)
    assert pipeline_path == sibling


def test_parse_trainrun_yaml_explicit_missing_reference_raises(
    tmp_path, mock_experiment_dict
):
    # An explicit pipeline reference that does not resolve must fail loudly, even
    # when a <name>_pipeline.yaml sibling happens to exist (no silent masking).
    tr = tmp_path / "run.yaml"
    tr.write_text(
        yaml.safe_dump(mock_experiment_dict), encoding="utf-8"
    )  # pipeline: pipeline.yaml
    (tmp_path / "run_pipeline.yaml").write_text("nodes: []\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="could not be resolved"):
        TrainRunService.parse_trainrun_yaml(tr)


def test_parse_trainrun_yaml_resolves_relative_splits_path(
    tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    """A relative ``data.splits.splits_path`` resolves against the trainrun dir.

    Parity with the CLI restore path: a gRPC-restored trainrun that references its
    splits file relatively must carry an absolute path, so the later Train call is
    never rejected as baseless and never resolves against the child's CWD.
    """
    cfg = {
        **mock_experiment_dict,
        "data": {"data_module": "npz_multi", "splits": {"splits_path": "splits.json"}},
    }
    tr = tmp_path / "run.yaml"
    tr.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(
        yaml.safe_dump(mock_pipeline_dict), encoding="utf-8"
    )

    parsed, _pipeline_path = TrainRunService.parse_trainrun_yaml(tr)

    assert Path(parsed.data.splits.splits_path).is_absolute()
    assert parsed.data.splits.splits_path == str(tmp_path / "splits.json")


# ---------------------------------------------------------------------------
# restore_train_run (builds a real pipeline from mock nodes)
# ---------------------------------------------------------------------------


def _write_trainrun(tmp_path, mock_experiment_dict, mock_pipeline_dict) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(mock_experiment_dict), encoding="utf-8")
    # Materialise the pipeline the trainrun references (pipeline: pipeline.yaml).
    (tmp_path / mock_experiment_dict["pipeline"]).write_text(
        yaml.safe_dump(mock_pipeline_dict), encoding="utf-8"
    )
    return p


def test_restore_train_run_creates_new_session(
    tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    sm, service = _service()
    req = cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path=str(
            _write_trainrun(tmp_path, mock_experiment_dict, mock_pipeline_dict)
        )
    )
    resp = service.restore_train_run(req, Mock())
    assert resp.session_id
    assert resp.session_id in sm.list_sessions()


def test_restore_train_run_reuses_target_session(
    tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    sm, service = _service()
    sid = sm.create_session()
    req = cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path=str(
            _write_trainrun(tmp_path, mock_experiment_dict, mock_pipeline_dict)
        )
    )
    resp = service.restore_train_run(req, Mock(), target_session_id=sid)
    assert resp.session_id == sid
    session = sm.get_session(sid)
    assert session.pipeline is not None
    assert session.trainrun_config is not None


def test_restore_train_run_missing_weights_is_not_found(
    tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    sm, service = _service()
    ctx = Mock()
    req = cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path=str(
            _write_trainrun(tmp_path, mock_experiment_dict, mock_pipeline_dict)
        ),
        weights_path=str(tmp_path / "missing.pt"),
    )
    service.restore_train_run(req, ctx)
    ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)


# ---------------------------------------------------------------------------
# where a restore puts the pipeline, an out-of-memory restore, the calibration block
# ---------------------------------------------------------------------------


def _restore_device(
    monkeypatch, tmp_path, experiment, mock_pipeline_dict, *, cuda=True
):
    devices = []

    def load_pipeline(*_args, **kwargs):
        devices.append(kwargs["device"])
        return MagicMock()

    monkeypatch.setattr(CuvisPipeline, "load_pipeline", staticmethod(load_pipeline))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    _, service = _service()
    context = Mock()
    request = cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path=str(_write_trainrun(tmp_path, experiment, mock_pipeline_dict))
    )
    service.restore_train_run(request, context)
    context.set_code.assert_not_called()
    return devices


@pytest.mark.parametrize(
    ("accelerator", "cuda", "device"),
    [
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("auto", True, "cuda"),
        ("gpu", True, "cuda"),
        ("auto", False, None),
    ],
)
def test_restore_train_run_restores_where_the_trainrun_asks(
    monkeypatch,
    tmp_path,
    mock_experiment_dict,
    mock_pipeline_dict,
    accelerator,
    cuda,
    device,
):
    experiment = copy.deepcopy(mock_experiment_dict)
    experiment["training"]["accelerator"] = accelerator
    assert _restore_device(
        monkeypatch, tmp_path, experiment, mock_pipeline_dict, cuda=cuda
    ) == [device]


def test_restore_train_run_without_a_training_block_keeps_cuda_when_available(
    monkeypatch, tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    experiment = copy.deepcopy(mock_experiment_dict)
    del experiment["training"]
    assert _restore_device(monkeypatch, tmp_path, experiment, mock_pipeline_dict) == [
        "cuda"
    ]


def test_restore_train_run_out_of_memory_is_resource_exhausted(
    monkeypatch, tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    def load_pipeline(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 1.00 GiB"
        )

    monkeypatch.setattr(CuvisPipeline, "load_pipeline", staticmethod(load_pipeline))
    _, service = _service()
    context = Mock()
    request = cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path=str(
            _write_trainrun(tmp_path, mock_experiment_dict, mock_pipeline_dict)
        )
    )
    service.restore_train_run(request, context)
    context.set_code.assert_called_once_with(grpc.StatusCode.RESOURCE_EXHAUSTED)
    context.set_details.assert_called_once_with(
        "Failed to restore train run: CUDA out of memory. Tried to allocate 1.00 GiB"
    )


def test_save_train_run_writes_the_calibration_block_back(
    tmp_path, mock_experiment_dict, mock_pipeline_dict
):
    sm, service = _service()
    sid = _session_with_pipeline(sm, mock_experiment_dict, mock_pipeline_dict)
    session = sm.get_session(sid)
    block = CalibrationConfig(objective="recall", target=0.9, level="frames")
    session.trainrun_config = session.trainrun_config.model_copy(
        update={"calibration": block}
    )
    out = tmp_path / "tr.yaml"
    response = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=False
        ),
        Mock(),
    )
    assert response.success is True
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert saved["calibration"] == {
        "objective": "recall",
        "target": 0.9,
        "level": "frames",
    }


@pytest.mark.parametrize(
    "round_trip", [False, True], ids=["raw yaml", "saved and restored"]
)
@pytest.mark.parametrize(
    ("training", "loss_nodes", "statistical_only"),
    [
        ({"accelerator": "cpu"}, [], True),
        ({"accelerator": "cpu"}, ["loss"], False),
        ({"accelerator": "cpu", "max_epochs": 5}, [], False),
        (None, [], True),
    ],
)
def test_a_restored_device_block_is_no_gradient_phase(
    monkeypatch,
    tmp_path,
    mock_experiment_dict,
    mock_pipeline_dict,
    training,
    loss_nodes,
    statistical_only,
    round_trip,
):
    """A device-only block is no gradient phase, also once SaveTrainRun wrote every field.

    ``round_trip`` writes the trainrun the way ``SaveTrainRun`` does (``to_dict``: every
    field with its value), so the restored block names all fields, not only the accelerator.
    """
    monkeypatch.setattr(
        CuvisPipeline, "load_pipeline", staticmethod(lambda *_a, **_k: MagicMock())
    )
    experiment = copy.deepcopy(mock_experiment_dict)
    experiment["training"] = training
    experiment["loss_nodes"] = loss_nodes
    if training is None:
        del experiment["training"]
    if round_trip:
        experiment = TrainRunConfig.from_dict(experiment).to_dict()
    sm, service = _service()
    context = Mock()
    response = service.restore_train_run(
        cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(_write_trainrun(tmp_path, experiment, mock_pipeline_dict))
        ),
        context,
    )
    context.set_code.assert_not_called()
    session = sm.get_session(response.session_id)
    assert training_service_mod._statistical_only(session) is statistical_only
