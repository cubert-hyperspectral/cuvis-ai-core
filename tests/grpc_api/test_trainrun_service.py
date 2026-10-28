"""Fast TrainRunService tests.

`tests/grpc_api/test_experiment_management.py` exercises save/restore through
the full gRPC server and is marked ``slow`` (deselected in the coverage job).
These tests drive ``TrainRunService`` directly with mock nodes so the
save-guard, yaml-parse, weights-isolation, and restore branches are covered in
the default (fast) run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.training.config import TrainRunConfig


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


def _session_with_calibration(sm, mock_experiment_dict, mock_pipeline_dict):
    """A session with a live pipeline and a calibration record set, for the writer tests."""
    import torch
    from cuvis_ai_schemas.pipeline.config import PipelineConfig

    from cuvis_ai_core.grpc.session_manager import CalibrationRecord
    from cuvis_ai_core.training.calibration import CalibrationOutcome

    sid = sm.create_session()
    session = sm.get_session(sid)
    session.trainrun_config = TrainRunConfig.from_dict(mock_experiment_dict)
    session._pipeline_config = PipelineConfig.from_dict(mock_pipeline_dict)
    node = MagicMock()
    node.name = "n"
    node.state_dict.return_value = {"w": torch.zeros(2)}
    session.pipeline = MagicMock()
    session.pipeline.nodes.return_value = [node]
    outcome = CalibrationOutcome(
        split="val",
        applicable=True,
        reports={"n": {"class": "Mock", "threshold": {"old": 0.5, "new": 0.42}}},
    )
    session.last_calibration = CalibrationRecord(
        outcome=outcome, data_config=session.trainrun_config.data
    )
    return sid, session, outcome


def test_save_train_run_writes_the_validation_summary_after_the_save(
    tmp_path, mock_experiment_dict, mock_pipeline_dict, monkeypatch
):
    """When the session holds a calibration record, SaveTrainRun calls the writer with
    the pipeline sibling path (never the trainrun path), after both files are written."""
    import cuvis_ai_core.grpc.trainrun_service as trainrun_service_mod

    sm, service = _service()
    sid, session, outcome = _session_with_calibration(
        sm, mock_experiment_dict, mock_pipeline_dict
    )

    calls: list[tuple] = []

    def _fake_writer(*args, **kwargs):
        # The sibling pipeline yaml and its weights must already exist on disk by
        # the time the writer is called.
        sibling = Path(args[0])
        assert sibling.name == "tr_pipeline.yaml"
        assert sibling.is_file()
        assert sibling.with_suffix(".pt").is_file()
        calls.append((args, kwargs))
        return sibling.with_name("validation_summary.json")

    monkeypatch.setattr(trainrun_service_mod, "write_validation_summary", _fake_writer)

    out = tmp_path / "tr.yaml"
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=True
        ),
        Mock(),
    )

    assert resp.success is True
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert not kwargs
    sibling_arg, outcome_arg, data_config_arg = args
    assert Path(sibling_arg) == out.parent / "tr_pipeline.yaml"
    assert outcome_arg is outcome
    assert data_config_arg is session.last_calibration.data_config


def test_save_train_run_skips_the_writer_without_a_calibration_record(
    tmp_path, mock_experiment_dict, mock_pipeline_dict, monkeypatch
):
    """No `last_calibration` on the session: the writer is never called."""
    import torch
    from cuvis_ai_schemas.pipeline.config import PipelineConfig

    import cuvis_ai_core.grpc.trainrun_service as trainrun_service_mod

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
    assert session.last_calibration is None

    calls: list = []
    monkeypatch.setattr(
        trainrun_service_mod,
        "write_validation_summary",
        lambda *a, **k: calls.append((a, k)),
    )

    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(tmp_path / "tr.yaml"), save_weights=True
        ),
        Mock(),
    )

    assert resp.success is True
    assert calls == []


def test_save_train_run_writer_failure_does_not_fail_the_save(
    tmp_path, mock_experiment_dict, mock_pipeline_dict, monkeypatch
):
    """A writer that raises is logged and must not take the save down with it."""
    import cuvis_ai_core.grpc.trainrun_service as trainrun_service_mod

    sm, service = _service()
    sid, session, _outcome = _session_with_calibration(
        sm, mock_experiment_dict, mock_pipeline_dict
    )

    def _boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(trainrun_service_mod, "write_validation_summary", _boom)

    # The production code logs this failure through logger.opt(exception=True).warning(...),
    # the same chained call the existing weights-save-failure path uses; a stub that
    # records the final .warning() call (ignoring .opt()'s arguments) captures it.
    warnings: list[str] = []

    class _LoggerStub:
        def opt(self, *_a, **_k):
            return self

        def warning(self, msg, *_a, **_k):
            warnings.append(str(msg))

    monkeypatch.setattr(trainrun_service_mod, "logger", _LoggerStub())

    out = tmp_path / "tr.yaml"
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=True
        ),
        Mock(),
    )

    assert resp.success is True
    assert out.exists()
    assert (out.parent / "tr_pipeline.pt").is_file()
    assert len(warnings) == 1
    assert "disk full" in warnings[0]


def test_save_train_run_writes_no_summary_when_the_weights_save_failed(
    tmp_path, mock_experiment_dict, mock_pipeline_dict, monkeypatch
):
    """A failed weights save writes no summary: an older `.pt` beside the yaml would bind
    it to weights the calibration never saw."""
    import cuvis_ai_core.grpc.trainrun_service as trainrun_service_mod

    sm, service = _service()
    sid, _session, _outcome = _session_with_calibration(
        sm, mock_experiment_dict, mock_pipeline_dict
    )
    calls: list = []
    monkeypatch.setattr(
        trainrun_service_mod,
        "write_validation_summary",
        lambda *a, **k: calls.append((a, k)),
    )

    def _no_space(*_a, **_k):
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(trainrun_service_mod.torch, "save", _no_space)

    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(tmp_path / "tr.yaml"), save_weights=True
        ),
        Mock(),
    )

    assert resp.success is True
    assert resp.weights_path == ""
    assert calls == []


def test_save_train_run_writer_runs_even_without_weights_requested(
    tmp_path, mock_experiment_dict, mock_pipeline_dict, monkeypatch
):
    """A calibration record on the session still triggers the writer call when
    `save_weights=False`; the writer itself decides whether a `.pt` exists (it must
    not, so the writer returns None and logs, which this test does not need to check,
    only that it was still invoked with the pipeline sibling path)."""
    import cuvis_ai_core.grpc.trainrun_service as trainrun_service_mod

    sm, service = _service()
    sid, session, outcome = _session_with_calibration(
        sm, mock_experiment_dict, mock_pipeline_dict
    )

    calls: list[tuple] = []
    monkeypatch.setattr(
        trainrun_service_mod,
        "write_validation_summary",
        lambda *a, **k: calls.append((a, k)),
    )

    out = tmp_path / "tr.yaml"
    resp = service.save_train_run(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=sid, trainrun_path=str(out), save_weights=False
        ),
        Mock(),
    )

    assert resp.success is True
    assert len(calls) == 1
    args, _kwargs = calls[0]
    assert Path(args[0]) == out.parent / "tr_pipeline.yaml"
    assert args[1] is outcome


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
