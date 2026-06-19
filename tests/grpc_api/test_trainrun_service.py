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
    # The yaml still saved; only the optional weights dump failed.
    assert resp.success is True
    assert out.exists()
    assert any(
        "weights save failed" in str(c.args[0]) for c in ctx.set_details.mock_calls
    )


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
