"""``set_train_run_config``: the run boundary that clears per-run session state.

Mock-based, no built pipeline: this RPC only writes config fields onto the session
state, so a ``Mock()`` stands in for the attached pipeline (the precondition checks
only ``is not None``).
"""

from __future__ import annotations

from unittest.mock import Mock

from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.session_manager import CalibrationRecord, SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.training.calibration import CalibrationOutcome
from cuvis_ai_core.training.config import DataConfig, TrainRunConfig


def _service():
    sm = SessionManager()
    return sm, PipelineService(sm)


def test_set_train_run_config_clears_last_calibration():
    """A fresh run must not inherit a calibration outcome from the previous one."""
    sm, service = _service()
    session_id = sm.create_session(pipeline=Mock())
    session = sm.get_session(session_id)
    session.last_calibration = CalibrationRecord(
        outcome=CalibrationOutcome(split="val", applicable=True),
        data_config=DataConfig(),
    )

    trainrun = TrainRunConfig(name="fresh_run", pipeline=None, data=DataConfig())
    response = service.set_train_run_config(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id, config=trainrun.to_proto()
        ),
        Mock(),
    )

    assert response.success
    assert sm.get_session(session_id).last_calibration is None


def test_set_train_run_config_failure_paths_leave_last_calibration_untouched():
    """A refused SetTrainRunConfig (no pipeline, a pipeline-carrying trainrun) is not a
    run boundary: it must not clear a calibration outcome that is still valid."""
    sm, service = _service()
    record = CalibrationRecord(
        outcome=CalibrationOutcome(split="val", applicable=True),
        data_config=DataConfig(),
    )

    # No pipeline attached: FAILED_PRECONDITION, no state change.
    session_id = sm.create_session()
    session = sm.get_session(session_id)
    session.last_calibration = record
    trainrun = TrainRunConfig(name="run", pipeline=None, data=DataConfig())
    ctx = Mock()
    response = service.set_train_run_config(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id, config=trainrun.to_proto()
        ),
        ctx,
    )
    assert not response.success
    assert sm.get_session(session_id).last_calibration is record

    # A trainrun carrying a pipeline reference: also refused, also no state change.
    session_id2 = sm.create_session(pipeline=Mock())
    session2 = sm.get_session(session_id2)
    session2.last_calibration = record
    trainrun_with_pipeline = TrainRunConfig(
        name="run", pipeline="some_pipeline.yaml", data=DataConfig()
    )
    response2 = service.set_train_run_config(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id2, config=trainrun_with_pipeline.to_proto()
        ),
        Mock(),
    )
    assert not response2.success
    assert sm.get_session(session_id2).last_calibration is record
