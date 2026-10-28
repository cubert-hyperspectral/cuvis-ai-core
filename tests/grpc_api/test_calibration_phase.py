"""The standalone calibration phase through the public ``Train`` dispatch.

Reuses the cancellation suite's tiny pipeline and in-memory datamodule: a session
without trained weights is refused with FAILED_PRECONDITION, a session whose
statistical fit completed streams the post-fit events without any trainer running.
"""

from __future__ import annotations

import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.training.config import DataConfig
from tests.grpc_api.test_train_cancellation import (
    _InMemoryContext,
    _make_session,
    _train_request,
)

pytestmark = pytest.mark.unit


def test_calibration_phase_is_refused_without_weights_and_runs_after_a_fit():
    manager = SessionManager()
    session_id, service = _make_session(manager)

    refused = _InMemoryContext()
    responses = list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_CALIBRATION), refused
        )
    )
    assert responses == []
    assert refused._code == grpc.StatusCode.FAILED_PRECONDITION
    assert "no trained weights" in refused._details

    fitted = list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )
    assert fitted[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert manager.get_session(session_id).weights_source == "fit"

    calibration = list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_CALIBRATION),
            _InMemoryContext(),
        )
    )
    hooks = [r.event.hook for r in calibration if r.HasField("event")]
    assert cuvis_ai_pb2.TRAIN_HOOK_CALIBRATION_END in hooks
    assert hooks[-1] == cuvis_ai_pb2.TRAIN_HOOK_POST_FIT_END
    assert calibration[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert calibration[-1].message.startswith("Calibration complete")


def test_calibration_with_a_request_level_data_config_leaves_the_saved_trainrun_untouched():
    """TRAINER_TYPE_CALIBRATION never calls _capture_experiment_context, so a
    TrainRequest.data override that would reach every other trainer type's saved
    trainrun must not touch this session's trainrun_config at all."""
    manager = SessionManager()
    session_id, service = _make_session(manager)
    session = manager.get_session(session_id)

    list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )

    original_trainrun = session.trainrun_config
    assert original_trainrun is not None

    override = DataConfig(
        data_module="cancelmod", batch_size=4, params={"n": 16, "dim": 4}
    )
    request = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_CALIBRATION,
        data=override.to_proto(),
    )

    responses = list(service.train(request, _InMemoryContext()))

    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert session.trainrun_config is original_trainrun


def test_unknown_trainer_type_is_still_an_invalid_argument():
    manager = SessionManager()
    session_id, service = _make_session(manager)
    ctx = _InMemoryContext()

    responses = list(service.train(_train_request(session_id, 99), ctx))

    assert responses == []
    assert ctx._code == grpc.StatusCode.INVALID_ARGUMENT
