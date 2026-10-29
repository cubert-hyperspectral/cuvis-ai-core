"""The standalone calibration phase through the public ``Train`` dispatch.

Reuses the cancellation suite's tiny pipeline and in-memory datamodule: a session
without trained weights is refused with FAILED_PRECONDITION, a session whose
statistical fit completed streams the post-fit events without any trainer running.
"""

from __future__ import annotations

import grpc
import pytest
import torch

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.grpc import training_service as training_service_mod
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.training import post_fit as post_fit_mod
from cuvis_ai_schemas.training import CalibrationConfig
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


def test_unknown_trainer_type_is_still_an_invalid_argument():
    manager = SessionManager()
    session_id, service = _make_session(manager)
    ctx = _InMemoryContext()

    responses = list(service.train(_train_request(session_id, 99), ctx))

    assert responses == []
    assert ctx._code == grpc.StatusCode.INVALID_ARGUMENT


# --- the trainrun's calibration block and an out-of-memory pass ---------------------------

_BLOCK = CalibrationConfig(objective="recall", target=0.9, level="frames")


def _with_block(manager: SessionManager, session_id: str):
    session = manager.get_session(session_id)
    session.trainrun_config = session.trainrun_config.model_copy(
        update={"calibration": _BLOCK}
    )
    return session


def _spy_post_fit(monkeypatch) -> list[dict]:
    built: list[dict] = []
    real = training_service_mod.PostFitCallback

    def spy(**kwargs):
        built.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(training_service_mod, "PostFitCallback", spy)
    return built


def test_the_calibration_phase_hands_the_block_on_and_lets_out_of_memory_through(
    monkeypatch,
):
    built = _spy_post_fit(monkeypatch)
    manager = SessionManager()
    session_id, service = _make_session(manager)
    _with_block(manager, session_id)
    # The statistical phase rebuilds the session's trainrun config; the block survives it.
    list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )
    list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_CALIBRATION),
            _InMemoryContext(),
        )
    )
    assert built[-1]["calibration"] == _BLOCK
    assert built[-1]["raise_out_of_memory"] is True


def test_a_statistical_only_run_records_out_of_memory_instead(monkeypatch):
    built = _spy_post_fit(monkeypatch)
    manager = SessionManager()
    session_id, service = _make_session(manager)
    session = _with_block(manager, session_id)
    session.training_config = (
        None  # no gradient phase follows: the post-fit steps run here
    )
    responses = list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )
    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert built[-1]["calibration"] == _BLOCK
    assert built[-1]["raise_out_of_memory"] is False


def test_an_out_of_memory_calibration_phase_ends_with_resource_exhausted(monkeypatch):
    manager = SessionManager()
    session_id, service = _make_session(manager)
    list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )

    def out_of_memory(self, pipeline, datamodule, **_kwargs):
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )

    monkeypatch.setattr(post_fit_mod.PostFitCallback, "run", out_of_memory)
    context = _InMemoryContext()
    list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_CALIBRATION), context
        )
    )
    assert context._code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert (
        context._details
        == "Training failed: CUDA out of memory. Tried to allocate 2.00 GiB"
    )
