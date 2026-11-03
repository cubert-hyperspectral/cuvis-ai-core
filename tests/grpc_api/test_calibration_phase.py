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
from cuvis_ai_core.training.config import TrainingConfig
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


def _statistically_fitted(manager: SessionManager) -> tuple[str, object]:
    """A session whose statistical fit completed, ready for the calibration phase."""
    session_id, service = _make_session(manager)
    list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )
    return session_id, service


def _calibrate(service, session_id: str) -> tuple[_InMemoryContext, list]:
    context = _InMemoryContext()
    responses = list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_CALIBRATION), context
        )
    )
    return context, responses


@pytest.mark.parametrize(
    ("trainer_type", "trainer"),
    [
        (
            cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            training_service_mod.StatisticalTrainer,
        ),
        (cuvis_ai_pb2.TRAINER_TYPE_GRADIENT, training_service_mod.GradientTrainer),
    ],
)
def test_an_out_of_memory_fit_ends_with_resource_exhausted(
    monkeypatch, trainer_type, trainer
):
    def out_of_memory(self, *_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )

    monkeypatch.setattr(trainer, "fit", out_of_memory)
    manager = SessionManager()
    session_id, service = _make_session(manager)
    context = _InMemoryContext()
    list(service.train(_train_request(session_id, trainer_type), context))
    assert context._code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert (
        context._details
        == "Training failed: CUDA out of memory. Tried to allocate 2.00 GiB"
    )


def test_a_stop_landing_with_out_of_memory_keeps_resource_exhausted(monkeypatch):
    manager = SessionManager()
    session_id, service = _statistically_fitted(manager)
    session = manager.get_session(session_id)

    def stop_then_out_of_memory(self, pipeline, datamodule, **_kwargs):
        # The client drops the stream at the moment the allocator fails.
        session.stop_event.set()
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )

    monkeypatch.setattr(post_fit_mod.PostFitCallback, "run", stop_then_out_of_memory)
    context, responses = _calibrate(service, session_id)
    assert context._code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert all(r.status != cuvis_ai_pb2.TRAIN_STATUS_CANCELLED for r in responses)


def test_a_stop_landing_with_another_error_is_still_a_cancel(monkeypatch):
    manager = SessionManager()
    session_id, service = _statistically_fitted(manager)
    session = manager.get_session(session_id)

    def stop_then_fail(self, pipeline, datamodule, **_kwargs):
        session.stop_event.set()
        raise RuntimeError("the loader was closed under the pass")

    monkeypatch.setattr(post_fit_mod.PostFitCallback, "run", stop_then_fail)
    context, responses = _calibrate(service, session_id)
    assert context._code is None
    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED


def test_a_value_error_raised_from_out_of_memory_is_resource_exhausted(monkeypatch):
    manager = SessionManager()
    session_id, service = _statistically_fitted(manager)

    def wrapped(self, pipeline, datamodule, **_kwargs):
        raise ValueError("Node 'det' cannot run on this device") from (
            torch.cuda.OutOfMemoryError("CUDA out of memory.")
        )

    monkeypatch.setattr(post_fit_mod.PostFitCallback, "run", wrapped)
    context, _ = _calibrate(service, session_id)
    assert context._code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert context._details == "Training failed: Node 'det' cannot run on this device"


def test_a_plain_value_error_in_a_pass_is_still_an_invalid_argument(monkeypatch):
    manager = SessionManager()
    session_id, service = _statistically_fitted(manager)

    def invalid(self, pipeline, datamodule, **_kwargs):
        raise ValueError("Node 'det' was given a bad shape")

    monkeypatch.setattr(post_fit_mod.PostFitCallback, "run", invalid)
    context, _ = _calibrate(service, session_id)
    assert context._code == grpc.StatusCode.INVALID_ARGUMENT
    assert context._details == "Node 'det' was given a bad shape"


def _device_only(manager: SessionManager, session_id: str, *, loss_nodes: list[str]):
    """Give the session the training block a client writes to restore on the CPU."""
    session = manager.get_session(session_id)
    session.training_config = TrainingConfig.model_validate({"accelerator": "cpu"})
    session.trainrun_config = session.trainrun_config.model_copy(
        update={"training": session.training_config, "loss_nodes": loss_nodes}
    )


def test_a_training_block_that_only_picks_the_device_keeps_the_statistical_post_fit(
    monkeypatch,
):
    built = _spy_post_fit(monkeypatch)
    manager = SessionManager()
    session_id, service = _make_session(manager)
    _device_only(manager, session_id, loss_nodes=[])
    responses = list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )
    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert len(built) == 1


def test_a_device_block_beside_loss_nodes_still_waits_for_the_gradient_phase(
    monkeypatch,
):
    built = _spy_post_fit(monkeypatch)
    manager = SessionManager()
    session_id, service = _make_session(manager)
    _device_only(manager, session_id, loss_nodes=["mse_loss"])
    list(
        service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
    )
    assert built == []
