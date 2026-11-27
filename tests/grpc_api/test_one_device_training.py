"""Gradient training through the server runs on one device.

Lightning trains on several devices by starting the runtime's command once more per extra
device, and such a copy serves nothing, so the run would die. The server pins an unset or
"auto" ``devices`` to 1 and refuses a config that asks for more.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import grpc
import pytest
import pytorch_lightning as pl

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.training_service import on_one_device
from cuvis_ai_core.training.config import TrainingConfig
from tests.grpc_api.test_train_cancellation import _make_session


@pytest.mark.parametrize("devices", [None, "auto"])
def test_unset_or_auto_devices_train_on_one(devices):
    config = TrainingConfig(devices=devices)

    assert on_one_device(config).devices == 1
    assert config.devices == devices


@pytest.mark.parametrize("devices", [1, "1", " 1 ", "0,", "1,"])
def test_one_device_by_count_or_by_index_passes_unchanged(devices):
    config = TrainingConfig(devices=devices)

    assert on_one_device(config) == config


@pytest.mark.parametrize("devices", ["cuda:0", "2-4", "garbage"])
def test_devices_text_lightning_cannot_read_passes_for_lightning_to_refuse(devices):
    config = TrainingConfig(devices=devices)

    assert on_one_device(config) == config


@pytest.mark.parametrize("devices", [2, 8, -1, "2", "-1", "0,1", "0, 1"])
def test_more_than_one_device_is_refused(devices):
    with pytest.raises(ValueError, match="one device"):
        on_one_device(TrainingConfig(devices=devices))


def _train_gradient(service, session_id: str) -> tuple[list, MagicMock]:
    context = MagicMock()
    responses = list(
        service.train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT
            ),
            context,
        )
    )
    return responses, context


def test_a_multi_device_config_answers_invalid_argument_before_any_fit():
    manager = SessionManager()
    session_id, service = _make_session(manager)
    session = manager.get_session(session_id)
    session.training_config = session.training_config.model_copy(update={"devices": 2})
    service._train_gradient = MagicMock()
    try:
        responses, context = _train_gradient(service, session_id)
    finally:
        manager.close_session(session_id)

    context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
    assert "one device" in context.set_details.call_args.args[0]
    assert responses == []
    service._train_gradient.assert_not_called()


@pytest.mark.parametrize("devices", [None, "auto"])
def test_the_fit_runs_on_one_device_and_the_session_keeps_the_clients_value(
    monkeypatch, devices
):
    recorded: dict = {}
    real_trainer = pl.Trainer

    def recording_trainer(**kwargs):
        recorded.update(kwargs)
        return real_trainer(**kwargs)

    monkeypatch.setattr(pl, "Trainer", recording_trainer)
    manager = SessionManager()
    session_id, service = _make_session(manager)
    session = manager.get_session(session_id)
    session.training_config = session.training_config.model_copy(
        update={"devices": devices, "max_epochs": 1}
    )
    try:
        responses, context = _train_gradient(service, session_id)
        kept = manager.get_session(session_id).training_config.devices
    finally:
        manager.close_session(session_id)

    context.set_code.assert_not_called()
    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert recorded["devices"] == 1
    assert kept == devices
