"""Unit tests for the gRPC progress-stream callbacks (no Lightning run needed)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

from cuvis_ai_core.grpc.callbacks import ProgressStreamCallback, StopTrainingCallback
from cuvis_ai_schemas.enums import ExecutionStage


class _Recorder:
    """Collect (context, losses, metrics, status) emissions."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, context, losses, metrics, status) -> None:
        self.calls.append((context, losses, metrics, status))


def _fake_trainer(**callback_metrics) -> SimpleNamespace:
    return SimpleNamespace(
        current_epoch=3,
        global_step=42,
        should_stop=False,
        callback_metrics={k: torch.tensor(v) for k, v in callback_metrics.items()},
    )


class TestStopTrainingCallback:
    """The stop event requests a graceful Lightning stop at batch boundaries."""

    def test_set_event_requests_stop_on_batch_end(self):
        event = threading.Event()
        callback = StopTrainingCallback(event)
        trainer = _fake_trainer()

        callback.on_train_batch_end(trainer, None, None, None, 0)
        assert not trainer.should_stop

        event.set()
        callback.on_train_batch_end(trainer, None, None, None, 1)
        assert trainer.should_stop

    def test_set_event_requests_stop_during_validation(self):
        event = threading.Event()
        event.set()
        callback = StopTrainingCallback(event)
        trainer = _fake_trainer()
        callback.on_validation_batch_end(trainer, None, None, None, 0)
        assert trainer.should_stop


class TestEpochLossEmissions:
    """Epoch-end emissions carry the aggregated losses (train_loss + val_loss)."""

    def test_validation_epoch_end_emits_val_loss(self):
        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer(val_loss=0.25, **{"metrics_anomaly/iou": 0.8})

        callback.on_validation_epoch_end(trainer, None)

        assert len(recorder.calls) == 1
        context, losses, metrics, status = recorder.calls[0]
        assert context.stage == ExecutionStage.VAL
        assert status == "running"
        assert losses == {"val_loss": 0.25}
        # Non-loss metrics still flow through the metrics map, not losses.
        assert "metrics_anomaly/iou" in metrics

    def test_train_epoch_end_emits_train_loss(self):
        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer(train_loss=1.5)

        callback.on_train_epoch_end(trainer, None)

        _, losses, _, status = recorder.calls[0]
        assert losses == {"train_loss": 1.5}
        assert status == "running"

    def test_per_batch_metrics_still_skip_loss_keys(self):
        """The per-batch metric path keeps filtering losses (unchanged behavior)."""
        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer(val_loss=0.25, **{"metrics_anomaly/iou": 0.8})

        metrics = callback._extract_metrics(trainer, "val")
        assert "val_loss" not in metrics
        assert "metrics_anomaly/iou" in metrics
