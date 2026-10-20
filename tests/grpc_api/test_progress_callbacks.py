"""Unit tests for the gRPC progress-stream callbacks (no Lightning run needed)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from cuvis_ai_core.grpc.callbacks import ProgressStreamCallback, StopTrainingCallback
from cuvis_ai_schemas.enums import ExecutionStage


class _Recorder:
    """Collect (context, losses, metrics, status) emissions and their events."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.events: list = []

    def __call__(self, context, losses, metrics, status, event=None) -> None:
        self.calls.append((context, losses, metrics, status))
        self.events.append(event)


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


class TestLifecycleEvents:
    """The lifecycle hooks carry a typed event; the per-batch emissions carry none."""

    @staticmethod
    def _module(calibrated: bool):
        from types import SimpleNamespace

        from cuvis_ai_core.deciders.base_decider import BinaryDecider

        class _Decider(BinaryDecider):
            def forward(self, logits, **_):
                return {"decisions": logits > 0}

            def calibrate(self, scores, targets):
                return {"threshold": {"old": 0.5, "new": 0.4}}

            @property
            def is_calibrated(self) -> bool:
                return calibrated

        pipeline = SimpleNamespace(nodes=lambda: [_Decider(name="dec")])
        return SimpleNamespace(pipeline=pipeline)

    def test_fit_start_reports_a_pending_calibration(self):
        from cuvis_ai_core.training.post_fit import TrainHook

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)

        callback.on_fit_start(_fake_trainer(), self._module(calibrated=False))
        callback.on_fit_start(_fake_trainer(), self._module(calibrated=True))

        pending, settled = recorder.events
        assert (
            pending.hook is TrainHook.FIT_START and pending.calibration_pending is True
        )
        assert (
            settled.hook is TrainHook.FIT_START and settled.calibration_pending is False
        )
        assert recorder.calls[0][3] == "running"

    def test_epoch_hooks_carry_their_events_and_batches_carry_none(self):
        from cuvis_ai_core.training.post_fit import TrainHook

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer(val_loss=0.25, **{"metrics_anomaly/auroc": 0.8})

        callback.on_train_batch_end(trainer, None, {"loss": torch.tensor(1.0)}, None, 0)
        callback.on_validation_epoch_end(trainer, None)
        callback.on_train_epoch_end(trainer, None)
        callback.on_train_end(trainer, None)

        hooks = [event.hook if event is not None else None for event in recorder.events]
        assert hooks == [
            None,
            TrainHook.VALIDATION_END,
            TrainHook.EPOCH_END,
            TrainHook.FIT_END,
        ]
        # The last epoch done is not the end of the run: the post-fit steps follow.
        assert recorder.calls[-1][3] == "running"
        assert recorder.calls[1][2] == pytest.approx({"metrics_anomaly/auroc": 0.8})

    def test_validation_hooks_stay_quiet_during_the_sanity_check(self):
        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer(val_loss=0.25)
        trainer.sanity_checking = True

        callback.on_validation_epoch_start(trainer, None)
        callback.on_validation_batch_end(trainer, None, None, None, 0)
        callback.on_validation_epoch_end(trainer, None)

        assert recorder.calls == []

    def test_every_stage_prefix_is_stripped_from_metric_keys(self):
        """A loss node without 'loss' in its name logs 'val/bce'; the metrics map names the
        metric and the context names the stage, whatever stage the emission is for."""
        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer(**{"val/bce": 0.3, "test_iou": 0.6, "metrics/ap": 0.9})

        metrics = callback._extract_metrics(trainer, "train")

        assert metrics == pytest.approx({"bce": 0.3, "iou": 0.6, "metrics/ap": 0.9})

    def test_last_epoch_and_step_follow_the_emissions(self):
        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer()

        callback.on_train_epoch_start(trainer, None)
        callback.on_train_batch_end(trainer, None, None, None, 2)

        assert (callback.last_epoch, callback.last_global_step) == (3, 42)
