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


class _Loader:
    """A loader the way Lightning hands it to a callback: a sized ``dataset`` attribute."""

    def __init__(self, frames: int) -> None:
        self.dataset = range(frames)


def _fake_trainer(**callback_metrics) -> SimpleNamespace:
    return SimpleNamespace(
        current_epoch=3,
        global_step=42,
        max_epochs=20,
        should_stop=False,
        sanity_checking=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        train_dataloader=None,
        val_dataloaders=None,
        datamodule=None,
        callback_metrics={k: torch.tensor(v) for k, v in callback_metrics.items()},
    )


def _fit_trainer(
    train_frames: int = 14, val_frames: int = 6, max_epochs: int | None = 2, **limits
) -> SimpleNamespace:
    """A trainer at the start of a fit whose loaders expose their frame counts."""
    trainer = _fake_trainer()
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.max_epochs = max_epochs
    trainer.train_dataloader = _Loader(train_frames)
    trainer.val_dataloaders = _Loader(val_frames)
    for name, value in limits.items():
        setattr(trainer, name, value)
    return trainer


def _drive_epoch(callback, trainer, train_batches, val_batches) -> None:
    """Fire one epoch's hooks in Lightning's order with batches of the given frame counts."""
    callback.on_train_epoch_start(trainer, None)
    for idx, frames in enumerate(train_batches):
        trainer.global_step += 1
        callback.on_train_batch_end(
            trainer, None, {"loss": torch.tensor(1.0)}, torch.zeros(frames, 2), idx
        )
    callback.on_validation_epoch_start(trainer, None)
    for idx, frames in enumerate(val_batches):
        callback.on_validation_batch_end(
            trainer, None, None, torch.zeros(frames, 2), idx
        )
    callback.on_validation_epoch_end(trainer, None)
    callback.on_train_epoch_end(trainer, None)


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
        from cuvis_ai_core.training.post_fit import EpochProgress, TrainHook

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)

        callback.on_fit_start(
            _fit_trainer(max_epochs=20), self._module(calibrated=False)
        )
        callback.on_fit_start(
            _fit_trainer(max_epochs=20), self._module(calibrated=True)
        )

        pending, settled = recorder.events
        assert (
            pending.hook is TrainHook.FIT_START and pending.calibration_pending is True
        )
        assert (
            settled.hook is TrainHook.FIT_START and settled.calibration_pending is False
        )
        assert pending.epochs == EpochProgress(1, 20)
        assert pending.progress is None
        assert recorder.calls[0][3] == "running"

    def test_epoch_hooks_and_batches_carry_their_events(self):
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
            TrainHook.TRAIN_PROGRESS,
            TrainHook.VALIDATION_END,
            TrainHook.EPOCH_END,
            TrainHook.FIT_END,
        ]
        # The batch event rides on the same response as the batch losses.
        assert recorder.calls[0][1] == {"total": 1.0}
        # The last epoch done is not the end of the run: the post-fit steps follow.
        assert recorder.calls[-1][3] == "running"
        assert recorder.calls[1][2] == pytest.approx({"metrics_anomaly/auroc": 0.8})

    def test_the_fit_reports_every_hook_with_its_epoch_and_frame_progress(self):
        """Two epochs over 14 train frames in batches of 4, 4, 4, 2 and 6 validation frames
        in batches of 4, 2: every response carries an event, the frame counts reach the
        split totals, the epoch pair reads 1/2 then 2/2 and FIT_END reports 2 of 2."""
        from cuvis_ai_core.training.post_fit import (
            EpochProgress,
            FrameProgress,
            TrainHook,
        )

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fit_trainer(train_frames=14, val_frames=6, max_epochs=2)

        callback.on_fit_start(trainer, self._module(calibrated=True))
        for epoch in range(2):
            trainer.current_epoch = epoch
            _drive_epoch(callback, trainer, [4, 4, 4, 2], [4, 2])
        callback.on_train_end(trainer, None)

        assert all(event is not None for event in recorder.events)
        per_epoch = [
            TrainHook.EPOCH_START,
            *([TrainHook.TRAIN_PROGRESS] * 4),
            TrainHook.VALIDATION_START,
            *([TrainHook.VALIDATION_PROGRESS] * 2),
            TrainHook.VALIDATION_END,
            TrainHook.EPOCH_END,
        ]
        hooks = [event.hook for event in recorder.events]
        assert hooks == [TrainHook.FIT_START, *per_epoch, *per_epoch, TrainHook.FIT_END]

        first, second = recorder.events[1:11], recorder.events[11:21]
        assert first[0].progress == FrameProgress(0, 14)
        assert [e.progress for e in first[1:5]] == [
            FrameProgress(4, 14),
            FrameProgress(8, 14),
            FrameProgress(12, 14),
            FrameProgress(14, 14),
        ]
        assert first[5].progress == FrameProgress(0, 6)
        assert [e.progress for e in first[6:8]] == [
            FrameProgress(4, 6),
            FrameProgress(6, 6),
        ]
        assert all(e.split is ExecutionStage.VAL for e in first[5:9])
        assert all(e.split is None for e in first[:5] + first[9:])
        # The epoch pair rides on every hook but the validation batches.
        with_epochs = [e for e in first if e.hook is not TrainHook.VALIDATION_PROGRESS]
        assert [e.epochs for e in with_epochs] == [EpochProgress(1, 2)] * 8
        assert all(e.epochs is None for e in first[6:8])
        assert second[0].epochs == EpochProgress(2, 2)
        assert second[1].progress == FrameProgress(
            4, 14
        )  # the count restarts per epoch
        assert recorder.events[-1].epochs == EpochProgress(2, 2)
        # The batch losses still travel on the batch response.
        assert recorder.calls[2][1] == {"total": 1.0}
        assert all(call[3] == "running" for call in recorder.calls)

    def test_an_early_stop_reports_the_completed_epochs_on_fit_end(self):
        from cuvis_ai_core.training.post_fit import EpochProgress

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fit_trainer(max_epochs=20)

        callback.on_fit_start(trainer, self._module(calibrated=True))
        for epoch in range(3):
            trainer.current_epoch = epoch
            _drive_epoch(callback, trainer, [4], [2])
        trainer.should_stop = True
        callback.on_train_end(trainer, None)

        assert recorder.events[-1].epochs == EpochProgress(3, 20)

    def test_an_unlimited_run_reports_no_epoch_total(self):
        from cuvis_ai_core.training.post_fit import EpochProgress

        for max_epochs in (-1, None):
            recorder = _Recorder()
            callback = ProgressStreamCallback(recorder)
            trainer = _fit_trainer(max_epochs=max_epochs)

            callback.on_fit_start(trainer, self._module(calibrated=True))
            trainer.current_epoch = 0
            _drive_epoch(callback, trainer, [4], [2])
            callback.on_train_end(trainer, None)

            assert recorder.events[0].epochs == EpochProgress(1, 0)
            assert recorder.events[-1].epochs == EpochProgress(1, 0)

    def test_a_loader_without_a_sized_dataset_reports_an_unknown_total(self):
        from cuvis_ai_core.training.post_fit import FrameProgress, TrainHook

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fit_trainer()
        trainer.train_dataloader = [torch.zeros(4, 2), torch.zeros(4, 2)]  # no .dataset
        trainer.val_dataloaders = SimpleNamespace(dataset=object())  # no len()

        trainer.current_epoch = 0
        _drive_epoch(callback, trainer, [4, 4], [2])

        by_hook: dict = {}
        for event in recorder.events:
            by_hook.setdefault(event.hook, []).append(event.progress)
        assert by_hook[TrainHook.EPOCH_START] == [FrameProgress(0, 0)]
        assert by_hook[TrainHook.TRAIN_PROGRESS] == [
            FrameProgress(4, 0),
            FrameProgress(8, 0),
        ]
        assert by_hook[TrainHook.VALIDATION_START] == [FrameProgress(0, 0)]
        assert by_hook[TrainHook.VALIDATION_PROGRESS] == [FrameProgress(2, 0)]

    def test_a_limited_pass_reports_an_unknown_total(self):
        """limit_val_batches = 2 stops the pass before the split's end, so the total is 0
        and the count stays below the split length; a fractional train limit the same."""
        from cuvis_ai_core.training.post_fit import FrameProgress, TrainHook

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fit_trainer(
            train_frames=124,
            val_frames=48,
            limit_train_batches=0.5,
            limit_val_batches=2,
        )

        trainer.current_epoch = 0
        _drive_epoch(callback, trainer, [4, 4], [4, 4])

        progress = {
            hook: [e.progress for e in recorder.events if e.hook is hook]
            for hook in (
                TrainHook.EPOCH_START,
                TrainHook.TRAIN_PROGRESS,
                TrainHook.VALIDATION_START,
                TrainHook.VALIDATION_PROGRESS,
            )
        }
        assert progress[TrainHook.EPOCH_START] == [FrameProgress(0, 0)]
        assert progress[TrainHook.TRAIN_PROGRESS][-1] == FrameProgress(8, 0)
        assert progress[TrainHook.VALIDATION_START] == [FrameProgress(0, 0)]
        assert progress[TrainHook.VALIDATION_PROGRESS][-1] == FrameProgress(8, 0)
        assert progress[TrainHook.VALIDATION_PROGRESS][-1].done < 48

    def test_the_frame_total_is_read_from_the_fit_loader_at_the_epoch_start(self):
        """At on_fit_start Lightning has not built its loaders; the total is read when the
        epoch starts, from the loader the trainer runs, never through the datamodule's
        loader factory."""
        from cuvis_ai_core.training.post_fit import FrameProgress

        def _never(*_args, **_kwargs):
            raise AssertionError("the datamodule's loader factory was called")

        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fit_trainer()
        trainer.train_dataloader = None
        trainer.val_dataloaders = None
        trainer.datamodule = SimpleNamespace(
            train_dataloader=_never, val_dataloader=_never
        )

        callback.on_fit_start(trainer, self._module(calibrated=True))
        assert recorder.events[0].progress is None

        trainer.train_dataloader = _Loader(5)
        trainer.val_dataloaders = _Loader(3)
        trainer.current_epoch = 0
        _drive_epoch(callback, trainer, [5], [3])

        assert recorder.events[1].progress == FrameProgress(0, 5)
        assert recorder.events[3].progress == FrameProgress(0, 3)

    def test_validation_hooks_stay_quiet_during_the_sanity_check(self):
        recorder = _Recorder()
        callback = ProgressStreamCallback(recorder)
        trainer = _fake_trainer(val_loss=0.25)
        trainer.sanity_checking = True

        callback.on_validation_epoch_start(trainer, None)
        callback.on_validation_batch_end(trainer, None, None, None, 0)
        callback.on_validation_epoch_end(trainer, None)

        assert recorder.calls == [] and recorder.events == []

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


class TestLightningFit:
    """A real Lightning fit drives the callback: the hook order, the frame totals read from
    the trainer's loaders and the epoch pair hold outside the fake trainer."""

    @staticmethod
    def _pipeline():
        from torch import nn

        from cuvis_ai_core.node.node import Node
        from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
        from cuvis_ai_schemas.pipeline import PortSpec

        vec = PortSpec(dtype=torch.float32, shape=(-1,))

        class Source(Node):
            INPUT_SPECS: dict[str, PortSpec] = {}
            OUTPUT_SPECS = {"out": vec}

            def forward(self, **inputs):
                return {"out": torch.ones(2)}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class Scale(Node):
            INPUT_SPECS = {"x": vec}
            OUTPUT_SPECS = {"out": vec}

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.weight = nn.Parameter(torch.ones(1))

            def forward(self, x, **inputs):
                return {"out": x * self.weight}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class Loss(Node):
            INPUT_SPECS = {"value": vec}
            OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}
            EXECUTION_STAGES = {ExecutionStage.TRAIN, ExecutionStage.VAL}

            def forward(self, value, **inputs):
                return {"loss": value.sum()}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        pipeline = CuvisPipeline("fit")
        source, scale, loss = (
            Source(name="source"),
            Scale(name="scale"),
            Loss(name="loss"),
        )
        pipeline.connect(source.outputs.out, scale.inputs.x)
        pipeline.connect(scale.outputs.out, loss.inputs.value)
        return pipeline, loss

    @staticmethod
    def _datamodule(train_frames: int, val_frames: int):
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader, Dataset

        class Frames(Dataset):
            def __init__(self, frames: int) -> None:
                self.frames = frames

            def __len__(self) -> int:
                return self.frames

            def __getitem__(self, index: int) -> dict:
                return {"x": torch.full((2,), float(index))}

        class FitDataModule(pl.LightningDataModule):
            def train_dataloader(self):
                return DataLoader(Frames(train_frames), batch_size=4)

            def val_dataloader(self):
                return DataLoader(Frames(val_frames), batch_size=4)

        return FitDataModule()

    def _fit(self, tmp_path, recorder, **config):
        from cuvis_ai_core.training.config import TrainingConfig
        from cuvis_ai_core.training.trainers import GradientTrainer

        pipeline, loss = self._pipeline()
        GradientTrainer(
            pipeline=pipeline,
            datamodule=self._datamodule(14, 6),
            loss_nodes=[loss],
            training_config=TrainingConfig(
                max_epochs=2,
                accelerator="cpu",
                enable_progress_bar=False,
                default_root_dir=str(tmp_path),
                log_every_n_steps=1,
                **config,
            ),
            callbacks=[ProgressStreamCallback(recorder)],
        ).fit()

    def test_a_lightning_fit_reports_the_hooks_with_frame_and_epoch_progress(
        self, tmp_path
    ):
        from cuvis_ai_core.training.post_fit import (
            EpochProgress,
            FrameProgress,
            TrainHook,
        )

        recorder = _Recorder()
        self._fit(tmp_path, recorder)

        assert all(event is not None for event in recorder.events)
        per_epoch = [
            TrainHook.EPOCH_START,
            *([TrainHook.TRAIN_PROGRESS] * 4),
            TrainHook.VALIDATION_START,
            *([TrainHook.VALIDATION_PROGRESS] * 2),
            TrainHook.VALIDATION_END,
            TrainHook.EPOCH_END,
        ]
        hooks = [event.hook for event in recorder.events]
        assert hooks == [TrainHook.FIT_START, *per_epoch, *per_epoch, TrainHook.FIT_END]

        first = recorder.events[1:11]
        assert [e.progress for e in first[:5]] == [
            FrameProgress(0, 14),
            FrameProgress(4, 14),
            FrameProgress(8, 14),
            FrameProgress(12, 14),
            FrameProgress(14, 14),
        ]
        assert [e.progress for e in first[5:8]] == [
            FrameProgress(0, 6),
            FrameProgress(4, 6),
            FrameProgress(6, 6),
        ]
        assert recorder.events[0].epochs == EpochProgress(1, 2)
        assert recorder.events[11].epochs == EpochProgress(2, 2)
        assert recorder.events[-1].epochs == EpochProgress(2, 2)
        # The batch responses carry the batch loss and a rising global step.
        batch_calls = [
            call
            for call, event in zip(recorder.calls, recorder.events, strict=True)
            if event.hook is TrainHook.TRAIN_PROGRESS
        ]
        assert all("total" in call[1] for call in batch_calls)
        assert [call[0].global_step for call in batch_calls] == list(range(1, 9))

    def test_a_limited_validation_pass_reports_no_total(self, tmp_path):
        from cuvis_ai_core.training.post_fit import FrameProgress, TrainHook

        recorder = _Recorder()
        self._fit(tmp_path, recorder, limit_val_batches=1)

        starts = [
            e.progress for e in recorder.events if e.hook is TrainHook.VALIDATION_START
        ]
        batches = [
            e.progress
            for e in recorder.events
            if e.hook is TrainHook.VALIDATION_PROGRESS
        ]
        assert starts == [FrameProgress(0, 0)] * 2
        assert batches == [FrameProgress(4, 0)] * 2
