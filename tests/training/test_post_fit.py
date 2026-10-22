"""The post-fit lifecycle (``PostFitCallback``): restore, calibrate, evaluate, as events.

Mock nodes stand in for the trained pipeline: a source that echoes scores and the
ground-truth mask, a calibratable decider that records the call, and a metric node that
runs at the validation and test stages. The tests pin the event sequence a client relies on,
the skips (no decider, no split, a failing pass), the stop gates and the hook choice.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAUROC

from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.calibration import DeciderCalibrationStatus
from cuvis_ai_core.training.evaluation import TrainingCancelled
from cuvis_ai_core.training.post_fit import (
    CheckpointRestored,
    EvaluationStatus,
    PostFitCallback,
    TrainEvent,
    TrainHook,
    describe_event,
)
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Metric
from cuvis_ai_schemas.pipeline import PortSpec

pytestmark = pytest.mark.unit

_SCORES = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
_MASK = PortSpec(dtype=torch.bool, shape=(-1, -1, -1, -1))
_DECISIONS = PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))


class _ScoreMaskSource(Node):
    """Echo the batch's scores and mask."""

    INPUT_SPECS = {"scores": _SCORES, "mask": _MASK}
    OUTPUT_SPECS = {"scores": _SCORES, "mask": _MASK}

    def forward(self, scores, mask, **_):
        """Pass both tensors through."""
        return {"scores": scores, "mask": mask}


class _Decider(BinaryDecider):
    """Calibratable decider that moves its threshold to a known value."""

    OUTPUT_SPECS = {"decisions": _DECISIONS}

    def forward(self, logits, **_):
        """Threshold the scores."""
        return {"decisions": logits >= self.threshold}

    def calibrate(self, scores, targets):
        """Set a fixed threshold and report it."""
        self.threshold = 0.42
        self.hparams["threshold"] = 0.42
        return {
            "class": type(self).__name__,
            "threshold": {"old": 0.5, "new": 0.42},
            "f1": 0.8,
        }


class _Metrics(Node):
    """A pooled AUROC over the frames, at the validation and test stages only."""

    INPUT_SPECS = {"scores": _SCORES, "mask": _MASK}
    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=())}
    POOLED_METRIC_NAMES = frozenset({"auroc"})
    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, fail_at: ExecutionStage | None = None, **kwargs):
        super().__init__(**kwargs)
        self.auroc = BinaryAUROC()
        self.fail_at = fail_at
        self.contexts = []

    def forward(self, scores, mask, context, **_):
        """Update the pooled metric; raise at the configured stage."""
        self.contexts.append(context)
        if context.stage is self.fail_at:
            raise RuntimeError("metric blew up")
        self.auroc.update(scores.flatten(), mask.flatten().int())
        return {
            "metrics": [
                Metric(
                    name="auroc",
                    value=float(self.auroc.compute()),
                    stage=context.stage,
                    epoch=context.epoch,
                )
            ]
        }

    def pooled_metrics(self):
        """Expose the pooled object."""
        return {"auroc": self.auroc}


class _SplitsModule(pl.LightningDataModule):
    """A datamodule with fixed val and test batches; records the setup stages it saw."""

    def __init__(self, val, test):
        super().__init__()
        self.val_ds = val
        self.test_ds = test
        self.stages: list[str] = []

    def setup(self, stage=None):
        """Record the stage."""
        self.stages.append(stage)

    def val_dataloader(self):
        """The val batches."""
        return self.val_ds

    def test_dataloader(self):
        """The test batches."""
        return self.test_ds


def _batch(score: float, anomalous: bool):
    scores = torch.full((1, 4, 4, 1), score)
    mask = torch.zeros((1, 4, 4, 1), dtype=torch.bool)
    if anomalous:
        mask[0, 0, 0, 0] = True
    return {"scores": scores, "mask": mask}


def _batches():
    return [_batch(1.0, True), _batch(1.0, True), _batch(0.0, False)]


def _pipeline(*, decider: bool = True, fail_at: ExecutionStage | None = None):
    pipe = CuvisPipeline("post_fit_test")
    src = _ScoreMaskSource(name="src")
    met = _Metrics(name="met", fail_at=fail_at)
    pipe.connect(src.outputs.scores, met.inputs.scores)
    pipe.connect(src.outputs.mask, met.inputs.mask)
    dec = None
    if decider:
        dec = _Decider(name="dec")
        pipe.connect(src.outputs.scores, dec.inputs.logits)
    return pipe, met, dec


def _module(restored: bool = True):
    record = CheckpointRestored(
        restored, path="best.ckpt", epoch=3, monitor="val/auroc", score=0.9
    )
    return SimpleNamespace(load_best_checkpoint=lambda: record)


def _hooks(events: list[TrainEvent]) -> list[TrainHook]:
    return [event.hook for event in events]


def test_run_emits_the_event_sequence_of_a_full_run():
    pipe, met, dec = _pipeline()
    dm = _SplitsModule(_batches(), _batches())
    events: list[TrainEvent] = []
    callback = PostFitCallback(emit=events.append)

    result = callback.run(
        pipe, dm, module=_module(), restore=True, epoch=5, global_step=11
    )

    assert dm.stages[:2] == ["validate", "test"]
    progress = TrainHook.CALIBRATION_PROGRESS
    evaluating = TrainHook.EVALUATION_PROGRESS
    assert _hooks(events) == [
        TrainHook.CHECKPOINT_RESTORED,
        TrainHook.CALIBRATION_START,
        progress,
        progress,
        progress,
        TrainHook.CALIBRATION_END,
        TrainHook.EVALUATION_START,
        evaluating,
        evaluating,
        evaluating,
        TrainHook.EVALUATION_END,
        TrainHook.EVALUATION_START,
        evaluating,
        evaluating,
        evaluating,
        TrainHook.EVALUATION_END,
        TrainHook.POST_FIT_END,
    ]
    assert events[0].checkpoint.restored is True
    assert events[2].progress.done == 1 and events[2].split is ExecutionStage.VAL
    calibration = events[5].calibration
    assert calibration.calibrated and dec.threshold == 0.42
    assert calibration.deciders[0].status is DeciderCalibrationStatus.CALIBRATED
    val_end, test_end = events[10].evaluation, events[15].evaluation
    assert (val_end.split, val_end.status, val_end.frames) == (
        ExecutionStage.VAL,
        EvaluationStatus.EVALUATED,
        3,
    )
    assert "met/auroc" in val_end.metrics
    assert (
        test_end.split is ExecutionStage.TEST
        and test_end.status is EvaluationStatus.EVALUATED
    )
    assert result.checkpoint.restored and result.calibration is calibration
    assert set(result.evaluations) == {"val", "test"}
    # The evaluation passes carry the epoch and step the caller handed over, not the fit's.
    evaluated = [c for c in met.contexts if c.stage is not ExecutionStage.INFERENCE]
    assert {(c.epoch, c.global_step) for c in evaluated} == {(5, 11)}


def test_run_without_a_calibratable_decider_emits_one_calibration_end():
    pipe, _, _ = _pipeline(decider=False)
    events: list[TrainEvent] = []

    PostFitCallback(emit=events.append).run(
        pipe, _SplitsModule(_batches(), None), restore=False
    )

    hooks = _hooks(events)
    assert hooks[0] is TrainHook.CALIBRATION_END
    assert TrainHook.CALIBRATION_START not in hooks
    assert TrainHook.CHECKPOINT_RESTORED not in hooks
    assert events[0].calibration.applicable is False


def test_run_reports_a_missing_split_and_a_failing_pass_and_still_finishes():
    pipe, _, _ = _pipeline(fail_at=ExecutionStage.TEST)
    events: list[TrainEvent] = []

    result = PostFitCallback(emit=events.append).run(
        pipe, _SplitsModule(_batches(), _batches()), restore=False
    )

    assert result.evaluations["val"].status is EvaluationStatus.EVALUATED
    failed = result.evaluations["test"]
    assert failed.status is EvaluationStatus.FAILED
    assert failed.error == "RuntimeError: metric blew up"
    assert _hooks(events)[-1] is TrainHook.POST_FIT_END

    events.clear()
    result = PostFitCallback(emit=events.append).run(
        pipe, _SplitsModule(_batches(), None), restore=False
    )
    missing = result.evaluations["test"]
    assert missing.status is EvaluationStatus.NO_SPLIT and missing.frames == 0
    ends = [e for e in events if e.hook is TrainHook.EVALUATION_END]
    assert [e.split for e in ends] == [ExecutionStage.VAL, ExecutionStage.TEST]


def test_on_train_end_skips_everything_when_the_stop_event_is_set():
    pipe, met, _ = _pipeline()
    stop = threading.Event()
    stop.set()
    events: list[TrainEvent] = []
    callback = PostFitCallback(emit=events.append, stop_event=stop)
    trainer = SimpleNamespace(
        datamodule=_SplitsModule(_batches(), _batches()), current_epoch=1, global_step=4
    )

    callback.on_train_end(
        trainer, SimpleNamespace(pipeline=pipe, load_best_checkpoint=_module)
    )

    assert events == [] and callback.result is None and met.contexts == []


def test_on_train_end_runs_with_the_trainer_datamodule_and_module():
    pipe, met, _ = _pipeline()
    events: list[TrainEvent] = []
    callback = PostFitCallback(emit=events.append)
    dm = _SplitsModule(_batches(), _batches())
    trainer = SimpleNamespace(datamodule=dm, current_epoch=2, global_step=10)

    callback.on_train_end(trainer, SimpleNamespace(pipeline=pipe, **vars(_module())))

    assert callback.result is not None and callback.result.checkpoint.restored is True
    assert _hooks(events)[0] is TrainHook.CHECKPOINT_RESTORED
    evaluated = [c for c in met.contexts if c.stage is not ExecutionStage.INFERENCE]
    assert {(c.epoch, c.global_step) for c in evaluated} == {(2, 11)}


def test_a_stop_during_a_step_raises_training_cancelled():
    pipe, _, _ = _pipeline()
    stop = threading.Event()

    def emit(event: TrainEvent) -> None:
        if event.hook is TrainHook.EVALUATION_PROGRESS:
            stop.set()

    with pytest.raises(TrainingCancelled):
        PostFitCallback(emit=emit, stop_event=stop).run(
            pipe, _SplitsModule(_batches(), _batches()), restore=False
        )


def test_the_callback_runs_at_train_end_not_fit_end():
    """Lightning tears down (model to the CPU) before on_fit_end; on_train_end keeps the device."""
    assert "on_train_end" in PostFitCallback.__dict__
    assert "on_fit_end" not in PostFitCallback.__dict__


def test_describe_event_lines():
    restored = TrainEvent(
        TrainHook.CHECKPOINT_RESTORED,
        checkpoint=CheckpointRestored(
            True, path="b.ckpt", epoch=3, monitor="val/x", score=0.5
        ),
    )
    kept = TrainEvent(
        TrainHook.CHECKPOINT_RESTORED,
        checkpoint=CheckpointRestored(False, reason="no checkpoint was saved"),
    )
    assert (
        describe_event(restored)
        == "Best checkpoint restored (epoch 3 val/x 0.5): b.ckpt"
    )
    assert (
        describe_event(kept)
        == "Using the final epoch's weights (no checkpoint was saved)"
    )
    assert describe_event(TrainEvent(TrainHook.CALIBRATION_PROGRESS)) is None
    assert (
        describe_event(TrainEvent(TrainHook.POST_FIT_END)) == "Post-fit steps complete"
    )
