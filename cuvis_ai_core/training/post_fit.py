"""The post-fit lifecycle: restore the best checkpoint, calibrate, evaluate, as events.

After any fit the same three steps run on the weights that are about to be saved: the best
checkpoint comes back when a checkpoint callback produced one, the calibratable deciders
are re-fitted on the validation split when it exists, and the validation and test splits
are evaluated when they exist. Nothing in a trainrun declares or disables the steps; each
runs when its inputs are there, and every step reports itself as a typed
:class:`TrainEvent` so a client (the gRPC Train stream, the CLI log) shows what happened
without parsing text.

:class:`PostFitCallback` is the one home of the steps. The gradient trainer runs it at
Lightning's ``on_train_end``: the last hook before the trainer's teardown moves the model to
the CPU, after the last checkpoint save. The statistical trainer and the standalone
calibration phase call :meth:`PostFitCallback.run` directly. A cancelled fit (the stop
event set, the loop ended through ``trainer.should_stop``) still reaches ``on_train_end``;
the callback then does nothing, so a cancelled run never calibrates.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from pytorch_lightning import Callback, LightningModule, Trainer

from cuvis_ai_core.training.calibration import (
    CalibrationOutcome,
    calibratable_deciders,
    calibrate_pipeline_deciders,
)
from cuvis_ai_core.training.evaluation import (
    TrainingCancelled,
    default_metric_nodes,
    evaluate_split,
)
from cuvis_ai_schemas.enums import ExecutionStage

if TYPE_CHECKING:
    from cuvis_ai_core.node.node import Node
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


class TrainHook(Enum):
    """The lifecycle hook an event reports. The names mirror the proto ``TrainHook``."""

    FIT_START = "fit_start"
    EPOCH_END = "epoch_end"
    VALIDATION_END = "validation_end"
    FIT_END = "fit_end"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    CALIBRATION_START = "calibration_start"
    CALIBRATION_PROGRESS = "calibration_progress"
    CALIBRATION_END = "calibration_end"
    EVALUATION_START = "evaluation_start"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_END = "evaluation_end"
    POST_FIT_END = "post_fit_end"


class EvaluationStatus(Enum):
    """How one evaluation pass ended. The names mirror the proto ``EvaluationStatus``."""

    EVALUATED = "evaluated"
    NO_SPLIT = "no_split"
    FAILED = "failed"


@dataclass(frozen=True)
class FrameProgress:
    """Frames seen so far over a split and the split's total (0 when unknown)."""

    done: int
    total: int


@dataclass(frozen=True)
class CheckpointRestored:
    """What the restore of the best checkpoint did."""

    restored: bool
    path: str = ""
    epoch: int = 0
    monitor: str = ""
    score: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class EvaluationOutcome:
    """One evaluation pass over a split."""

    split: ExecutionStage
    status: EvaluationStatus
    frames: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class TrainEvent:
    """One lifecycle event with the data of its hook."""

    hook: TrainHook
    split: ExecutionStage | None = None
    progress: FrameProgress | None = None
    checkpoint: CheckpointRestored | None = None
    calibration: CalibrationOutcome | None = None
    evaluation: EvaluationOutcome | None = None
    calibration_pending: bool = False


EmitFn = Callable[[TrainEvent], None]


@dataclass(frozen=True)
class PostFitResult:
    """Everything the post-fit steps produced, for the caller that saves the run."""

    checkpoint: CheckpointRestored | None
    calibration: CalibrationOutcome
    evaluations: dict[str, EvaluationOutcome]


_SPLITS: tuple[tuple[str, ExecutionStage], ...] = (
    ("val", ExecutionStage.VAL),
    ("test", ExecutionStage.TEST),
)


def describe_event(event: TrainEvent) -> str | None:
    """One log line for an event, or ``None`` for the per-batch progress hooks."""
    hook = event.hook
    if hook is TrainHook.CHECKPOINT_RESTORED and event.checkpoint is not None:
        restored = event.checkpoint
        if restored.restored:
            score = (
                f" {restored.monitor} {restored.score:.4g}"
                if restored.score is not None
                else ""
            )
            return f"Best checkpoint restored (epoch {restored.epoch}{score}): {restored.path}"
        return f"Using the final epoch's weights ({restored.reason})"
    if hook is TrainHook.CALIBRATION_START:
        return f"Calibrating decider thresholds on {_split_name(event.split)}"
    if hook is TrainHook.CALIBRATION_END and event.calibration is not None:
        outcome = event.calibration
        return f"Calibration: {outcome.summary()}" if outcome.applicable else None
    if hook is TrainHook.EVALUATION_START:
        return f"Evaluating {_split_name(event.split)}"
    if hook is TrainHook.EVALUATION_END and event.evaluation is not None:
        result = event.evaluation
        name = _split_name(result.split)
        if result.status is EvaluationStatus.EVALUATED:
            values = ", ".join(
                f"{key} {value:.4g}" for key, value in sorted(result.metrics.items())
            )
            return f"Evaluated {name} ({result.frames} frames): {values or 'no metric node'}"
        if result.status is EvaluationStatus.NO_SPLIT:
            return f"No {name} split; evaluation skipped"
        return f"Evaluating {name} failed: {result.error}; the run is still saved"
    if hook is TrainHook.POST_FIT_END:
        return "Post-fit steps complete"
    return None


def _split_name(split: ExecutionStage | None) -> str:
    return split.value if split is not None else "split"


def log_event(event: TrainEvent) -> None:
    """The default sink: one info line per event, nothing for progress."""
    line = describe_event(event)
    if line is not None:
        logger.info(f"  {line}")


class PostFitCallback(Callback):
    """Run the post-fit steps and report each as a :class:`TrainEvent`.

    Parameters
    ----------
    emit : callable, optional
        Where the events go; the log by default.
    stop_event : threading.Event, optional
        The run's cooperative cancel flag. A set flag before ``on_train_end`` skips every
        step; one set during a step ends it at the next batch with
        :class:`~cuvis_ai_core.training.evaluation.TrainingCancelled`.
    release_cuda_cache : bool
        Hand the caching allocator's unused blocks back before each step (the trainrun's
        ``release_cuda_cache_on_validation``).
    metric_nodes : collection of Node, optional
        The nodes the evaluation reduces; every node with a ``metrics`` port by default.
    """

    def __init__(
        self,
        *,
        emit: EmitFn | None = None,
        stop_event: threading.Event | None = None,
        release_cuda_cache: bool = False,
        metric_nodes: Collection[Node] | None = None,
    ) -> None:
        super().__init__()
        self.emit: EmitFn = emit if emit is not None else log_event
        self.stop_event = stop_event
        self.release_cuda_cache = release_cuda_cache
        self.metric_nodes = list(metric_nodes) if metric_nodes is not None else None
        self.result: PostFitResult | None = None

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run the steps once the fit loop is done and the model is still on its device."""
        if self.stop_event is not None and self.stop_event.is_set():
            return
        self.result = self.run(
            pl_module.pipeline,
            trainer.datamodule,
            module=pl_module,
            restore=True,
            epoch=trainer.current_epoch,
            global_step=trainer.global_step + 1,
        )

    def run(
        self,
        pipeline: CuvisPipeline,
        datamodule: Any,
        *,
        module: Any | None = None,
        restore: bool = True,
        epoch: int = 0,
        global_step: int = 0,
    ) -> PostFitResult:
        """Restore, calibrate and evaluate, each step when its inputs exist.

        ``module`` is the gradient trainer whose ``load_best_checkpoint`` performs the
        restore; with ``restore=False`` (a statistical fit, the standalone calibration
        phase) the step is skipped without an event.
        """
        datamodule.setup("validate")
        datamodule.setup("test")

        checkpoint: CheckpointRestored | None = None
        if restore:
            if module is None:
                checkpoint = CheckpointRestored(False, reason="no gradient trainer")
            else:
                checkpoint = module.load_best_checkpoint()
            self.emit(TrainEvent(TrainHook.CHECKPOINT_RESTORED, checkpoint=checkpoint))

        self._release()
        calibration = self._calibrate(pipeline, datamodule)

        evaluations: dict[str, EvaluationOutcome] = {}
        for split, stage in _SPLITS:
            self._release()
            evaluations[split] = self._evaluate(
                pipeline, datamodule, split, stage, epoch=epoch, global_step=global_step
            )

        self.emit(TrainEvent(TrainHook.POST_FIT_END))
        return PostFitResult(
            checkpoint=checkpoint, calibration=calibration, evaluations=evaluations
        )

    def _calibrate(
        self, pipeline: CuvisPipeline, datamodule: Any
    ) -> CalibrationOutcome:
        """Calibrate on val when a calibratable decider exists; one END event either way."""
        if not calibratable_deciders(pipeline):
            outcome = CalibrationOutcome(split="val", applicable=False)
            self.emit(
                TrainEvent(
                    TrainHook.CALIBRATION_END,
                    split=ExecutionStage.VAL,
                    calibration=outcome,
                )
            )
            return outcome
        self.emit(TrainEvent(TrainHook.CALIBRATION_START, split=ExecutionStage.VAL))
        outcome = calibrate_pipeline_deciders(
            pipeline,
            datamodule,
            split="val",
            on_batch=lambda done, total: self.emit(
                TrainEvent(
                    TrainHook.CALIBRATION_PROGRESS,
                    split=ExecutionStage.VAL,
                    progress=FrameProgress(done, total),
                )
            ),
        )
        self.emit(
            TrainEvent(
                TrainHook.CALIBRATION_END, split=ExecutionStage.VAL, calibration=outcome
            )
        )
        return outcome

    def _evaluate(
        self,
        pipeline: CuvisPipeline,
        datamodule: Any,
        split: str,
        stage: ExecutionStage,
        *,
        epoch: int,
        global_step: int,
    ) -> EvaluationOutcome:
        """Evaluate one split; a missing split or a failing pass is reported, never raised."""
        if getattr(datamodule, f"{split}_ds", None) is None:
            outcome = EvaluationOutcome(split=stage, status=EvaluationStatus.NO_SPLIT)
            self.emit(
                TrainEvent(TrainHook.EVALUATION_END, split=stage, evaluation=outcome)
            )
            return outcome
        self.emit(TrainEvent(TrainHook.EVALUATION_START, split=stage))
        metric_nodes = (
            self.metric_nodes
            if self.metric_nodes is not None
            else default_metric_nodes(pipeline)
        )
        try:
            summary = evaluate_split(
                pipeline,
                getattr(datamodule, f"{split}_dataloader")(),
                stage=stage,
                metric_nodes=metric_nodes,
                epoch=epoch,
                global_step=global_step,
                stop_event=self.stop_event,
                on_batch=lambda done, total: self.emit(
                    TrainEvent(
                        TrainHook.EVALUATION_PROGRESS,
                        split=stage,
                        progress=FrameProgress(done, total),
                    )
                ),
            )
        except TrainingCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - best effort: the run is still saved
            logger.opt(exception=True).warning(
                f"Post-fit evaluation of {split} failed."
            )
            outcome = EvaluationOutcome(
                split=stage,
                status=EvaluationStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            outcome = EvaluationOutcome(
                split=stage,
                status=EvaluationStatus.EVALUATED,
                frames=summary.frames,
                metrics=dict(summary.metrics),
            )
        self.emit(TrainEvent(TrainHook.EVALUATION_END, split=stage, evaluation=outcome))
        return outcome

    def _release(self) -> None:
        """Give the allocator's unused blocks back before a step, when asked to."""
        if self.release_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "CheckpointRestored",
    "EmitFn",
    "EvaluationOutcome",
    "EvaluationStatus",
    "FrameProgress",
    "PostFitCallback",
    "PostFitResult",
    "TrainEvent",
    "TrainHook",
    "describe_event",
    "log_event",
]
