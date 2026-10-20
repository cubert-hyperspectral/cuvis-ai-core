"""Custom PyTorch Lightning callbacks for streaming gRPC progress.

``ProgressStreamCallback`` turns the fit into Train stream responses: per-batch losses,
per-epoch metrics, and one typed :class:`~cuvis_ai_core.training.post_fit.TrainEvent` at
each lifecycle hook a client switches on (fit start, validation end, epoch end, fit end).
The post-fit steps after the last epoch report through the same event type from
:class:`~cuvis_ai_core.training.post_fit.PostFitCallback`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer

from cuvis_ai_core.training.calibration import calibration_pending
from cuvis_ai_core.training.post_fit import TrainEvent, TrainHook
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

ProgressHandler = Callable[
    [Context, dict[str, float], dict[str, float], str, "TrainEvent | None"], None
]
"""``(context, losses, metrics, status, event)``: one Train stream response."""

_STAGE_PREFIXES = ("train", "val", "test")


class StopTrainingCallback(Callback):
    """Stop a Lightning run cooperatively when a stop event is set.

    Checked at batch boundaries: setting ``trainer.should_stop`` makes the
    training epoch loop end after the current batch, so cancellation never
    interrupts an in-flight forward/backward pass.
    """

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self.stop_event = stop_event

    def _check(self, trainer: Trainer) -> None:
        """Request a graceful stop if the event fired."""
        if self.stop_event.is_set():
            trainer.should_stop = True

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._check(trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._check(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._check(trainer)


class ProgressStreamCallback(Callback):
    """Callback that forwards training progress to a handler function.

    The handler receives the context, the losses, the metrics, the status string
    and the lifecycle event of the emission (``None`` for the per-batch
    emissions). The validation hooks stay quiet during Lightning's sanity check,
    so no sanity-check value ever reaches a chip.
    """

    def __init__(self, progress_handler: ProgressHandler) -> None:
        super().__init__()
        self.progress_handler = progress_handler
        self._current_epoch = 0
        self._current_batch = 0
        self._global_step = 0

    @property
    def last_epoch(self) -> int:
        """The epoch of the last emission; the post-fit events keep it."""
        return self._current_epoch

    @property
    def last_global_step(self) -> int:
        """The global step of the last emission; the post-fit events keep it."""
        return self._global_step

    # ----------------------------- Lifecycle hooks -----------------------------
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Announce the fit and whether a decider still waits for its calibration."""
        pipeline = getattr(pl_module, "pipeline", None)
        pending = calibration_pending(pipeline) if pipeline is not None else False
        self._emit_progress(
            ExecutionStage.TRAIN,
            {},
            {},
            "running",
            TrainEvent(TrainHook.FIT_START, calibration_pending=pending),
        )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._current_epoch = trainer.current_epoch
        self._emit_progress(ExecutionStage.TRAIN, {}, {}, "running")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._current_batch = batch_idx
        self._global_step = trainer.global_step
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "train")
        self._emit_progress(ExecutionStage.TRAIN, losses, metrics, "running")

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        self._emit_progress(ExecutionStage.VAL, {}, {}, "running")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "val")
        self._emit_progress(ExecutionStage.VAL, losses, metrics, "running")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # Epoch-aggregated emission INCLUDING losses: per-batch emissions carry
        # only that batch's loss, and _extract_metrics deliberately skips loss
        # keys, so without this hook an epoch-level val_loss never reaches the
        # stream. A live loss curve needs the train and val loss per epoch.
        if getattr(trainer, "sanity_checking", False):
            return
        losses = self._extract_epoch_losses(trainer)
        metrics = self._extract_metrics(trainer, "val")
        self._emit_progress(
            ExecutionStage.VAL,
            losses,
            metrics,
            "running",
            TrainEvent(TrainHook.VALIDATION_END),
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        losses = self._extract_epoch_losses(trainer)
        metrics = self._extract_metrics(trainer, "train")
        self._emit_progress(
            ExecutionStage.TRAIN,
            losses,
            metrics,
            "running",
            TrainEvent(TrainHook.EPOCH_END),
        )

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit_progress(ExecutionStage.TEST, {}, {}, "running")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "test")
        self._emit_progress(ExecutionStage.TEST, losses, metrics, "running")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """The last epoch is done; the post-fit steps follow, so the status stays running."""
        self._emit_progress(
            ExecutionStage.TRAIN, {}, {}, "running", TrainEvent(TrainHook.FIT_END)
        )

    # ----------------------------- Helpers -----------------------------
    def _extract_losses(self, outputs: Any) -> dict[str, float]:
        """Extract loss values from batch outputs."""
        losses: dict[str, float] = {}

        if outputs is None:
            return losses

        if isinstance(outputs, dict):
            if "loss" in outputs:
                losses["total"] = float(outputs["loss"])
            for key, value in outputs.items():
                if "loss" in key.lower() and key != "loss":
                    try:
                        losses[key] = float(value)
                    except Exception:
                        continue
        elif hasattr(outputs, "item"):
            losses["total"] = float(outputs.item())

        return losses

    def _extract_epoch_losses(self, trainer: Trainer) -> dict[str, float]:
        """Extract epoch-aggregated loss values from trainer.callback_metrics."""
        losses: dict[str, float] = {}
        callback_metrics = getattr(trainer, "callback_metrics", {}) or {}

        for key, value in callback_metrics.items():
            if "loss" not in key.lower():
                continue
            try:
                losses[key] = float(value.item())
            except Exception:
                try:
                    losses[key] = float(value)
                except Exception:
                    continue

        return losses

    def _extract_metrics(self, trainer: Trainer, stage: str) -> dict[str, float]:
        """Extract metrics from trainer.callback_metrics.

        Every stage prefix (``train``, ``val``, ``test`` with ``_`` or ``/``) is
        stripped, not only the current stage's: the metrics map names the metric,
        the context names the stage.
        """
        metrics: dict[str, float] = {}
        callback_metrics = getattr(trainer, "callback_metrics", {}) or {}

        for key, value in callback_metrics.items():
            if "loss" in key.lower():
                continue

            metric_name = key
            for prefix in _STAGE_PREFIXES:
                metric_name = metric_name.replace(f"{prefix}_", "").replace(
                    f"{prefix}/", ""
                )

            try:
                metrics[metric_name] = float(value.item())
            except Exception:
                try:
                    metrics[metric_name] = float(value)
                except Exception:
                    continue

        return metrics

    def _emit_progress(
        self,
        stage: ExecutionStage,
        losses: dict[str, float],
        metrics: dict[str, float],
        status: str,
        event: TrainEvent | None = None,
    ) -> None:
        context = Context(
            stage=stage,
            epoch=self._current_epoch,
            batch_idx=self._current_batch,
            global_step=self._global_step,
        )
        self.progress_handler(context, losses, metrics, status, event)


__all__ = ["ProgressHandler", "ProgressStreamCallback", "StopTrainingCallback"]
