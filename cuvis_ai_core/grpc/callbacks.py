"""Custom PyTorch Lightning callbacks for streaming gRPC progress.

``ProgressStreamCallback`` turns the fit into Train stream responses, each with one typed
:class:`~cuvis_ai_core.training.post_fit.TrainEvent` a client switches on: the fit start
with the epoch count, the epoch start with the train frame total, one event per train
batch (frames so far, on the response that carries the batch losses), the validation
start and one per validation batch, the validation end with the epoch's metrics, the
epoch end and the fit end with the completed-epoch count. The post-fit steps after the
last epoch report through the same event type from
:class:`~cuvis_ai_core.training.post_fit.PostFitCallback`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer

from cuvis_ai_core.training.calibration import calibration_pending
from cuvis_ai_core.training.evaluation import batch_frames, frame_total
from cuvis_ai_core.training.post_fit import (
    EpochProgress,
    FrameProgress,
    TrainEvent,
    TrainHook,
)
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
    and the lifecycle event of the emission; every emission of a fit carries one
    (the hooks of a Lightning ``test()`` pass carry none). The validation hooks
    stay quiet during Lightning's sanity check, so no sanity-check value ever
    reaches a chip. Frame totals come from the loaders the trainer runs, read when
    a pass starts; 0 stands for unknown.
    """

    def __init__(self, progress_handler: ProgressHandler) -> None:
        super().__init__()
        self.progress_handler = progress_handler
        self._current_epoch = 0
        self._current_batch = 0
        self._global_step = 0
        self._completed_epochs = 0
        self._train_done = 0
        self._train_total = 0
        self._val_done = 0
        self._val_total = 0

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
        """Announce the fit, its epoch count and whether a decider waits for its calibration.

        No frame total here: Lightning builds its loaders after this hook.
        """
        pipeline = getattr(pl_module, "pipeline", None)
        pending = calibration_pending(pipeline) if pipeline is not None else False
        self._emit_progress(
            ExecutionStage.TRAIN,
            {},
            {},
            "running",
            TrainEvent(
                TrainHook.FIT_START,
                calibration_pending=pending,
                epochs=self._epoch_progress(trainer),
            ),
        )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Open the epoch with the train frame total of the loader the fit runs."""
        self._current_epoch = trainer.current_epoch
        self._train_done = 0
        self._train_total = self._split_frame_total(trainer, "train")
        self._emit_progress(
            ExecutionStage.TRAIN,
            {},
            {},
            "running",
            TrainEvent(
                TrainHook.EPOCH_START,
                epochs=self._epoch_progress(trainer),
                progress=FrameProgress(0, self._train_total),
            ),
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """The batch losses and the frames seen so far, on one response."""
        self._current_batch = batch_idx
        self._global_step = trainer.global_step
        self._train_done += batch_frames(batch)
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "train")
        self._emit_progress(
            ExecutionStage.TRAIN,
            losses,
            metrics,
            "running",
            TrainEvent(
                TrainHook.TRAIN_PROGRESS,
                epochs=self._epoch_progress(trainer),
                progress=FrameProgress(self._train_done, self._train_total),
            ),
        )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        self._val_done = 0
        self._val_total = self._split_frame_total(trainer, "val")
        self._emit_progress(
            ExecutionStage.VAL,
            {},
            {},
            "running",
            TrainEvent(
                TrainHook.VALIDATION_START,
                split=ExecutionStage.VAL,
                epochs=self._epoch_progress(trainer),
                progress=FrameProgress(0, self._val_total),
            ),
        )

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
        self._val_done += batch_frames(batch)
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "val")
        self._emit_progress(
            ExecutionStage.VAL,
            losses,
            metrics,
            "running",
            TrainEvent(
                TrainHook.VALIDATION_PROGRESS,
                split=ExecutionStage.VAL,
                progress=FrameProgress(self._val_done, self._val_total),
            ),
        )

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
            TrainEvent(
                TrainHook.VALIDATION_END,
                split=ExecutionStage.VAL,
                epochs=self._epoch_progress(trainer),
            ),
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._completed_epochs += 1
        losses = self._extract_epoch_losses(trainer)
        metrics = self._extract_metrics(trainer, "train")
        self._emit_progress(
            ExecutionStage.TRAIN,
            losses,
            metrics,
            "running",
            TrainEvent(TrainHook.EPOCH_END, epochs=self._epoch_progress(trainer)),
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
        """The last epoch is done; the post-fit steps follow, so the status stays running.

        The epoch pair counts the epochs that completed, which an early stop leaves
        below the total.
        """
        self._emit_progress(
            ExecutionStage.TRAIN,
            {},
            {},
            "running",
            TrainEvent(
                TrainHook.FIT_END,
                epochs=EpochProgress(
                    self._completed_epochs, self._epoch_total(trainer)
                ),
            ),
        )

    # ----------------------------- Helpers -----------------------------
    @staticmethod
    def _epoch_total(trainer: Trainer) -> int:
        """The trainer's ``max_epochs`` when above 0, else 0 (Lightning's -1: unlimited)."""
        max_epochs = getattr(trainer, "max_epochs", None)
        return max_epochs if isinstance(max_epochs, int) and max_epochs > 0 else 0

    def _epoch_progress(self, trainer: Trainer) -> EpochProgress:
        """The epoch under way, one-based for display, and the total."""
        current = int(getattr(trainer, "current_epoch", 0)) + 1
        return EpochProgress(current, self._epoch_total(trainer))

    @staticmethod
    def _split_frame_total(trainer: Trainer, split: str) -> int:
        """The frames of the pass the trainer runs over ``split``, 0 when unknown.

        Read from the loader the trainer holds (``train_dataloader`` or
        ``val_dataloaders``, set once the fit loop built them), never through the
        datamodule's loader factory, which would build a second loader. 0 when the
        loader has no sized dataset or the trainer limits the pass
        (``limit_train_batches``, ``limit_val_batches``): the split's length would
        then overstate the frames the pass sees.
        """
        loader_attr, limit_attr = (
            ("train_dataloader", "limit_train_batches")
            if split == "train"
            else ("val_dataloaders", "limit_val_batches")
        )
        limit = getattr(trainer, limit_attr, 1.0)
        whole = limit is None or (isinstance(limit, float) and limit >= 1.0)
        loader = getattr(trainer, loader_attr, None)
        if not whole or loader is None:
            return 0
        try:
            return frame_total(loader)
        except Exception:
            return 0

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
