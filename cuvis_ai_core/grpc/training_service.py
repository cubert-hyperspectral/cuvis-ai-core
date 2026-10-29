"""Training operations service component."""

from __future__ import annotations

import gc
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import grpc
import torch
from loguru import logger

from cuvis_ai_core.grpc.callbacks import ProgressStreamCallback, StopTrainingCallback
from cuvis_ai_core.training.calibration import (
    CalibrationOutcome,
    uncalibrated_required_consumers,
)
from cuvis_ai_core.training.post_fit import (
    PostFitCallback,
    PostFitResult,
    TrainEvent,
    TrainHook,
    describe_event,
)
from cuvis_ai_core.training.config import (
    CalibrationConfig,
    DataConfig,
    TrainingConfig,
    TrainRunConfig,
    create_callbacks_from_config,
)
from cuvis_ai_core.training.evaluation import is_cuda_out_of_memory
from cuvis_ai_core.training.trainers import (
    GradientTrainer,
    StatisticalTrainer,
    TrainingCancelled,
)
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from .error_handling import (
    get_session_and_pipeline,
    get_session_or_error,
    grpc_handler,
)
from .session_manager import SessionManager, SessionState
from .v1 import cuvis_ai_pb2

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    import pytorch_lightning as pl


# Base text of the CANCELLED answer. A run that also raised gets the failure
# appended so a stop can never hide a crash.
_CANCELLED_DETAIL = (
    "Training cancelled; the pipeline holds the state of the last completed step"
)


def _run_outcome(
    training_error: Exception | None,
    stop_set_at_failure: bool,
    stop_set_now: bool,
) -> Exception | None:
    """Decide what a finished gradient run raises: the error, a cancel, or nothing.

    A stop takes precedence only when it was already requested at the
    moment the run failed, because then the failure is teardown noise from
    a trainer that was winding down. An error that arrived first wins even
    if a stop followed it, so a crash is never reported as a clean
    cancellation. A swallowed error is logged and travels on the
    :class:`TrainingCancelled` it lost to.

    Parameters
    ----------
    training_error : Exception, optional
        What the training thread raised, if anything.
    stop_set_at_failure : bool
        Whether the session's stop event was already set when the run failed.
    stop_set_now : bool
        Whether the stop event is set now that the run has finished.

    A :class:`TrainingCancelled` the run raised itself (the statistical fit at a
    batch boundary) is the cancellation, not noise, and is returned as it is.

    Returns
    -------
    Exception or None
        The exception to raise, or ``None`` when the run completed.
    """
    if isinstance(training_error, TrainingCancelled):
        return training_error
    if training_error is not None and not stop_set_at_failure:
        return training_error
    if not stop_set_now:
        return training_error
    if training_error is not None:
        logger.opt(exception=training_error).warning(
            f"Gradient training was stopped by request and also raised: "
            f"{training_error!r}. Reporting the run as cancelled."
        )
        return TrainingCancelled(
            "Gradient training stopped via StopTrain", swallowed_error=training_error
        )
    return TrainingCancelled("Gradient training stopped via StopTrain")


class TrainingService:
    """Handles statistical and gradient training operations."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def train(
        self,
        request: cuvis_ai_pb2.TrainRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train the pipeline with statistical or gradient methods."""
        resolved = get_session_and_pipeline(
            self.session_manager, request.session_id, context
        )
        if resolved is None:
            return
        session, _ = resolved

        stop_event = session.stop_event

        # A stop that landed before this stream opened still cancels it: the
        # flag survives until the next SetTrainRunConfig, so a StopTrain
        # issued between two trainer phases of one run (statistical done,
        # gradient not yet started) terminates the later phase here instead
        # of racing it.
        if stop_event.is_set():
            response = self._cancelled_response(
                "Training run was cancelled; not starting this trainer phase"
            )
            session.latest_train_response = response
            yield response
            return

        # Client-drop cancels: when the caller abandons the stream, the RPC
        # terminates and this callback fires while training may still be
        # executing between yields. Guarded so a normally-finished stream
        # (which also terminates the RPC) does not cancel the run's next
        # trainer phase.
        stream_finished = threading.Event()

        def _on_stream_terminated() -> None:
            """Cancel the run when the stream died before finishing."""
            if not stream_finished.is_set():
                stop_event.set()

        add_callback = getattr(context, "add_callback", None)
        if callable(add_callback):
            add_callback(_on_stream_terminated)

        # Get data config - either from request or from session's experiment config
        data_config_py: DataConfig | None = None

        if request.HasField("data"):
            # Use data config from request
            data_config_py = DataConfig.from_proto(request.data)
        elif session.data_config is not None:
            # Use data config from session (loaded via RestoreTrainRun/SetTrainRunConfig)
            data_config_py = session.data_config
        else:
            stream_finished.set()
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                "Training requires data config to be provided either in request or loaded via RestoreTrainRun"
            )
            return

        try:
            # A relative splits_path cannot be resolved on the wire Train path:
            # there is no trainrun file to anchor it. RestoreTrainRun resolves a
            # relative splits_path against the trainrun dir before it reaches the
            # session; a raw TrainRequest.data / SetTrainRunConfig with a relative
            # path is genuinely baseless. Reject it with a clear error instead of
            # resolving against the child's wiped CWD (wrong file / FileNotFound).
            splits = data_config_py.splits
            if (
                splits is not None
                and splits.splits_path
                and not Path(splits.splits_path).is_absolute()
            ):
                raise ValueError(
                    "splits_path must be absolute on the gRPC Train path (no trainrun "
                    "base dir to resolve against); send an absolute path or load the "
                    "trainrun via RestoreTrainRun"
                )

            # The DataModule class named by ``data_config.data_module`` comes from a
            # plugin registered into the session's ``node_registry.data_modules``;
            # core imports no SDK and no concrete module here.
            from cuvis_ai_core.data.datamodule import create_data_module

            datamodule = create_data_module(session.node_registry, data_config_py)
            self._refuse_uncalibrated_consumers(session.pipeline)
            # Read before the experiment capture below, which writes a default
            # TrainingConfig into the session for a statistical phase.
            statistical_only = session.training_config is None
            training_config_py: TrainingConfig | None = None
            inner: Iterator[cuvis_ai_pb2.TrainResponse]

            if request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL:
                # Statistical training - single pass; the post-fit steps follow
                # only when no gradient phase comes after it.
                # Preserve existing training_config from session (if loaded via RestoreTrainRun)
                # or create default if not present
                training_config_py = session.training_config or TrainingConfig()
                self._capture_experiment_context(
                    session, data_config_py, training_config_py
                )
                inner = self._train_statistical(
                    session, datamodule, post_fit=statistical_only
                )

            elif request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_GRADIENT:
                # Gradient training - streaming progress
                # Get training config - either from request or from session
                if request.HasField("training"):
                    training_config_py = self._deserialize_training_config(
                        request.training
                    )
                elif session.training_config is not None:
                    training_config_py = session.training_config
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "Gradient training requires training config either in request or loaded via RestoreTrainRun"
                    )
                    return

                self._capture_experiment_context(
                    session, data_config_py, training_config_py
                )
                inner = self._train_gradient(session, datamodule, training_config_py)

            elif request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_CALIBRATION:
                # The post-fit steps alone, on weights the session already holds.
                refusal = self._calibration_precondition(session)
                if refusal is not None:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details(refusal)
                    return
                inner = self._train_calibration(session, datamodule)

            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Unknown trainer type: {request.trainer_type}")
                return

            # Every progress response is recorded before it goes on the wire so
            # GetTrainStatus reports the latest state even after the stream ends.
            for response in inner:
                session.latest_train_response = response
                yield response

        except TrainingCancelled as cancelled:
            detail = _CANCELLED_DETAIL
            swallowed = getattr(cancelled, "swallowed_error", None)
            if swallowed is not None:
                detail = (
                    f"{detail} (stopped by request; the run also raised: {swallowed})"
                )
            response = self._cancelled_response(detail)
            session.latest_train_response = response
            yield response
            return
        except GeneratorExit:
            # The caller closed the stream mid-run (tab closed, app quit,
            # connection lost): cancel the run so the training thread does not
            # keep the GPU busy to completion.
            stop_event.set()
            raise
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return
        except Exception as exc:
            # A GPU out-of-memory error gets its own code: a client may run the pass
            # again on the CPU, and the code is the verdict, not the message.
            context.set_code(
                grpc.StatusCode.RESOURCE_EXHAUSTED
                if is_cuda_out_of_memory(exc)
                else grpc.StatusCode.INTERNAL
            )
            context.set_details(f"Training failed: {str(exc)}")
            return
        finally:
            stream_finished.set()
            # Best-effort VRAM trim after a run ends (complete, cancelled, or
            # failed). The pipeline itself stays resident until CloseSession.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @grpc_handler("Failed to stop training")
    def stop_train(
        self,
        request: cuvis_ai_pb2.StopTrainRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.StopTrainResponse:
        """Request cooperative cancellation of the session's training run.

        Sets the session's stop flag; training honors it at the next batch /
        node boundary and the active Train stream ends with a terminal
        TRAIN_STATUS_CANCELLED. The flag stays set until the next
        SetTrainRunConfig, so a stop issued between trainer phases also
        cancels the not-yet-started phase of the same run.
        """
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.StopTrainResponse(
                accepted=False, message="Session not found"
            )

        session.stop_event.set()
        return cuvis_ai_pb2.StopTrainResponse(
            accepted=True,
            message=(
                "Stop requested; training stops at the next batch/node boundary "
                "and the Train stream ends with TRAIN_STATUS_CANCELLED"
            ),
        )

    @grpc_handler("Failed to get status")
    def get_train_status(
        self,
        request: cuvis_ai_pb2.GetTrainStatusRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetTrainStatusResponse:
        """Get current training status (the latest progress the Train stream yielded)."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.GetTrainStatusResponse()

        if session.latest_train_response is not None:
            return cuvis_ai_pb2.GetTrainStatusResponse(
                latest_progress=session.latest_train_response
            )

        # No Train stream has produced progress in this session yet.
        return cuvis_ai_pb2.GetTrainStatusResponse(
            latest_progress=cuvis_ai_pb2.TrainResponse(
                context=cuvis_ai_pb2.Context(
                    stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
                    epoch=0,
                    batch_idx=0,
                    global_step=0,
                ),
                status=cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED,
                message="No training activity in this session yet",
            )
        )

    @grpc_handler("Failed to get capabilities")
    def get_training_capabilities(
        self,
        request: cuvis_ai_pb2.GetTrainingCapabilitiesRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetTrainingCapabilitiesResponse:
        """Return supported optimizers, schedulers, and callbacks."""
        from cuvis_ai_core.training.optimizer_registry import (
            get_supported_optimizers,
            get_supported_schedulers,
        )

        supported_optimizers = get_supported_optimizers()
        supported_schedulers = get_supported_schedulers()

        callbacks = [
            cuvis_ai_pb2.CallbackTypeInfo(
                type="EarlyStopping",
                description="Stop training when a monitored metric stops improving.",
                parameters=[
                    cuvis_ai_pb2.ParamSpec(
                        name="monitor",
                        type="string",
                        required=True,
                        description="Metric to monitor (e.g., 'val_loss').",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="patience",
                        type="int",
                        required=False,
                        default_value="10",
                        description="Number of epochs with no improvement.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="mode",
                        type="string",
                        required=False,
                        default_value="min",
                        validation="in ['min', 'max']",
                        description="Optimization direction for monitored metric.",
                    ),
                ],
            ),
            cuvis_ai_pb2.CallbackTypeInfo(
                type="ModelCheckpoint",
                description="Persist checkpoints during training.",
                parameters=[
                    cuvis_ai_pb2.ParamSpec(
                        name="dirpath",
                        type="string",
                        required=True,
                        description="Directory to store checkpoints.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="monitor",
                        type="string",
                        required=True,
                        description="Metric to monitor for best checkpoint.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="mode",
                        type="string",
                        required=False,
                        default_value="max",
                        validation="in ['min', 'max']",
                        description="Optimization direction for checkpoint metric.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="save_top_k",
                        type="int",
                        required=False,
                        default_value="1",
                        description="Number of best checkpoints to keep.",
                    ),
                ],
            ),
            cuvis_ai_pb2.CallbackTypeInfo(
                type="LearningRateMonitor",
                description="Log learning rate during training.",
                parameters=[
                    cuvis_ai_pb2.ParamSpec(
                        name="logging_interval",
                        type="string",
                        required=False,
                        default_value="epoch",
                        validation="in ['step', 'epoch']",
                        description="Frequency to log learning rate.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="log_momentum",
                        type="bool",
                        required=False,
                        default_value="False",
                        description="Whether to log optimizer momentum.",
                    ),
                ],
            ),
        ]

        optimizer_params = cuvis_ai_pb2.OptimizerParamsSchema(
            parameters=[
                cuvis_ai_pb2.ParamSpec(
                    name="lr",
                    type="float",
                    required=True,
                    description="Learning rate.",
                ),
                cuvis_ai_pb2.ParamSpec(
                    name="weight_decay",
                    type="float",
                    required=False,
                    default_value="0.0",
                    description="Weight decay (L2 regularization).",
                ),
                cuvis_ai_pb2.ParamSpec(
                    name="betas",
                    type="tuple",
                    required=False,
                    description="Adam/AdamW betas (beta1, beta2).",
                ),
            ]
        )

        scheduler_params = cuvis_ai_pb2.SchedulerParamsSchema(
            parameters=[
                cuvis_ai_pb2.ParamSpec(
                    name="monitor",
                    type="string",
                    required=False,
                    description="Metric to monitor for scheduler decisions.",
                ),
                cuvis_ai_pb2.ParamSpec(
                    name="factor",
                    type="float",
                    required=False,
                    default_value="0.1",
                    description="LR reduction factor for ReduceLROnPlateau.",
                ),
                cuvis_ai_pb2.ParamSpec(
                    name="patience",
                    type="int",
                    required=False,
                    default_value="10",
                    description="Epochs to wait before reducing LR.",
                ),
            ]
        )

        return cuvis_ai_pb2.GetTrainingCapabilitiesResponse(
            supported_optimizers=supported_optimizers,
            supported_schedulers=supported_schedulers,
            supported_callbacks=callbacks,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
        )

    def _capture_experiment_context(
        self,
        session: SessionState,
        data_config: DataConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Persist experiment context on the session for SaveTrainRun."""
        pipeline_config = session.pipeline_config

        metadata_name = (
            pipeline_config.metadata.name if pipeline_config.metadata else ""
        )
        pipeline_name = getattr(session.pipeline, "name", metadata_name or "experiment")

        session.data_config = data_config
        session.training_config = training_config

        # Preserve loss_nodes, metric_nodes, and unfreeze_nodes from existing trainrun_config if available
        loss_nodes = []
        metric_nodes = []
        unfreeze_nodes = []
        freeze_nodes = []
        calibration = self._calibration_of(session)

        if session.trainrun_config is not None:
            loss_nodes = session.trainrun_config.loss_nodes
            metric_nodes = session.trainrun_config.metric_nodes
            unfreeze_nodes = session.trainrun_config.unfreeze_nodes
            freeze_nodes = session.trainrun_config.freeze_nodes

        # The live pipeline is held on the session (``session.pipeline`` /
        # ``_pipeline_config``); the trainrun's ``pipeline`` field is a disk
        # reference, materialised by SaveTrainRun (which writes the pipeline to a
        # sibling YAML), so it stays unset for the in-memory experiment context.
        session.trainrun_config = TrainRunConfig(
            name=str(pipeline_name),
            pipeline=None,
            data=data_config,
            training=training_config,
            loss_nodes=loss_nodes,
            metric_nodes=metric_nodes,
            freeze_nodes=freeze_nodes,
            unfreeze_nodes=unfreeze_nodes,
            calibration=calibration,
        )

    @staticmethod
    def _calibration_of(session: SessionState) -> CalibrationConfig | None:
        """The calibration block of the session's trainrun, or ``None`` without one."""
        trainrun = session.trainrun_config
        return trainrun.calibration if trainrun is not None else None

    def _train_statistical(
        self,
        session: SessionState,
        datamodule: "pl.LightningDataModule",
        *,
        post_fit: bool = True,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train with the statistical method; the post-fit steps follow a statistical-only run.

        ``post_fit`` is False when a gradient phase comes next: the restore, the
        calibration and the evaluation then run once, at the end of that fit.
        """
        yield cuvis_ai_pb2.TrainResponse(
            context=cuvis_ai_pb2.Context(
                stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
                epoch=0,
                batch_idx=0,
                global_step=0,
            ),
            status=cuvis_ai_pb2.TRAIN_STATUS_RUNNING,
            message="Starting statistical training",
        )

        progress_queue: queue.Queue[cuvis_ai_pb2.TrainResponse] = queue.Queue()
        fit_context = Context(
            stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0, global_step=0
        )
        trainer = StatisticalTrainer(
            pipeline=session.pipeline,
            datamodule=datamodule,
            stop_event=session.stop_event,
            metric_nodes=self._metric_nodes(session),
            emit=lambda event: progress_queue.put(
                self._event_response(fit_context, event)
            ),
        )

        # Fit the pipeline (initializes normalizers, selectors, PCA, RX) on a
        # worker thread, so each node's start, batches and end reach the stream
        # while it runs. A StopTrain raises TrainingCancelled inside the fit;
        # the drain re-raises it and train() maps it to the CANCELLED response.
        yield from self._drain_worker(trainer.fit, progress_queue, session.stop_event)
        session.trainer = trainer
        session.weights_source = "fit"

        message = "Statistical training complete"
        if post_fit:
            base = Context(
                stage=ExecutionStage.TRAIN, epoch=1, batch_idx=0, global_step=1
            )
            outcome = yield from self._run_post_fit(
                session, datamodule, restore=False, base_context=base
            )
            if outcome.applicable:
                message = f"{message}; {outcome.summary()}"

        yield cuvis_ai_pb2.TrainResponse(
            context=cuvis_ai_pb2.Context(
                stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
                epoch=1,
                batch_idx=0,
                global_step=1,
            ),
            status=cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
            message=message,
        )

    def _train_gradient(
        self,
        session: SessionState,
        datamodule: "pl.LightningDataModule",
        training_config: TrainingConfig,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train with the gradient method and stream progress, the post-fit steps included.

        The fit runs on a worker thread; the Lightning callbacks queue one
        response per emission and the post-fit callback queues one per
        lifecycle event, all drained here in order. A StopTrain during the fit
        or during a post-fit step ends the stream CANCELLED.
        """
        # Seed as early as the Train RPC allows so training-time randomness
        # (shuffling, dropout, augmentation) is reproducible.
        import pytorch_lightning as pl

        pl.seed_everything(training_config.seed, workers=True)

        loss_nodes, metric_nodes = self._configure_gradient_components(session)

        progress_queue: queue.Queue[cuvis_ai_pb2.TrainResponse] = queue.Queue()

        def progress_handler(
            context_obj: Context,
            losses: dict,
            metrics: dict,
            status: str,
            event: TrainEvent | None = None,
        ) -> None:
            """Queue one progress response for the streaming reply."""
            message = (_event_message(event) if event is not None else "") or (
                "Gradient training"
            )
            progress_queue.put(
                self._create_progress_response(
                    context_obj,
                    losses,
                    metrics,
                    status=status,
                    message=message,
                    event=event,
                )
            )

        progress = ProgressStreamCallback(progress_handler)

        def emit(event: TrainEvent) -> None:
            """Queue one post-fit event; its context keeps the fit's last epoch and step."""
            base = Context(
                stage=ExecutionStage.TRAIN,
                epoch=progress.last_epoch,
                batch_idx=0,
                global_step=progress.last_global_step,
            )
            progress_queue.put(self._event_response(base, event))

        calibration = self._calibration_of(session)
        post_fit = PostFitCallback(
            emit=emit,
            stop_event=session.stop_event,
            release_cuda_cache=training_config.release_cuda_cache_on_validation,
            metric_nodes=metric_nodes or None,
            calibration=calibration,
        )
        callback_list = [progress, StopTrainingCallback(session.stop_event), post_fit]
        callback_list.extend(create_callbacks_from_config(training_config.callbacks))

        trainer = GradientTrainer(
            pipeline=session.pipeline,
            datamodule=datamodule,
            training_config=training_config,
            loss_nodes=loss_nodes,
            metric_nodes=metric_nodes,
            callbacks=callback_list,
            calibration=calibration,
        )
        session.trainer = trainer

        yield from self._drain_worker(trainer.fit, progress_queue, session.stop_event)
        session.weights_source = "fit"

        result = post_fit.result
        outcome = (
            result.calibration
            if result is not None
            else CalibrationOutcome(split="val", applicable=False)
        )
        if outcome.calibrated:
            session.pipeline_config = None
        suffix = f"; {outcome.summary()}" if outcome.applicable else ""

        final_context = Context(
            stage=ExecutionStage.TRAIN,
            epoch=training_config.max_epochs,
            batch_idx=0,
            global_step=getattr(getattr(trainer, "_trainer", None), "global_step", 0),
        )
        yield self._create_progress_response(
            final_context,
            losses={},
            metrics={},
            status="complete",
            message=f"Gradient training complete{suffix}",
        )

    def _train_calibration(
        self,
        session: SessionState,
        datamodule: "pl.LightningDataModule",
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Run the post-fit steps on a session that already holds trained weights; no fit."""
        base = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0, global_step=0)
        yield self._create_progress_response(
            base,
            losses={},
            metrics={},
            status="running",
            message="Starting calibration",
        )
        # Nothing is unsaved here, so a GPU out-of-memory error ends the call with its
        # own code instead of a recorded skip: the client may run the pass on the CPU.
        outcome = yield from self._run_post_fit(
            session,
            datamodule,
            restore=False,
            base_context=base,
            raise_out_of_memory=True,
        )
        suffix = (
            f"; {outcome.summary()}"
            if outcome.applicable
            else "; no calibratable decider"
        )
        yield self._create_progress_response(
            base,
            losses={},
            metrics={},
            status="complete",
            message=f"Calibration complete{suffix}",
        )

    @staticmethod
    def _calibration_precondition(session: SessionState) -> str | None:
        """Why the calibration phase cannot run on this session, or ``None`` when it can."""
        if session.weights_source is None:
            return (
                "The session holds no trained weights: restore a saved run with its "
                "weights, load weights, or train before a calibration"
            )
        unfitted = [
            node.name
            for node in session.pipeline.nodes()
            if node.requires_initial_fit
            and not getattr(node, "_statistically_initialized", False)
        ]
        if unfitted:
            return (
                f"Nodes {', '.join(unfitted)} need a statistical fit before a calibration; "
                "train or load weights first"
            )
        return None

    @staticmethod
    def _refuse_uncalibrated_consumers(pipeline) -> None:
        """Refuse a training graph that feeds a required decisions input from an uncalibrated decider.

        Such a node would fail its first forward at a training stage (the decider emits
        no decisions before its calibration); the refusal names both nodes before any fit.
        """
        consumers = uncalibrated_required_consumers(pipeline)
        if not consumers:
            return
        clauses = ", ".join(
            f"{node.name} reads decisions from {decider.name}"
            for node, decider in consumers
        )
        raise ValueError(
            f"{clauses}, whose thresholds are not calibrated; set thresholds in the "
            "pipeline or take the node out of the training graph"
        )

    @staticmethod
    def _metric_nodes(session: SessionState) -> list | None:
        """The trainrun's metric nodes by name, or ``None`` for discovery."""
        trainrun = getattr(session, "trainrun_config", None)
        names = getattr(trainrun, "metric_nodes", None)
        if (
            not isinstance(names, (list, tuple))
            or not names
            or session.pipeline is None
        ):
            return None
        wanted = set(names)
        return [node for node in session.pipeline.nodes() if node.name in wanted]

    def _run_post_fit(
        self,
        session: SessionState,
        datamodule: "pl.LightningDataModule",
        *,
        restore: bool,
        base_context: Context,
        raise_out_of_memory: bool = False,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Run the post-fit steps on a worker thread and stream their events.

        The generator's return value is the calibration outcome; a calibrated
        pipeline drops the session's cached config so the save re-derives it.
        ``raise_out_of_memory`` lets a GPU out-of-memory error end the steps
        (see :class:`PostFitCallback`).
        """
        progress_queue: queue.Queue[cuvis_ai_pb2.TrainResponse] = queue.Queue()
        results: list[PostFitResult] = []
        post_fit = PostFitCallback(
            emit=lambda event: progress_queue.put(
                self._event_response(base_context, event)
            ),
            stop_event=session.stop_event,
            metric_nodes=self._metric_nodes(session),
            calibration=self._calibration_of(session),
            raise_out_of_memory=raise_out_of_memory,
        )

        def _run() -> None:
            """The steps, on the worker thread."""
            results.append(post_fit.run(session.pipeline, datamodule, restore=restore))

        yield from self._drain_worker(_run, progress_queue, session.stop_event)
        outcome = (
            results[0].calibration
            if results
            else CalibrationOutcome(split="val", applicable=False)
        )
        if outcome.calibrated:
            session.pipeline_config = None
        return outcome

    @staticmethod
    def _drain_worker(
        run: Callable[[], object],
        progress_queue: "queue.Queue[cuvis_ai_pb2.TrainResponse]",
        stop_event: threading.Event,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Run ``run`` on a worker thread and yield the queued responses while it runs.

        When the thread is done the drain empties the queue, then the run's
        outcome is decided (``_run_outcome``): an error is raised as is, a
        requested stop as :class:`TrainingCancelled`.
        """
        complete = threading.Event()
        error: Exception | None = None
        stop_set_at_failure = False

        def _target() -> None:
            """Run the work and record its failure for the stream."""
            nonlocal error, stop_set_at_failure
            try:
                run()
            except (
                Exception
            ) as exc:  # pragma: no cover - surfaced via the stream outcome
                error = exc
                # Read the stop flag here, not after the loop: only a stop that
                # was already requested when the run blew up may take
                # precedence over the error.
                stop_set_at_failure = stop_event.is_set()
            finally:
                complete.set()

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

        while not complete.is_set() or not progress_queue.empty():
            try:
                progress = progress_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            else:
                yield progress

        outcome = _run_outcome(error, stop_set_at_failure, stop_event.is_set())
        if outcome is not None:
            raise outcome

    def _event_response(
        self, base: Context, event: TrainEvent
    ) -> cuvis_ai_pb2.TrainResponse:
        """One RUNNING response for a post-fit event, with the status line a human reads."""
        stage = event.split if event.split is not None else base.stage
        context = Context(
            stage=stage, epoch=base.epoch, batch_idx=0, global_step=base.global_step
        )
        return self._create_progress_response(
            context,
            losses={},
            metrics={},
            status="running",
            message=_event_message(event),
            event=event,
        )

    def _deserialize_training_config(
        self, config_proto: cuvis_ai_pb2.TrainingConfig
    ) -> TrainingConfig:
        """Decode TrainingConfig from proto bytes."""
        if not config_proto.config_bytes:
            raise ValueError("Training config cannot be empty")

        try:
            return TrainingConfig.from_proto(config_proto)
        except Exception as exc:
            raise ValueError(f"Invalid training config: {exc}") from exc

    def _configure_gradient_components(
        self, session: SessionState
    ) -> tuple[list, list]:
        """Configure loss and metric nodes from trainrun config."""
        pipeline = session.pipeline

        # Require explicit trainrun config
        if session.trainrun_config is None:
            raise ValueError(
                "Gradient training requires explicit TrainRunConfig with loss_nodes and metric_nodes. "
                "Please provide train run configuration via RestoreTrainRun or session creation."
            )

        trainrun_config = session.trainrun_config

        # Validate required fields
        if not trainrun_config.loss_nodes:
            raise ValueError(
                "trainrun_config.loss_nodes must specify at least one loss node for gradient training. "
                "Add loss node names to your trainrun config YAML."
            )

        # Build node lookup map
        node_map = {node.name: node for node in pipeline.nodes()}

        # Look up loss nodes by name
        loss_nodes = []
        for loss_name in trainrun_config.loss_nodes:
            if loss_name not in node_map:
                raise ValueError(
                    f"Loss node '{loss_name}' not found in pipeline. "
                    f"Available nodes: {', '.join(sorted(node_map.keys()))}"
                )
            loss_nodes.append(node_map[loss_name])

        # Look up metric nodes by name
        metric_nodes = []
        for metric_name in trainrun_config.metric_nodes:
            if metric_name not in node_map:
                raise ValueError(
                    f"Metric node '{metric_name}' not found in pipeline. "
                    f"Available nodes: {', '.join(sorted(node_map.keys()))}"
                )
            metric_nodes.append(node_map[metric_name])

        # Handle unfreeze_nodes from trainrun config
        if trainrun_config.unfreeze_nodes:
            pipeline.unfreeze_nodes_by_name(list(trainrun_config.unfreeze_nodes))

        # Validate we have trainable parameters
        has_trainable = any(p.requires_grad for p in pipeline.parameters())
        if not has_trainable:
            raise ValueError(
                "No trainable parameters found after unfreezing. "
                "Configure unfreeze_nodes in trainrun config to include trainable nodes."
            )

        return loss_nodes, metric_nodes

    def _cancelled_response(self, message: str) -> cuvis_ai_pb2.TrainResponse:
        """Build the terminal CANCELLED response of a stopped run."""
        return cuvis_ai_pb2.TrainResponse(
            context=cuvis_ai_pb2.Context(
                stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
                epoch=0,
                batch_idx=0,
                global_step=0,
            ),
            status=cuvis_ai_pb2.TRAIN_STATUS_CANCELLED,
            message=message,
        )

    def _create_progress_response(
        self,
        context_obj: Context,
        losses: dict,
        metrics: dict,
        status: str,
        message: str = "",
        event: TrainEvent | None = None,
    ) -> cuvis_ai_pb2.TrainResponse:
        """Map internal progress to proto TrainResponse."""
        status_map = {
            "running": cuvis_ai_pb2.TRAIN_STATUS_RUNNING,
            "complete": cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
            "error": cuvis_ai_pb2.TRAIN_STATUS_ERROR,
            "cancelled": cuvis_ai_pb2.TRAIN_STATUS_CANCELLED,
        }

        response = cuvis_ai_pb2.TrainResponse(
            context=cuvis_ai_pb2.Context(
                stage=_STAGE_TO_PROTO.get(
                    context_obj.stage, cuvis_ai_pb2.EXECUTION_STAGE_TRAIN
                ),
                epoch=context_obj.epoch,
                batch_idx=context_obj.batch_idx,
                global_step=context_obj.global_step,
            ),
            losses={k: float(v) for k, v in (losses or {}).items()},
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            status=status_map.get(status, cuvis_ai_pb2.TRAIN_STATUS_RUNNING),
            message=message,
        )
        if event is not None:
            response.event.CopyFrom(_event_to_proto(event))
        return response


_STAGE_TO_PROTO = {
    ExecutionStage.TRAIN: cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
    ExecutionStage.VAL: cuvis_ai_pb2.EXECUTION_STAGE_VAL,
    ExecutionStage.TEST: cuvis_ai_pb2.EXECUTION_STAGE_TEST,
    ExecutionStage.INFERENCE: cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE,
}
_SPLIT_STAGES = {
    "train": ExecutionStage.TRAIN,
    "val": ExecutionStage.VAL,
    "test": ExecutionStage.TEST,
}


_PASS_NAMES = {
    TrainHook.EPOCH_START: "Training",
    TrainHook.TRAIN_PROGRESS: "Training",
    TrainHook.VALIDATION_START: "Validating",
    TrainHook.VALIDATION_PROGRESS: "Validating",
    TrainHook.CALIBRATION_PROGRESS: "Calibrating decider thresholds",
}


def _event_message(event: TrainEvent) -> str:
    """The status line of an event response; the hooks with progress carry the counts."""
    if event.progress is not None:
        if event.hook in (
            TrainHook.STATISTICAL_NODE_START,
            TrainHook.STATISTICAL_PROGRESS,
        ):
            what = f"Fitting {event.node}"
        else:
            what = _PASS_NAMES.get(
                event.hook,
                f"Evaluating {event.split.value if event.split else 'split'}",
            )
        done, total = event.progress.done, event.progress.total
        counts = f"{done}/{total}" if total else f"{done}"
        return f"{what} ({counts})"
    return describe_event(event) or ""


def _event_to_proto(event: TrainEvent) -> cuvis_ai_pb2.TrainEvent:
    """The typed event as its proto message; enum members map by name."""
    proto = cuvis_ai_pb2.TrainEvent(
        hook=getattr(cuvis_ai_pb2, f"TRAIN_HOOK_{event.hook.name}"),
        calibration_pending=event.calibration_pending,
    )
    if event.split is not None:
        proto.split = _STAGE_TO_PROTO[event.split]
    if event.progress is not None:
        proto.progress.done = event.progress.done
        proto.progress.total = event.progress.total
    if event.epochs is not None:
        proto.epochs.current = event.epochs.current
        proto.epochs.total = event.epochs.total
    if event.node:
        proto.node = event.node
    if event.checkpoint is not None:
        restored = event.checkpoint
        proto.checkpoint.restored = restored.restored
        proto.checkpoint.path = restored.path
        proto.checkpoint.epoch = restored.epoch
        proto.checkpoint.monitor = restored.monitor
        proto.checkpoint.reason = restored.reason
        if restored.score is not None:
            proto.checkpoint.score = restored.score
    if event.calibration is not None:
        _calibration_to_proto(event.calibration, proto.calibration)
    if event.evaluation is not None:
        evaluation = event.evaluation
        proto.evaluation.split = _STAGE_TO_PROTO[evaluation.split]
        proto.evaluation.status = getattr(
            cuvis_ai_pb2, f"EVALUATION_STATUS_{evaluation.status.name}"
        )
        proto.evaluation.frames = evaluation.frames
        proto.evaluation.metrics.update(
            {key: float(value) for key, value in evaluation.metrics.items()}
        )
        proto.evaluation.error = evaluation.error
    return proto


def _calibration_to_proto(
    outcome: CalibrationOutcome, target: cuvis_ai_pb2.CalibrationOutcome
) -> None:
    """Fill the proto calibration outcome from the structured one."""
    target.split = _STAGE_TO_PROTO[_SPLIT_STAGES.get(outcome.split, ExecutionStage.VAL)]
    target.applicable = outcome.applicable
    target.phase_skip = (
        cuvis_ai_pb2.CALIBRATION_PHASE_SKIP_NONE
        if outcome.phase_skip is None
        else getattr(cuvis_ai_pb2, f"CALIBRATION_PHASE_SKIP_{outcome.phase_skip.name}")
    )
    target.phase_detail = outcome.phase_detail
    target.frames = outcome.frames
    target.anomalous_frames = outcome.anomalous_frames
    for record in outcome.deciders:
        decider = target.deciders.add()
        decider.name = record.name
        decider.class_name = record.class_name
        decider.status = getattr(
            cuvis_ai_pb2, f"DECIDER_CALIBRATION_STATUS_{record.status.name}"
        )
        decider.skip = (
            cuvis_ai_pb2.DECIDER_SKIP_NONE
            if record.skip is None
            else getattr(cuvis_ai_pb2, f"DECIDER_SKIP_{record.skip.name}")
        )
        decider.skip_detail = record.skip_detail
        for key, change in record.thresholds.items():
            entry = decider.thresholds[key]
            if change.before is not None:
                entry.before = change.before
            entry.after = change.after
        decider.metrics.update(
            {key: float(value) for key, value in record.metrics.items()}
        )
        decider.thresholds_present = record.thresholds_present


__all__ = ["TrainingService"]
