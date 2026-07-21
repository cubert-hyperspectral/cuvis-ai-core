"""Cooperative training cancellation: StopTrain, TRAIN_STATUS_CANCELLED, stop_event.

Covers the full cancel contract at the TrainingService level (no gRPC server):
mid-stream stops for both trainers, the between-phase window (the stop flag
survives until the next SetTrainRunConfig), client-drop cancellation, the
normal-completion guard, GetTrainStatus latest-progress, and the parent->child
StopTrain forwarding hop.
"""

from __future__ import annotations

import threading

import pytest
import torch
from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.grpc.orchestrator_bridge import (
    _InMemoryChildHandle,
    _InMemoryContext,
    forward_stop_train,
)
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.training_service import TrainingService
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.run_runtime.service import RunRuntimeServicer
from cuvis_ai_core.training.config import (
    DataConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_core.training.trainers import StatisticalTrainer, TrainingCancelled
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec
from tests.fixtures.mock_nodes import SimpleLossNode

# ---------------------------------------------------------------------------
# Tiny pipeline + datamodule
# ---------------------------------------------------------------------------


class _StatNode(Node):
    """Minimal statistical node: fits a mean over the ``features`` stream."""

    INPUT_SPECS = {"features": PortSpec(dtype=torch.float32, shape=(-1, -1))}
    OUTPUT_SPECS = {"normalized": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("mean", torch.zeros(1))
        self._requires_initial_fit_override = True

    def statistical_initialization(self, input_stream) -> None:
        """Consume the whole stream and fit the mean."""
        chunks = [inputs["features"] for inputs in input_stream]
        self.mean = torch.cat(chunks, dim=0).mean(dim=0, keepdim=True)
        self._requires_initial_fit_override = False
        self._statistically_initialized = True

    def forward(self, features, **kwargs):
        """Center the features and lift them to [B, F, 1, 1]."""
        out = features - self.mean
        return {"normalized": out.unsqueeze(-1).unsqueeze(-1)}

    def load(self, params: dict, serial_dir: str) -> None:
        """Nothing to load."""


class _Projection(Node):
    """Trainable linear layer so gradient training has parameters to move."""

    INPUT_SPECS = {"features": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}
    OUTPUT_SPECS = {"projected": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def __init__(self, dim: int = 4, **kwargs):
        self.dim = dim
        super().__init__(dim=dim, **kwargs)
        self.linear = torch.nn.Linear(dim, dim)

    @property
    def is_trainable(self) -> bool:
        """This node has trainable parameters."""
        return True

    def unfreeze(self) -> None:
        """Enable gradients on the projection parameters."""
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, features, **kwargs):
        """Project [B, F, 1, 1] features through the linear layer."""
        flat = features.squeeze(-1).squeeze(-1)
        return {"projected": self.linear(flat).unsqueeze(-1).unsqueeze(-1)}

    def load(self, params: dict, serial_dir: str) -> None:
        """Nothing to load."""


class _EventDataset(Dataset):
    """Fixed random features; optionally fires an event at a given item index."""

    def __init__(
        self,
        n: int,
        dim: int,
        fire_event: threading.Event | None = None,
        fire_at: int | None = None,
    ) -> None:
        gen = torch.Generator().manual_seed(7)
        self._x = torch.randn((n, dim), generator=gen)
        self._fire_event = fire_event
        self._fire_at = fire_at
        self._served = 0

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._served += 1
        if self._fire_event is not None and self._served == self._fire_at:
            self._fire_event.set()
        return {"features": self._x[idx]}


class _CancelDataModule(BaseCuvisAIDataModule):
    """Module-owned-splits datamodule; can fire a stop event mid-epoch.

    ``stop_event`` / ``stop_after_items`` arrive through ``DataConfig.params``
    (in-process only, never serialized): after ``stop_after_items`` train items
    have been served, the event fires — deterministically mid-fit.
    """

    DATA_MODULE_NAME = "cancelmod"

    def __init__(
        self,
        *,
        n: int = 16,
        dim: int = 4,
        stop_event: threading.Event | None = None,
        stop_after_items: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._n = n
        self._dim = dim
        self._stop_event = stop_event
        self._stop_after_items = stop_after_items

    @staticmethod
    def validate_params(params: dict) -> None:
        """All params are optional."""

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Serve the event-firing dataset for train, plain ones elsewhere."""
        if stage == "train":
            return _EventDataset(
                self._n, self._dim, self._stop_event, self._stop_after_items
            )
        return _EventDataset(max(2, self._n // 4), self._dim)


def _build_pipeline() -> tuple[CuvisPipeline, _StatNode, SimpleLossNode]:
    pipeline = CuvisPipeline("cancel_test")
    stat = _StatNode(name="stat")
    proj = _Projection(dim=4, name="projection")
    loss = SimpleLossNode(name="mse_loss")
    loss.execution_stages = {ExecutionStage.TRAIN, ExecutionStage.VAL}
    pipeline.connect(
        (stat.normalized, proj.features),
        (proj.outputs.projected, loss.predictions),
        (stat.normalized, loss.targets),
    )
    return pipeline, stat, loss


def _make_session(
    manager: SessionManager,
    *,
    stop_after_items: int | None = None,
    wire_event_to_session: bool = True,
) -> tuple[str, TrainingService]:
    """Create a session with the tiny pipeline + cancelmod datamodule attached."""
    service = TrainingService(manager)
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    pipeline, _, _ = _build_pipeline()
    session.pipeline = pipeline
    session.node_registry.data_modules["cancelmod"] = _CancelDataModule

    params: dict = {"n": 16, "dim": 4}
    if stop_after_items is not None:
        params["stop_after_items"] = stop_after_items
        if wire_event_to_session:
            params["stop_event"] = session.stop_event
    # Set on the session directly (RestoreTrainRun/SetTrainRunConfig path); an
    # Event in params can never cross the wire, this stays in-process.
    session.data_config = DataConfig(
        data_module="cancelmod", batch_size=4, num_workers=0, params=params
    )
    session.training_config = TrainingConfig(
        max_epochs=100,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        optimizer=OptimizerConfig(name="adam", lr=1e-2),
    )
    session.trainrun_config = TrainRunConfig(
        name="cancel_test",
        pipeline=None,
        data=session.data_config,
        training=session.training_config,
        loss_nodes=["mse_loss"],
        metric_nodes=[],
        unfreeze_nodes=["projection"],
    )
    return session_id, service


def _train_request(session_id: str, trainer_type: int) -> cuvis_ai_pb2.TrainRequest:
    return cuvis_ai_pb2.TrainRequest(session_id=session_id, trainer_type=trainer_type)


# ---------------------------------------------------------------------------
# StatisticalTrainer unit level
# ---------------------------------------------------------------------------


class TestStatisticalTrainerCancel:
    """Stop-event behavior inside StatisticalTrainer itself."""

    def test_preset_event_cancels_before_first_node(self):
        """An already-set event aborts fit() before any node is touched."""
        pipeline, stat, _ = _build_pipeline()
        dm = _CancelDataModule(n=8, dim=4)
        event = threading.Event()
        event.set()
        trainer = StatisticalTrainer(pipeline=pipeline, datamodule=dm, stop_event=event)
        with pytest.raises(TrainingCancelled):
            trainer.fit()
        assert not stat._statistically_initialized

    def test_event_fired_mid_stream_cancels_per_batch(self):
        """The input stream checks the event before every batch."""
        pipeline, _, _ = _build_pipeline()
        event = threading.Event()
        dm = _CancelDataModule(
            n=16, dim=4, stop_event=event, stop_after_items=2, batch_size=2
        )
        trainer = StatisticalTrainer(pipeline=pipeline, datamodule=dm, stop_event=event)
        with pytest.raises(TrainingCancelled):
            trainer.fit()

    def test_no_event_trains_to_completion(self):
        """Without a stop event the trainer behaves exactly as before."""
        pipeline, stat, _ = _build_pipeline()
        dm = _CancelDataModule(n=8, dim=4)
        StatisticalTrainer(pipeline=pipeline, datamodule=dm).fit()
        assert stat._statistically_initialized


# ---------------------------------------------------------------------------
# TrainingService stream level
# ---------------------------------------------------------------------------


class TestTrainStreamCancellation:
    """Terminal CANCELLED semantics on the Train stream."""

    def test_stop_mid_statistical_stream_yields_cancelled(self):
        manager = SessionManager()
        session_id, service = _make_session(manager, stop_after_items=3)
        ctx = _InMemoryContext()

        responses = list(
            service.train(
                _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL), ctx
            )
        )

        assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED
        session = manager.get_session(session_id)
        assert session.stop_event.is_set()
        assert (
            session.latest_train_response.status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED
        )

    def test_stop_between_phases_cancels_next_stream(self):
        """The stop flag survives a finished stream until SetTrainRunConfig."""
        manager = SessionManager()
        session_id, service = _make_session(manager)
        ctx = _InMemoryContext()

        # Phase 1 completes normally.
        first = list(
            service.train(
                _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL), ctx
            )
        )
        assert first[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

        # StopTrain lands in the between-phase window.
        stop = service.stop_train(
            cuvis_ai_pb2.StopTrainRequest(session_id=session_id), _InMemoryContext()
        )
        assert stop.accepted

        # The next phase must not start: immediate terminal CANCELLED.
        second = list(
            service.train(
                _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_GRADIENT),
                _InMemoryContext(),
            )
        )
        assert len(second) == 1
        assert second[0].status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED
        assert manager.get_session(session_id).stop_event.is_set()

    def test_set_train_run_config_clears_stop_flag(self):
        """SetTrainRunConfig is the run boundary: a new run starts un-cancelled."""
        manager = SessionManager()
        session_id, service = _make_session(manager)
        session = manager.get_session(session_id)
        pipeline_service = PipelineService(manager)

        service.stop_train(
            cuvis_ai_pb2.StopTrainRequest(session_id=session_id), _InMemoryContext()
        )
        assert session.stop_event.is_set()

        trainrun = TrainRunConfig(
            name="fresh_run",
            pipeline=None,
            data=DataConfig(
                data_module="cancelmod",
                batch_size=4,
                params={"n": 8, "dim": 4},
            ),
            training=session.training_config,
            loss_nodes=["mse_loss"],
        )
        response = pipeline_service.set_train_run_config(
            cuvis_ai_pb2.SetTrainRunConfigRequest(
                session_id=session_id, config=trainrun.to_proto()
            ),
            _InMemoryContext(),
        )
        assert response.success
        assert not session.stop_event.is_set()

        # And the fresh run trains to completion.
        responses = list(
            service.train(
                _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
                _InMemoryContext(),
            )
        )
        assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

    @pytest.mark.slow
    def test_stop_mid_gradient_stream_yields_cancelled(self):
        manager = SessionManager()
        session_id, service = _make_session(manager)
        ctx = _InMemoryContext()

        # Statistical init first so the pipeline is fit for gradient training.
        stat = list(
            service.train(
                _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
                _InMemoryContext(),
            )
        )
        assert stat[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

        responses = []
        for response in service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_GRADIENT), ctx
        ):
            responses.append(response)
            if len(responses) == 3:
                stop = service.stop_train(
                    cuvis_ai_pb2.StopTrainRequest(session_id=session_id),
                    _InMemoryContext(),
                )
                assert stop.accepted

        assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED
        # Cancelled long before the configured 100 epochs.
        assert max(r.context.epoch for r in responses) < 50

    def test_client_drop_callback_cancels_running_training(self):
        """Firing the RPC-termination callback mid-stream sets the stop flag."""
        manager = SessionManager()
        session_id, service = _make_session(manager)
        session = manager.get_session(session_id)
        ctx = _InMemoryContext()

        gen = service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL), ctx
        )
        first = next(gen)
        assert first.status == cuvis_ai_pb2.TRAIN_STATUS_RUNNING
        assert ctx.callbacks, "train() must register an RPC-termination callback"

        # Simulate the client dropping the stream while training runs.
        for callback in ctx.callbacks:
            callback()
        assert session.stop_event.is_set()

        remaining = list(gen)
        assert remaining[-1].status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED

    def test_generator_close_sets_stop_event(self):
        """gRPC closes the response generator on cancellation -> run cancelled."""
        manager = SessionManager()
        session_id, service = _make_session(manager)
        session = manager.get_session(session_id)

        gen = service.train(
            _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
            _InMemoryContext(),
        )
        next(gen)
        gen.close()
        assert session.stop_event.is_set()

    def test_normal_completion_does_not_cancel_next_phase(self):
        """The termination callback of a finished stream must NOT set the flag."""
        manager = SessionManager()
        session_id, service = _make_session(manager)
        session = manager.get_session(session_id)
        ctx = _InMemoryContext()

        responses = list(
            service.train(
                _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL), ctx
            )
        )
        assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

        # The RPC terminates after normal completion; grpc still fires the
        # callback. It must not cancel the run's next trainer phase.
        for callback in ctx.callbacks:
            callback()
        assert not session.stop_event.is_set()


# ---------------------------------------------------------------------------
# GetTrainStatus + StopTrain handler
# ---------------------------------------------------------------------------


class TestTrainStatusAndStopHandler:
    """GetTrainStatus reports real progress; StopTrain validates the session."""

    def test_status_before_any_training_is_unspecified(self):
        manager = SessionManager()
        session_id, service = _make_session(manager)

        status = service.get_train_status(
            cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id),
            _InMemoryContext(),
        )
        assert status.latest_progress.status == cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED
        assert "No training activity" in status.latest_progress.message

    def test_status_reports_latest_stream_response(self):
        manager = SessionManager()
        session_id, service = _make_session(manager)

        responses = list(
            service.train(
                _train_request(session_id, cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL),
                _InMemoryContext(),
            )
        )
        status = service.get_train_status(
            cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id),
            _InMemoryContext(),
        )
        assert status.latest_progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
        assert status.latest_progress.message == responses[-1].message

    def test_stop_train_unknown_session_not_accepted(self):
        service = TrainingService(SessionManager())
        ctx = _InMemoryContext()
        response = service.stop_train(
            cuvis_ai_pb2.StopTrainRequest(session_id="missing"), ctx
        )
        assert not response.accepted
        import grpc as _grpc

        assert ctx.code() == _grpc.StatusCode.NOT_FOUND


# ---------------------------------------------------------------------------
# Parent -> child forwarding
# ---------------------------------------------------------------------------


class TestStopTrainForwarding:
    """The parent's StopTrain reaches the child runtime's session state."""

    def test_forward_stop_train_reaches_child_session(self):
        child = RunRuntimeServicer()
        init = child.InitializeSession(
            cuvis_ai_pb2.InitializeSessionRequest(
                session_id="shared-id", resolved_plugins_json=b""
            ),
            _InMemoryContext(),
        )
        assert init.ok

        parent_manager = SessionManager()
        parent_manager.create_session_with_id("shared-id")
        parent_session = parent_manager.get_session("shared-id")
        parent_session.child_handle = _InMemoryChildHandle(child)

        ctx = _InMemoryContext()
        response = forward_stop_train(
            parent_manager,
            cuvis_ai_pb2.StopTrainRequest(session_id="shared-id"),
            ctx,
        )
        assert response.accepted
        child_session = child._session_manager.get_session("shared-id")
        assert child_session.stop_event.is_set()

    def test_forward_stop_train_without_child_is_precondition_failure(self):
        import grpc as _grpc

        parent_manager = SessionManager()
        session_id = parent_manager.create_session()
        ctx = _InMemoryContext()
        response = forward_stop_train(
            parent_manager,
            cuvis_ai_pb2.StopTrainRequest(session_id=session_id),
            ctx,
        )
        assert not response.accepted
        assert ctx.code() == _grpc.StatusCode.FAILED_PRECONDITION
