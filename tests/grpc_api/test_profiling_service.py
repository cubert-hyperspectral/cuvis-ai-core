"""Tests for ProfilingService gRPC handler.

Follows the test_service_error_paths.py pattern: direct service method calls
with SessionManager and Mock gRPC context.
"""

from __future__ import annotations

from unittest.mock import Mock

import grpc
import torch

from cuvis_ai_core.grpc.profiling_service import ProfilingService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec


class _IdentityNode(Node):
    INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
    OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

    def forward(self, x, **kwargs):
        return {"y": x}


def _make_pipeline() -> CuvisPipeline:
    """Create a minimal pipeline for testing."""
    pipeline = CuvisPipeline("test")
    n1 = _IdentityNode()
    n2 = _IdentityNode()
    pipeline.connect(n1.outputs.y, n2.x)
    return pipeline


class TestSetProfilingErrors:
    """Test error paths for SetProfiling."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = ProfilingService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_invalid_session(self):
        request = cuvis_ai_pb2.SetProfilingRequest(
            session_id="nonexistent", enabled=True
        )
        self.service.set_profiling(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_no_pipeline(self):
        session_id = self.session_manager.create_session()
        request = cuvis_ai_pb2.SetProfilingRequest(session_id=session_id, enabled=True)
        self.service.set_profiling(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)

    def test_negative_skip_first_n(self):
        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)
        session.pipeline = _make_pipeline()

        request = cuvis_ai_pb2.SetProfilingRequest(
            session_id=session_id, enabled=True, skip_first_n=-1
        )
        self.service.set_profiling(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)


class TestSetProfilingSuccess:
    """Test success paths for SetProfiling."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = ProfilingService(self.session_manager)
        self.ctx = Mock()
        self.session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(self.session_id)
        session.pipeline = _make_pipeline()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_enable(self):
        request = cuvis_ai_pb2.SetProfilingRequest(
            session_id=self.session_id, enabled=True
        )
        response = self.service.set_profiling(request, self.ctx)
        assert response.profiling_enabled is True

        session = self.session_manager.get_session(self.session_id)
        assert session.pipeline.profiling_enabled is True

    def test_disable(self):
        # Enable first
        self.service.set_profiling(
            cuvis_ai_pb2.SetProfilingRequest(session_id=self.session_id, enabled=True),
            self.ctx,
        )
        # Disable
        response = self.service.set_profiling(
            cuvis_ai_pb2.SetProfilingRequest(session_id=self.session_id, enabled=False),
            self.ctx,
        )
        assert response.profiling_enabled is False

    def test_optional_fields_default(self):
        """Omitting optional fields should use Python defaults."""
        request = cuvis_ai_pb2.SetProfilingRequest(
            session_id=self.session_id, enabled=True
        )
        response = self.service.set_profiling(request, self.ctx)
        assert response.profiling_enabled is True

        session = self.session_manager.get_session(self.session_id)
        assert session.pipeline._synchronize_cuda is False

    def test_reset_clears_stats(self):
        """SetProfiling(reset=True) should clear accumulated stats."""
        session = self.session_manager.get_session(self.session_id)

        # Enable and run forward
        session.pipeline.set_profiling(enabled=True)
        session.pipeline.forward(batch={"x": torch.tensor([1.0])})
        assert len(session.pipeline.get_profiling_summary()) > 0

        # Reset via gRPC
        request = cuvis_ai_pb2.SetProfilingRequest(
            session_id=self.session_id, enabled=True, reset=True
        )
        self.service.set_profiling(request, self.ctx)
        assert session.pipeline.get_profiling_summary() == []


class TestGetProfilingSummaryErrors:
    """Test error paths for GetProfilingSummary."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = ProfilingService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_invalid_session(self):
        request = cuvis_ai_pb2.GetProfilingSummaryRequest(session_id="nonexistent")
        self.service.get_profiling_summary(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_no_pipeline(self):
        session_id = self.session_manager.create_session()
        request = cuvis_ai_pb2.GetProfilingSummaryRequest(session_id=session_id)
        self.service.get_profiling_summary(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)


class TestGetProfilingSummarySuccess:
    """Test success paths for GetProfilingSummary."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = ProfilingService(self.session_manager)
        self.ctx = Mock()
        self.session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(self.session_id)
        session.pipeline = _make_pipeline()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_empty_when_profiling_disabled(self):
        request = cuvis_ai_pb2.GetProfilingSummaryRequest(session_id=self.session_id)
        response = self.service.get_profiling_summary(request, self.ctx)
        assert len(response.node_stats) == 0

    def test_returns_stats_after_forward(self):
        session = self.session_manager.get_session(self.session_id)
        session.pipeline.set_profiling(enabled=True)
        session.pipeline.forward(batch={"x": torch.tensor([1.0])})

        request = cuvis_ai_pb2.GetProfilingSummaryRequest(session_id=self.session_id)
        response = self.service.get_profiling_summary(request, self.ctx)
        assert len(response.node_stats) == 2  # Two nodes in pipeline

        for stat in response.node_stats:
            assert stat.count == 1
            assert stat.mean_ms > 0
            assert stat.stage == cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE

    def test_repeated_forward_increases_count(self):
        session = self.session_manager.get_session(self.session_id)
        session.pipeline.set_profiling(enabled=True)

        batch = {"x": torch.tensor([1.0])}
        session.pipeline.forward(batch=batch)
        session.pipeline.forward(batch=batch)
        session.pipeline.forward(batch=batch)

        request = cuvis_ai_pb2.GetProfilingSummaryRequest(session_id=self.session_id)
        response = self.service.get_profiling_summary(request, self.ctx)
        assert response.node_stats[0].count == 3

    def test_stage_filter(self):
        session = self.session_manager.get_session(self.session_id)
        # The identity node has ALWAYS execution stages, so it runs in both
        session.pipeline.set_profiling(enabled=True)
        session.pipeline.forward(
            batch={"x": torch.tensor([1.0])}, stage=ExecutionStage.INFERENCE
        )
        session.pipeline.forward(
            batch={"x": torch.tensor([1.0])}, stage=ExecutionStage.TRAIN
        )

        # Filter by inference
        request = cuvis_ai_pb2.GetProfilingSummaryRequest(
            session_id=self.session_id,
            stage=cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE,
        )
        response = self.service.get_profiling_summary(request, self.ctx)
        assert len(response.node_stats) == 2  # 2 nodes, inference only
        assert all(
            s.stage == cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE
            for s in response.node_stats
        )

        # No filter — should return both stages × 2 nodes = 4
        request_all = cuvis_ai_pb2.GetProfilingSummaryRequest(
            session_id=self.session_id
        )
        response_all = self.service.get_profiling_summary(request_all, self.ctx)
        assert len(response_all.node_stats) == 4
