"""Tests for main gRPC service delegation wiring."""

from __future__ import annotations

from unittest.mock import Mock

from cuvis_ai_core.grpc import orchestrator_bridge
from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def test_main_service_forwards_profiling_methods(monkeypatch) -> None:
    """Profiling RPCs route through the orchestrator bridge, not a local service.

    Profiling state lives on the live pipeline, which only the child runtime
    holds; a parent-local ProfilingService call could never succeed.
    """
    service = CuvisAIService()
    request = object()
    context = object()

    set_response = cuvis_ai_pb2.SetProfilingResponse(profiling_enabled=True)
    summary_response = cuvis_ai_pb2.GetProfilingSummaryResponse()

    forward_set = Mock(return_value=set_response)
    forward_summary = Mock(return_value=summary_response)
    monkeypatch.setattr(orchestrator_bridge, "forward_set_profiling", forward_set)
    monkeypatch.setattr(
        orchestrator_bridge, "forward_get_profiling_summary", forward_summary
    )

    assert service.SetProfiling(request, context) is set_response
    assert service.GetProfilingSummary(request, context) is summary_response

    forward_set.assert_called_once_with(service.session_manager, request, context)
    forward_summary.assert_called_once_with(service.session_manager, request, context)
