"""Tests for main gRPC service delegation wiring."""

from __future__ import annotations

from unittest.mock import Mock

from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def test_main_service_delegates_profiling_methods() -> None:
    service = CuvisAIService()
    request = object()
    context = object()

    set_response = cuvis_ai_pb2.SetProfilingResponse(profiling_enabled=True)
    summary_response = cuvis_ai_pb2.GetProfilingSummaryResponse()

    service.profiling_service = Mock()
    service.profiling_service.set_profiling.return_value = set_response
    service.profiling_service.get_profiling_summary.return_value = summary_response

    assert service.SetProfiling(request, context) is set_response
    assert service.GetProfilingSummary(request, context) is summary_response

    service.profiling_service.set_profiling.assert_called_once_with(request, context)
    service.profiling_service.get_profiling_summary.assert_called_once_with(
        request, context
    )
