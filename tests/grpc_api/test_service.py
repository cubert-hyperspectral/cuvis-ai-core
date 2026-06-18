"""Thin delegation tests for ``CuvisAIService``.

``service.py`` is a pure multiplexer: each RPC handler forwards to a
component service or an ``orchestrator_bridge`` forward. These tests drive
every delegating handler once (through the autouse in-memory orchestrator) so
the passthrough bodies are covered without standing up a real gRPC server.
"""

from __future__ import annotations

from unittest.mock import Mock

from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def test_delegating_handlers_forward_without_error():
    service = CuvisAIService()
    ctx = Mock()

    unary_calls = [
        ("GetParameterSchema", cuvis_ai_pb2.GetParameterSchemaRequest()),
        (
            "LoadPipelineWeights",
            cuvis_ai_pb2.LoadPipelineWeightsRequest(session_id="ghost"),
        ),
        (
            "SetTrainRunConfig",
            cuvis_ai_pb2.SetTrainRunConfigRequest(session_id="ghost"),
        ),
        ("Inference", cuvis_ai_pb2.InferenceRequest(session_id="ghost")),
        ("GetTrainStatus", cuvis_ai_pb2.GetTrainStatusRequest(session_id="ghost")),
        ("SavePipeline", cuvis_ai_pb2.SavePipelineRequest(session_id="ghost")),
        ("SaveTrainRun", cuvis_ai_pb2.SaveTrainRunRequest(session_id="ghost")),
        (
            "RestoreTrainRun",
            cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path="absent.yaml"),
        ),
        ("LoadPlugin", cuvis_ai_pb2.LoadPluginRequest(session_id="ghost")),
        (
            "ListLoadedPlugins",
            cuvis_ai_pb2.ListLoadedPluginsRequest(session_id="ghost"),
        ),
        (
            "GetPluginInfo",
            cuvis_ai_pb2.GetPluginInfoRequest(session_id="ghost", plugin_name="x"),
        ),
        (
            "ListAvailableNodes",
            cuvis_ai_pb2.ListAvailableNodesRequest(session_id="ghost"),
        ),
        ("ClearPluginCache", cuvis_ai_pb2.ClearPluginCacheRequest()),
    ]
    for handler, request in unary_calls:
        result = getattr(service, handler)(request, ctx)
        assert result is not None

    # Train returns a server-streaming iterator; consuming it runs the body.
    list(service.Train(cuvis_ai_pb2.TrainRequest(session_id="ghost"), ctx))
