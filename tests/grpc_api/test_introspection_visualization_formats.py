"""The text formats of ``GetPipelineVisualization`` on a session pipeline."""

import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2


def _visualize(grpc_stub, session_id: str, fmt: str):
    return grpc_stub.GetPipelineVisualization(
        cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id=session_id, format=fmt)
    )


def test_dot_returns_the_graphviz_source(grpc_stub, session):
    response = _visualize(grpc_stub, session(), "dot")
    assert response.format == "dot"
    assert "digraph" in response.image_data.decode("utf-8")


def test_graphviz_alias_is_lower_cased(grpc_stub, session):
    response = _visualize(grpc_stub, session(), "GRAPHVIZ")
    assert response.format == "graphviz"
    assert "digraph" in response.image_data.decode("utf-8")


def test_mermaid_returns_a_flowchart(grpc_stub, session):
    response = _visualize(grpc_stub, session(), "mermaid")
    assert response.format == "mermaid"
    assert response.image_data.decode("utf-8").startswith("flowchart")


def test_unsupported_format_is_invalid_argument(grpc_stub, session):
    with pytest.raises(grpc.RpcError) as exc_info:
        _visualize(grpc_stub, session(), "bmp")
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Unsupported visualization format: bmp" in exc_info.value.details()
