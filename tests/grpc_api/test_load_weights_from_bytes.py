"""LoadPipelineWeights with client bytes: loaded from memory, refused when not plain data."""

import tempfile
from dataclasses import dataclass
from unittest.mock import Mock

import grpc
import pytest
import torch

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


@dataclass
class _Canary:
    """A harmless class client bytes may not carry."""

    label: str = "canary"


@pytest.fixture
def no_temp_files(monkeypatch):
    """Arm it once the test's own files are saved: any temporary file after that fails the test.

    Saving a pipeline writes through a staging file, so the guard waits for the setup.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("client weight bytes must not be written to disk")

    def arm():
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)
        monkeypatch.setattr(tempfile, "mkstemp", refuse)

    return arm


@pytest.fixture
def session(mock_statistical_trainable_node):
    """A session holding a fitted single-node pipeline, plus the service and a context."""
    manager = SessionManager()
    service = PipelineService(manager)
    pipeline = CuvisPipeline("bytes")
    node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
    pipeline._graph.add_node(node)
    session_id = manager.create_session()
    manager.get_session(session_id).pipeline = pipeline
    return service, session_id, pipeline, node, Mock()


def test_bytes_load_from_memory_and_restore_the_state(session, tmp_path, no_temp_files):
    service, session_id, pipeline, node, ctx = session
    node.statistical_initialization(iter([{"data": torch.randn(1, 4, 4, 3)}]))
    config_path = tmp_path / "fitted.yaml"
    pipeline.save_to_file(config_path)
    blob = config_path.with_suffix(".pt").read_bytes()
    expected = node.fitted_mean.clone()
    node.fitted_mean.zero_()
    no_temp_files()

    response = service.load_pipeline_weights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id, weights_bytes=blob
        ),
        ctx,
    )

    assert response.success is True
    assert response.resolved_path == "(client-bytes)"
    assert torch.equal(node.fitted_mean, expected)


def test_damaged_archive_bytes_answer_invalid_argument(session, tmp_path):
    """A truncated upload is the client's to fix; the next valid upload still loads."""
    service, session_id, pipeline, node, ctx = session
    node.statistical_initialization(iter([{"data": torch.randn(1, 4, 4, 3)}]))
    config_path = tmp_path / "fitted.yaml"
    pipeline.save_to_file(config_path)
    blob = config_path.with_suffix(".pt").read_bytes()
    expected = node.fitted_mean.clone()
    node.fitted_mean.zero_()

    refused = service.load_pipeline_weights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id, weights_bytes=blob[:-64]
        ),
        ctx,
    )

    assert refused.success is False
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
    assert torch.count_nonzero(node.fitted_mean) == 0

    loaded = service.load_pipeline_weights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id, weights_bytes=blob
        ),
        Mock(),
    )

    assert loaded.success is True
    assert torch.equal(node.fitted_mean, expected)


def test_bytes_with_a_non_tensor_object_are_refused(session, tmp_path):
    service, session_id, _, node, ctx = session
    before = node.fitted_mean.clone()
    blob = tmp_path / "canary.pt"
    torch.save({"state_dict": {node.name: {"fitted_mean": _Canary()}}}, blob)

    response = service.load_pipeline_weights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id, weights_bytes=blob.read_bytes()
        ),
        ctx,
    )

    assert response.success is False
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
    assert "_Canary" in ctx.set_details.call_args.args[0]
    assert torch.equal(node.fitted_mean, before)
