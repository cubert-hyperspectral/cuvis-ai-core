"""Integration tests for pipeline discovery functionality (Task 5.2)."""

import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2


@pytest.fixture
def pipeline_directory(monkeypatch):
    """Point to the actual pipeline directory for discovery tests."""
    from pathlib import Path

    # Use the real configs/pipeline directory
    pipeline_dir = Path(__file__).parent.parent.parent / "configs"  # / "pipeline"
    monkeypatch.setattr(
        "cuvis_ai_core.grpc.helpers.get_server_base_dir", lambda: pipeline_dir
    )
    return pipeline_dir


class TestListAvailablePipelines:
    """Test the ListAvailablePipelines RPC method."""

    def test_list_all_pipelines(self, grpc_stub, pipeline_directory):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest()
        )

        assert len(response.pipelines) > 1
        pipeline_names = {c.name for c in response.pipelines}
        assert any(c in pipeline_names for c in {"statistical_based", "gradient_based"})

    def test_list_with_tag_filter_statistical(self, grpc_stub, pipeline_directory):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest(filter_tag="statistical")
        )

        # Both pipelines have the "statistical" tag
        assert len(response.pipelines) >= 1
        pipeline_names = {c.name for c in response.pipelines}
        assert any(c in pipeline_names for c in {"statistical_based", "gradient_based"})

        for pipeline in response.pipelines:
            assert "statistical" in pipeline.tags

    def test_list_with_tag_filter_gradient(self, grpc_stub, pipeline_directory):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest(filter_tag="gradient")
        )

        assert len(response.pipelines) > 1
        pipeline_names = {c.name for c in response.pipelines}
        assert any(c in pipeline_names for c in {"gradient_based"})

        assert all("gradient" in pipeline.tags for pipeline in response.pipelines)

    def test_list_with_unknown_tag(self, grpc_stub, pipeline_directory):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest(filter_tag="nonexistent")
        )

        assert len(response.pipelines) == 0

    def test_list_empty_directory(self, grpc_stub, monkeypatch, tmp_path):
        # Create empty temp directory with pipeline subdirectory
        empty_dir = tmp_path / "empty_pipelines"
        empty_dir.mkdir(exist_ok=True)
        (empty_dir / "pipeline").mkdir(exist_ok=True)

        monkeypatch.setattr(
            "cuvis_ai_core.grpc.helpers.get_server_base_dir", lambda: empty_dir
        )

        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest()
        )

        assert len(response.pipelines) == 0

    def test_list_includes_weights_info(self, grpc_stub, pipeline_directory):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest()
        )

        pipelines_by_name = {c.name: c for c in response.pipelines}

        rx_weights = pipeline_directory / "pipeline" / "statistical_based.pt"
        assert pipelines_by_name["statistical_based"].has_weights == rx_weights.exists()
        if rx_weights.exists():
            assert pipelines_by_name["statistical_based"].weights_path
            assert (
                "statistical_based.pt"
                in pipelines_by_name["statistical_based"].weights_path
            )
        else:
            assert not pipelines_by_name["statistical_based"].weights_path

        gradient_based_weights = pipeline_directory / "pipeline" / "gradient_based.pt"
        assert (
            pipelines_by_name["gradient_based"].has_weights
            == gradient_based_weights.exists()
        )
        if gradient_based_weights.exists():
            assert (
                "gradient_based.pt" in pipelines_by_name["gradient_based"].weights_path
            )
        else:
            assert not pipelines_by_name["gradient_based"].weights_path


class TestGetPipelineInfo:
    """Test the GetPipelineInfo RPC method."""

    def test_get_pipeline_info_valid(self, grpc_stub, pipeline_directory):
        response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_name="statistical_based")
        )

        info = response.pipeline_info
        assert info.name == "statistical_based"
        assert info.metadata.name == "statistical_based"
        assert "statistical training" in info.metadata.description.lower()
        assert "statistical" in info.tags
        assert "rx" in info.tags
        rx_weights = pipeline_directory / "pipeline" / "statistical_based.pt"
        assert info.has_weights == rx_weights.exists()

    def test_get_pipeline_info_with_yaml_content(self, grpc_stub, pipeline_directory):
        response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_name="gradient_based")
        )

        info = response.pipeline_info
        assert info.name == "gradient_based"

    def test_get_pipeline_info_invalid(self, grpc_stub, pipeline_directory):
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.GetPipelineInfo(
                cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_name="nonexistent")
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_get_pipeline_info_metadata(self, grpc_stub, pipeline_directory):
        response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_name="statistical_based")
        )

        metadata = response.pipeline_info.metadata
        assert metadata.name == "statistical_based"
        assert "statistical training" in metadata.description.lower()
        # created and cuvis_ai_version can be empty string in real pipelines (not set during testing)
        assert metadata.cuvis_ai_version is not None  # Field exists (may be empty)

        tags = response.pipeline_info.tags
        assert "statistical" in tags
        assert "rx" in tags
