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
        pipeline_paths = {c.pipeline_path for c in response.pipelines}
        assert any(
            c in pipeline_paths
            for c in {"statistical_based.yaml", "gradient_based.yaml"}
        )

    def test_list_with_tag_filter_statistical(self, grpc_stub, pipeline_directory):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest(filter_tag="statistical")
        )

        # Both pipelines have the "statistical" tag
        assert len(response.pipelines) >= 1
        pipeline_paths = {c.pipeline_path for c in response.pipelines}
        assert any(
            c in pipeline_paths
            for c in {"statistical_based.yaml", "gradient_based.yaml"}
        )

        for pipeline in response.pipelines:
            assert "statistical" in pipeline.metadata.tags

    def test_list_with_tag_filter_gradient(self, grpc_stub, pipeline_directory):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest(filter_tag="gradient")
        )

        assert len(response.pipelines) > 1
        pipeline_paths = {c.pipeline_path for c in response.pipelines}
        assert any(c in pipeline_paths for c in {"gradient_based.yaml"})

        assert all(
            "gradient" in pipeline.metadata.tags for pipeline in response.pipelines
        )

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

        pipelines_by_path = {c.pipeline_path: c for c in response.pipelines}

        rx_weights = pipeline_directory / "pipeline" / "statistical_based.pt"
        assert (
            bool(pipelines_by_path["statistical_based.yaml"].weights_path)
            == rx_weights.exists()
        )
        if rx_weights.exists():
            assert pipelines_by_path["statistical_based.yaml"].weights_path
            assert (
                "statistical_based.pt"
                in pipelines_by_path["statistical_based.yaml"].weights_path
            )
        else:
            assert not pipelines_by_path["statistical_based.yaml"].weights_path

        gradient_based_weights = pipeline_directory / "pipeline" / "gradient_based.pt"
        assert (
            bool(pipelines_by_path["gradient_based.yaml"].weights_path)
            == gradient_based_weights.exists()
        )
        if gradient_based_weights.exists():
            assert (
                "gradient_based.pt"
                in pipelines_by_path["gradient_based.yaml"].weights_path
            )
        else:
            assert not pipelines_by_path["gradient_based.yaml"].weights_path


class TestSubdirectoryPipelineDiscovery:
    """Test discovery of pipelines in subdirectories."""

    @pytest.fixture
    def nested_pipeline_dir(self, monkeypatch, tmp_path):
        """Create a pipeline directory with subdirectories."""
        base = tmp_path / "configs"
        base.mkdir()
        pipeline_dir = base / "pipeline"
        pipeline_dir.mkdir()

        # Top-level pipeline
        (pipeline_dir / "top_level.yaml").write_text(
            "metadata:\n  name: top_level\n  tags: [top]\nnodes: []\nconnections: []\n"
        )

        # Subdirectory pipeline
        subdir = pipeline_dir / "anomaly" / "adaclip"
        subdir.mkdir(parents=True)
        (subdir / "baseline.yaml").write_text(
            "metadata:\n  name: baseline\n  tags: [anomaly]\nnodes: []\nconnections: []\n"
        )
        (subdir / "baseline.pt").write_bytes(b"fake weights")

        # Another subdirectory
        rxdir = pipeline_dir / "anomaly" / "rx"
        rxdir.mkdir(parents=True)
        (rxdir / "statistical.yaml").write_text(
            "metadata:\n  name: statistical\n  tags: [anomaly, statistical]\nnodes: []\nconnections: []\n"
        )

        monkeypatch.setattr(
            "cuvis_ai_core.grpc.helpers.get_server_base_dir", lambda: base
        )
        return base

    def test_discovers_subdirectory_pipelines(self, grpc_stub, nested_pipeline_dir):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest()
        )

        names = {p.pipeline_path for p in response.pipelines}
        assert "top_level.yaml" in names
        assert "anomaly/adaclip/baseline.yaml" in names
        assert "anomaly/rx/statistical.yaml" in names
        assert len(response.pipelines) == 3

    def test_subdirectory_names_are_relative_paths(
        self, grpc_stub, nested_pipeline_dir
    ):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest()
        )

        names = {p.pipeline_path for p in response.pipelines}
        # Paths use forward slashes and include .yaml extension.
        for name in names:
            assert name.endswith(".yaml")
            assert "\\" not in name

    def test_subdirectory_tag_filter(self, grpc_stub, nested_pipeline_dir):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest(filter_tag="anomaly")
        )

        names = {p.pipeline_path for p in response.pipelines}
        assert "anomaly/adaclip/baseline.yaml" in names
        assert "anomaly/rx/statistical.yaml" in names
        assert "top_level.yaml" not in names

    def test_subdirectory_weights_detected(self, grpc_stub, nested_pipeline_dir):
        response = grpc_stub.ListAvailablePipelines(
            cuvis_ai_pb2.ListAvailablePipelinesRequest()
        )

        by_path = {p.pipeline_path: p for p in response.pipelines}
        assert bool(by_path["anomaly/adaclip/baseline.yaml"].weights_path) is True
        assert bool(by_path["anomaly/rx/statistical.yaml"].weights_path) is False

    def test_get_info_for_subdirectory_pipeline(self, grpc_stub, nested_pipeline_dir):
        response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(
                pipeline_path="anomaly/adaclip/baseline.yaml"
            )
        )

        info = response.pipeline_info
        assert info.pipeline_path == "anomaly/adaclip/baseline.yaml"
        assert bool(info.weights_path) is True


class TestGetPipelineInfo:
    """Test the GetPipelineInfo RPC method."""

    def test_get_pipeline_info_valid(self, grpc_stub, pipeline_directory):
        response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_path="statistical_based.yaml")
        )

        info = response.pipeline_info
        assert info.pipeline_path == "statistical_based.yaml"
        assert info.metadata.name == "statistical_based"
        assert "statistical training" in info.metadata.description.lower()
        assert "statistical" in info.metadata.tags
        assert "rx" in info.metadata.tags
        rx_weights = pipeline_directory / "pipeline" / "statistical_based.pt"
        assert bool(info.weights_path) == rx_weights.exists()

    def test_get_pipeline_info_with_yaml_content(self, grpc_stub, pipeline_directory):
        response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_path="gradient_based.yaml")
        )

        info = response.pipeline_info
        assert info.pipeline_path == "gradient_based.yaml"

    def test_get_pipeline_info_invalid(self, grpc_stub, pipeline_directory):
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.GetPipelineInfo(
                cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_path="nonexistent.yaml")
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_get_pipeline_info_rejects_absolute_path(
        self, grpc_stub, pipeline_directory
    ):
        absolute_path = (
            pipeline_directory / "pipeline" / "statistical_based.yaml"
        ).resolve()
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.GetPipelineInfo(
                cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_path=str(absolute_path))
            )
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_get_pipeline_info_rejects_missing_extension(
        self, grpc_stub, pipeline_directory
    ):
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.GetPipelineInfo(
                cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_path="statistical_based")
            )
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_get_pipeline_info_metadata(self, grpc_stub, pipeline_directory):
        response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_path="statistical_based.yaml")
        )

        metadata = response.pipeline_info.metadata
        assert metadata.name == "statistical_based"
        assert "statistical training" in metadata.description.lower()
        # created and cuvis_ai_version can be empty string in real pipelines (not set during testing)
        assert metadata.cuvis_ai_version is not None  # Field exists (may be empty)

        tags = response.pipeline_info.metadata.tags
        assert "statistical" in tags
        assert "rx" in tags
