import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2


class TestGetPipelineInputs:
    def test_get_pipeline_inputs(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )

        assert response.input_names
        assert "cube" in response.input_names
        assert response.input_specs

    def test_input_specs_have_details(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )

        cube_spec = response.input_specs["cube"]
        assert cube_spec.name == "cube"
        assert cube_spec.shape
        assert cube_spec.dtype != cuvis_ai_pb2.D_TYPE_UNSPECIFIED

    def test_invalid_session(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.GetPipelineInputs(
                cuvis_ai_pb2.GetPipelineInputsRequest(session_id="invalid")
            )

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


class TestGetPipelineOutputs:
    def test_get_pipeline_outputs(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )

        assert response.output_names
        assert response.output_specs

    def test_output_specs_have_details(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )

        first_output = next(iter(response.output_specs.values()))
        assert first_output.name
        assert first_output.dtype != cuvis_ai_pb2.D_TYPE_UNSPECIFIED


class TestGetPipelineVisualization:
    def test_get_visualization_png(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineVisualization(
            cuvis_ai_pb2.GetPipelineVisualizationRequest(
                session_id=session_id, format="png"
            )
        )

        assert response.image_data
        assert response.format == "png"

    def test_get_visualization_svg(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineVisualization(
            cuvis_ai_pb2.GetPipelineVisualizationRequest(
                session_id=session_id, format="svg"
            )
        )

        assert response.image_data
        assert response.format == "svg"

    def test_default_format_png(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineVisualization(
            cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id=session_id)
        )

        assert response.image_data
        assert response.format == "png"


def test_get_pipeline_visualization_from_config_content(grpc_stub):
    """Sessionless config preview: a ``config_content`` request renders straight from the
    YAML with no loaded session, so a client can preview a pipeline the moment it is
    selected. ``format=dot`` returns DOT source (no graphviz binary needed)."""
    yaml_cfg = (
        "metadata:\n  name: Preview\n"
        "nodes:\n"
        "- name: mask_cleanup\n"
        "  class_name: cuvis_ai.node.mask_ops.MaskRobustifier\n"
        "  hparams: {min_area: 1}\n"
        "connections: []\n"
    )
    response = grpc_stub.GetPipelineVisualization(
        cuvis_ai_pb2.GetPipelineVisualizationRequest(
            config_content=yaml_cfg, format="dot"
        )
    )

    assert response.format == "dot"
    assert b"digraph" in response.image_data


def test_get_pipeline_visualization_oversize_config_is_invalid_argument(grpc_stub):
    """The sessionless render caps input size; an oversize config is rejected as
    INVALID_ARGUMENT (the cap raises ValueError, which @grpc_handler maps), not
    rendered or silently degraded."""
    oversize = "metadata:\n  name: x\n" + "#" + "z" * 1_100_000  # > 1 MB
    with pytest.raises(grpc.RpcError) as exc:
        grpc_stub.GetPipelineVisualization(
            cuvis_ai_pb2.GetPipelineVisualizationRequest(
                config_content=oversize, format="dot"
            )
        )
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
