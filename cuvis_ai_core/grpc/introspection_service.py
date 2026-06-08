"""Pipeline introspection and visualization service component."""

from __future__ import annotations

import tempfile
from pathlib import Path

import grpc

from .error_handling import get_session_or_error, grpc_handler
from .helpers import spec_to_tensor_spec
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class IntrospectionService:
    """Pipeline introspection helpers."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    @grpc_handler("Failed to get inputs")
    def get_pipeline_inputs(
        self,
        request: cuvis_ai_pb2.GetPipelineInputsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineInputsResponse:
        """Return pipeline entrypoint specifications for the session."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.GetPipelineInputsResponse()

        input_specs_dict = session.pipeline.get_input_specs()
        input_specs = {
            name: spec_to_tensor_spec(name, spec)
            for name, spec in input_specs_dict.items()
        }

        return cuvis_ai_pb2.GetPipelineInputsResponse(
            input_names=list(input_specs.keys()),
            input_specs=input_specs,
        )

    @grpc_handler("Failed to get outputs")
    def get_pipeline_outputs(
        self,
        request: cuvis_ai_pb2.GetPipelineOutputsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineOutputsResponse:
        """Return pipeline exit specifications for the session."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.GetPipelineOutputsResponse()

        output_specs_dict = session.pipeline.get_output_specs()
        output_specs = {
            name: spec_to_tensor_spec(name, spec)
            for name, spec in output_specs_dict.items()
        }

        return cuvis_ai_pb2.GetPipelineOutputsResponse(
            output_names=list(output_specs.keys()),
            output_specs=output_specs,
        )

    @grpc_handler("Failed to get pipeline visualization")
    def get_pipeline_visualization(
        self,
        request: cuvis_ai_pb2.GetPipelineVisualizationRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineVisualizationResponse:
        """Return a visualization of the session pipeline."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.GetPipelineVisualizationResponse()

        from cuvis_ai_core.pipeline.visualizer import PipelineVisualizer

        format_type = (request.format or "png").lower()
        visualizer = PipelineVisualizer(session.pipeline)

        try:
            if format_type in {"png", "svg"}:
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / f"pipeline.{format_type}"
                    rendered = visualizer.render_graphviz(
                        output_path=output_path, format=format_type
                    )
                    image_data = Path(rendered).read_bytes()
            elif format_type in {"dot", "graphviz"}:
                dot_source = visualizer.to_graphviz()
                image_data = dot_source.encode("utf-8")
            elif format_type in {"mermaid"}:
                mermaid_source = visualizer.to_mermaid()
                image_data = mermaid_source.encode("utf-8")
            else:
                raise ValueError(f"Unsupported visualization format: {format_type}")
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetPipelineVisualizationResponse()
        except Exception:
            # Fallback to a DOT string if rendering dependencies are unavailable
            dot_source = visualizer.to_graphviz()
            image_data = dot_source.encode("utf-8")

        return cuvis_ai_pb2.GetPipelineVisualizationResponse(
            image_data=image_data,
            format=format_type,
        )


__all__ = ["IntrospectionService"]
