"""Pipeline introspection and visualization service component."""

from __future__ import annotations

import tempfile
from pathlib import Path

import grpc
from loguru import logger

from .error_handling import (
    get_session_and_pipeline,
    grpc_handler,
)
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
        resolved = get_session_and_pipeline(
            self.session_manager, request.session_id, context
        )
        if resolved is None:
            return cuvis_ai_pb2.GetPipelineInputsResponse()
        _, pipeline = resolved

        input_specs_dict = pipeline.get_input_specs()
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
        resolved = get_session_and_pipeline(
            self.session_manager, request.session_id, context
        )
        if resolved is None:
            return cuvis_ai_pb2.GetPipelineOutputsResponse()
        _, pipeline = resolved

        output_specs_dict = pipeline.get_output_specs()
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
        resolved = get_session_and_pipeline(
            self.session_manager, request.session_id, context
        )
        if resolved is None:
            return cuvis_ai_pb2.GetPipelineVisualizationResponse()
        _, pipeline = resolved

        from cuvis_ai_core.pipeline.visualizer import PipelineVisualizer

        format_type = (request.format or "png").lower()
        visualizer = PipelineVisualizer(pipeline)

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
        except Exception as exc:
            # No renderer on this host (no Graphviz binary, a broken install):
            # answer the DOT source and label it so, as the sessionless preview
            # does; a client that decodes by format must not get DOT as a PNG.
            logger.warning(
                f"Rendering the pipeline as {format_type} failed, returning DOT: {exc}"
            )
            dot_source = visualizer.to_graphviz()
            image_data = dot_source.encode("utf-8")
            format_type = "dot"

        return cuvis_ai_pb2.GetPipelineVisualizationResponse(
            image_data=image_data,
            format=format_type,
        )


__all__ = ["IntrospectionService"]
