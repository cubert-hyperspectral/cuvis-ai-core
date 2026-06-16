"""Pipeline construction and management service component."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import grpc
import torch

from cuvis_ai_core.training.config import TrainRunConfig

from . import helpers
from .error_handling import get_session_or_error, grpc_handler, require_pipeline
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class PipelineService:
    """Pipeline construction, weight loading, and persistence."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    @grpc_handler("Failed to load weights")
    def load_pipeline_weights(
        self,
        request: cuvis_ai_pb2.LoadPipelineWeightsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineWeightsResponse:
        """Load weights into an existing pipeline (path or raw bytes)."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)

        if not require_pipeline(session, context):
            return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)

        strict = request.strict if request.HasField("strict") else True
        resolved_path = ""

        if request.HasField("weights_path") and request.weights_path:
            resolved = helpers.find_weights_file(
                request.weights_path, session.search_paths
            )
            session.pipeline._restore_weights_from_checkpoint(
                weights_path=str(resolved),
                strict_weight_loading=strict,
            )
            resolved_path = str(resolved)
        elif request.HasField("weights_bytes") and request.weights_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(request.weights_bytes)
                tmp_path = Path(tmp.name)

            try:
                session.pipeline._restore_weights_from_checkpoint(
                    weights_path=str(tmp_path),
                    strict_weight_loading=strict,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            resolved_path = "(client-bytes)"
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Provide either weights_path or weights_bytes")
            return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)

        return cuvis_ai_pb2.LoadPipelineWeightsResponse(
            success=True, resolved_path=resolved_path
        )

    @grpc_handler("Failed to set train run config")
    def set_train_run_config(
        self,
        request: cuvis_ai_pb2.SetTrainRunConfigRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SetTrainRunConfigResponse:
        """Attach data/training/trainrun config to a session with an already-built pipeline.

        Pipeline construction is an explicit step: callers must call
        ``LoadPipeline`` (or ``RestoreTrainRun``) before
        ``SetTrainRunConfig``. A trainrun config carrying a ``pipeline:``
        reference is rejected — there is exactly one entry point for
        pipeline creation, not two.
        """
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

        if not request.config.config_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("trainrun config_bytes is required")
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

        trainrun_config = TrainRunConfig.from_proto(request.config)

        if session.pipeline is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                "No pipeline attached to the session. Call LoadPipeline "
                "(or RestoreTrainRun) before SetTrainRunConfig."
            )
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

        if trainrun_config.pipeline is not None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                "Trainrun config carries a 'pipeline:' reference; "
                "SetTrainRunConfig does not build pipelines. Remove the "
                "pipeline reference and call LoadPipeline explicitly first."
            )
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

        session.data_config = trainrun_config.data
        session.training_config = trainrun_config.training
        session.trainrun_config = trainrun_config

        return cuvis_ai_pb2.SetTrainRunConfigResponse(
            success=True, pipeline_from_config=False
        )

    @grpc_handler("Failed to save pipeline")
    def save_pipeline(
        self,
        request: cuvis_ai_pb2.SavePipelineRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SavePipelineResponse:
        """Save trained pipeline (structure + weights) to disk."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.SavePipelineResponse(success=False)

        if not request.pipeline_path:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("pipeline_path is required")
            return cuvis_ai_pb2.SavePipelineResponse(success=False)

        from datetime import datetime

        from cuvis_ai_core import __version__
        from cuvis_ai_core.training.config import PipelineMetadata

        # Use resolve_pipeline_save_path for consistent path resolution
        pipeline_path = helpers.resolve_pipeline_save_path(request.pipeline_path)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)

        # Build metadata from proto or defaults
        metadata = PipelineMetadata(
            name=request.metadata.name if request.metadata.name else pipeline_path.stem,
            description=request.metadata.description
            if request.metadata.description
            else "",
            created=request.metadata.created
            if request.metadata.created
            else datetime.now().isoformat(),
            cuvis_ai_version=request.metadata.cuvis_ai_version
            if request.metadata.cuvis_ai_version
            else __version__,
            tags=list(request.metadata.tags) if request.metadata.tags else [],
            author=request.metadata.author if request.metadata.author else "",
        )

        # Save pipeline using CuvisPipeline.save_to_file
        session.pipeline.save_to_file(
            str(pipeline_path),
            metadata=metadata,
        )

        # Compute weights path (save_to_file creates it as pipeline_path.with_suffix('.pt'))
        weights_path = pipeline_path.with_suffix(".pt")

        return cuvis_ai_pb2.SavePipelineResponse(
            success=True,
            pipeline_path=str(pipeline_path),
            weights_path=str(weights_path),
        )

    @grpc_handler("Failed to load pipeline")
    def load_pipeline(
        self,
        request: cuvis_ai_pb2.LoadPipelineRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineResponse:
        """Build a pipeline from resolved config bytes and attach it to the session.

        This method is the **in-process** body that runs inside the
        child runtime. Plugins must already be registered on
        ``session.node_registry`` (via
        :meth:`cuvis_ai_core.utils.node_registry.NodeRegistry.register_preinstalled`,
        which the child invokes from ``InitializeSession``). Plugin
        resolution and dependency installation happen at the
        orchestrator level — see
        :mod:`cuvis_ai_core.grpc.orchestrator_bridge`.
        """
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        if not request.pipeline or not request.pipeline.config_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("pipeline.config_bytes is required")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        from cuvis_ai_core.pipeline.factory import PipelineBuilder
        from cuvis_ai_core.training.config import PipelineConfig

        config_dict = json.loads(request.pipeline.config_bytes)
        # Some YAML sources include a top-level version field that is not part of the schema.
        config_dict.pop("version", None)
        pipeline_config = PipelineConfig(**config_dict)

        pipeline = PipelineBuilder(
            node_registry=session.node_registry
        ).build_from_config(pipeline_config.to_dict())
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")

        self.session_manager.set_pipeline(
            request.session_id,
            pipeline,
            pipeline_config=pipeline_config,
        )

        metadata_proto = (
            pipeline_config.metadata.to_proto()
            if getattr(pipeline_config, "metadata", None)
            else cuvis_ai_pb2.PipelineMetadata()
        )
        return cuvis_ai_pb2.LoadPipelineResponse(success=True, metadata=metadata_proto)


__all__ = ["PipelineService"]
