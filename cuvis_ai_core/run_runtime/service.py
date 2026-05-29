"""RunRuntimeServicer — the gRPC servicer hosted inside the child process.

Inside a composed venv every plugin is already an installed Python
package, so the child does NOT re-run the manifest-driven plugin
install path. ``InitializeSession`` instead receives the resolved
plugin dict the parent already computed and registers each plugin's
classes via :func:`load_preinstalled_plugins`. ``LoadPipeline``
builds the pipeline from the request bytes using the session's
already-populated ``NodeRegistry``; everything downstream
(``Inference`` / ``Train``) delegates to the existing service
classes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator, Mapping

import grpc
import torch
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from loguru import logger

from cuvis_ai_core.grpc.error_handling import get_session_or_error
from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.training_service import TrainingService
from cuvis_ai_core.pipeline.restore_preinstalled import load_preinstalled_plugins
from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig

PluginConfig = GitPluginConfig | LocalPluginConfig


class RunRuntimeServicer(cuvis_ai_pb2_grpc.RunRuntimeServicer):
    """The child runtime's gRPC servicer.

    Wraps a single :class:`SessionManager` and the existing service
    objects so request handlers reuse the in-process implementations
    they already trust.
    """

    def __init__(self, shutdown_event: threading.Event | None = None) -> None:
        self._session_manager = SessionManager()
        self._pipeline_service = PipelineService(self._session_manager)
        self._inference_service = InferenceService(self._session_manager)
        self._training_service = TrainingService(self._session_manager)
        # Threading event the __main__ entry point waits on; set by
        # StopRun so the gRPC server can shut down cleanly.
        self._shutdown_event = shutdown_event or threading.Event()

    @property
    def shutdown_event(self) -> threading.Event:
        return self._shutdown_event

    # ------------------------------------------------------------------
    # Internal init
    # ------------------------------------------------------------------

    def InitializeSession(
        self,
        request: cuvis_ai_pb2.InitializeSessionRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.InitializeSessionResponse:
        if not request.session_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session_id is required")
            return cuvis_ai_pb2.InitializeSessionResponse(ok=False)

        try:
            resolved_plugins = _decode_resolved_plugins(request.resolved_plugins_json)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"resolved_plugins_json could not be parsed: {exc}")
            return cuvis_ai_pb2.InitializeSessionResponse(ok=False)

        self._session_manager.create_session_with_id(request.session_id)
        try:
            session = self._session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to attach session inside child runtime: {exc}")
            return cuvis_ai_pb2.InitializeSessionResponse(ok=False)

        session.search_paths = list(request.search_paths)
        load_preinstalled_plugins(session.node_registry, resolved_plugins)
        # Record per-plugin catalog metadata for ListLoadedPlugins / GetPluginInfo.
        for name, cfg in resolved_plugins.items():
            session.registered_plugins[name] = cfg.model_dump()

        logger.info(
            f"Initialised child session {request.session_id} with "
            f"{len(resolved_plugins)} preinstalled plugins; "
            f"output_dir={request.output_dir!r}, scratch_dir={request.scratch_dir!r}"
        )
        return cuvis_ai_pb2.InitializeSessionResponse(ok=True)

    # ------------------------------------------------------------------
    # Pipeline materialisation (skip install loop; plugins preinstalled)
    # ------------------------------------------------------------------

    def LoadPipeline(
        self,
        request: cuvis_ai_pb2.LoadPipelineRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineResponse:
        session = get_session_or_error(
            self._session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        if not request.pipeline or not request.pipeline.config_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("pipeline.config_bytes is required")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        from cuvis_ai_core.pipeline.factory import PipelineBuilder
        from cuvis_ai_core.training.config import PipelineConfig

        try:
            config_dict = json.loads(request.pipeline.config_bytes)
            config_dict.pop("version", None)
            pipeline_config = PipelineConfig(**config_dict)

            pipeline = PipelineBuilder(
                node_registry=session.node_registry
            ).build_from_config(pipeline_config.to_dict())
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")

            self._session_manager.set_pipeline(
                request.session_id,
                pipeline,
                pipeline_config=pipeline_config,
            )
        except Exception as exc:
            logger.exception("Child LoadPipeline failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"LoadPipeline failed: {exc}")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        metadata_proto = (
            pipeline_config.metadata.to_proto()
            if getattr(pipeline_config, "metadata", None)
            else cuvis_ai_pb2.PipelineMetadata()
        )
        return cuvis_ai_pb2.LoadPipelineResponse(
            success=True,
            metadata=metadata_proto,
        )

    def LoadPipelineWeights(
        self,
        request: cuvis_ai_pb2.LoadPipelineWeightsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineWeightsResponse:
        return self._pipeline_service.load_pipeline_weights(request, context)

    def RestoreTrainRun(
        self,
        request: cuvis_ai_pb2.RestoreTrainRunRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.RestoreTrainRunResponse:
        # The child only mirrors the public RestoreTrainRun shape; the
        # actual handler lives on a public servicer this child does not
        # import. The parent currently always wraps RestoreTrainRun by
        # first parsing the trainrun YAML and calling LoadPipeline /
        # set_pipeline directly, so this RPC is a stub the parent does
        # not invoke yet. Returning UNIMPLEMENTED keeps the contract
        # explicit instead of silently misbehaving.
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(
            "RunRuntime.RestoreTrainRun is not yet wired; the parent handles "
            "trainrun restore by composing the env then forwarding a "
            "LoadPipeline. Wire this in Phase 3b follow-up if needed."
        )
        return cuvis_ai_pb2.RestoreTrainRunResponse()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def Inference(
        self,
        request: cuvis_ai_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.InferenceResponse:
        return self._inference_service.inference(request, context)

    def Train(
        self,
        request: cuvis_ai_pb2.TrainRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        yield from self._training_service.train(request, context)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def CloseSession(
        self,
        request: cuvis_ai_pb2.CloseSessionRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.CloseSessionResponse:
        try:
            self._session_manager.close_session(request.session_id)
        except ValueError:
            # Missing session is not an error here — the parent may have
            # already cleaned us up. Idempotent close keeps the lifecycle
            # easy to reason about.
            pass
        return cuvis_ai_pb2.CloseSessionResponse(success=True)

    def StopRun(
        self,
        request: cuvis_ai_pb2.StopRunRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.StopRunResponse:
        logger.info(
            f"StopRun received (session_id={request.session_id!r}, "
            f"grace_seconds={request.grace_seconds}); signalling shutdown."
        )
        if request.session_id:
            try:
                self._session_manager.close_session(request.session_id)
            except ValueError:
                pass
        self._shutdown_event.set()
        return cuvis_ai_pb2.StopRunResponse(ok=True)

    def HealthCheck(
        self,
        request: cuvis_ai_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.HealthCheckResponse:
        return cuvis_ai_pb2.HealthCheckResponse(
            status=cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
        )


def _decode_resolved_plugins(blob: bytes) -> dict[str, PluginConfig]:
    """Parse the JSON-serialised resolved plugin dict the parent sends.

    The parent computes the dict via
    :func:`cuvis_ai_core.utils.plugin_resolver.resolve_pipeline_plugins`
    and serialises it as
    ``{name: GitPluginConfig.model_dump() | LocalPluginConfig.model_dump()}``.
    Discriminating on the presence of ``repo`` vs ``path`` avoids any
    ambiguity in pydantic's union heuristics for the two config types.
    """
    if not blob:
        return {}
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise TypeError("resolved_plugins_json must decode to a dict")
    out: dict[str, PluginConfig] = {}
    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            raise TypeError(f"Plugin '{name}' config must be a dict, got {type(cfg)!r}")
        if "repo" in cfg:
            out[name] = GitPluginConfig(**cfg)
        elif "path" in cfg:
            out[name] = LocalPluginConfig(**cfg)
        else:
            raise ValueError(
                f"Plugin '{name}' config has neither 'repo' nor 'path' "
                f"keys; cannot infer GitPluginConfig vs LocalPluginConfig."
            )
    return out
