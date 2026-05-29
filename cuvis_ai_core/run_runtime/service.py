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
from typing import Iterator

import grpc
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from loguru import logger

from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.training_service import TrainingService
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
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
        self._trainrun_service = TrainRunService(self._session_manager)
        # Session id the parent handed us via InitializeSession. The
        # child runs at most one session per process, so caching the id
        # here lets RestoreTrainRun attach its pipeline to that exact
        # session even though RestoreTrainRunRequest carries no id.
        self._initialized_session_id: str | None = None
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

        self._initialized_session_id = request.session_id
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
        # The shared ``PipelineService.load_pipeline`` body builds the
        # pipeline against ``session.node_registry``. Plugins are
        # already registered there from ``InitializeSession`` so the
        # build resolves every node by class name without touching the
        # in-process install / clone / sys.path code paths.
        return self._pipeline_service.load_pipeline(request, context)

    def LoadPipelineWeights(
        self,
        request: cuvis_ai_pb2.LoadPipelineWeightsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineWeightsResponse:
        return self._pipeline_service.load_pipeline_weights(request, context)

    def SavePipeline(
        self,
        request: cuvis_ai_pb2.SavePipelineRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SavePipelineResponse:
        return self._pipeline_service.save_pipeline(request, context)

    def SaveTrainRun(
        self,
        request: cuvis_ai_pb2.SaveTrainRunRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SaveTrainRunResponse:
        return self._trainrun_service.save_train_run(request, context)

    def GetPipelineInputs(self, request, context):
        from cuvis_ai_core.grpc.introspection_service import IntrospectionService

        if not hasattr(self, "_introspection_service"):
            self._introspection_service = IntrospectionService(self._session_manager)
        return self._introspection_service.get_pipeline_inputs(request, context)

    def GetPipelineOutputs(self, request, context):
        from cuvis_ai_core.grpc.introspection_service import IntrospectionService

        if not hasattr(self, "_introspection_service"):
            self._introspection_service = IntrospectionService(self._session_manager)
        return self._introspection_service.get_pipeline_outputs(request, context)

    def GetPipelineVisualization(self, request, context):
        from cuvis_ai_core.grpc.introspection_service import IntrospectionService

        if not hasattr(self, "_introspection_service"):
            self._introspection_service = IntrospectionService(self._session_manager)
        return self._introspection_service.get_pipeline_visualization(request, context)

    def SetTrainRunConfig(self, request, context):
        return self._pipeline_service.set_train_run_config(request, context)

    def GetTrainStatus(self, request, context):
        return self._training_service.get_train_status(request, context)

    def RestoreTrainRun(
        self,
        request: cuvis_ai_pb2.RestoreTrainRunRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.RestoreTrainRunResponse:
        # The public RestoreTrainRunRequest carries no session_id; the
        # parent has already created its session and handed us the id
        # via InitializeSession. We reuse that id so the trainrun's
        # pipeline lands on the same session the parent reports back
        # to its caller.
        if self._initialized_session_id is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                "InitializeSession must be called before RestoreTrainRun."
            )
            return cuvis_ai_pb2.RestoreTrainRunResponse()
        return self._trainrun_service.restore_train_run(
            request,
            context,
            target_session_id=self._initialized_session_id,
        )

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
