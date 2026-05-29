"""gRPC service implementation delegating to modular components."""

from __future__ import annotations

from collections.abc import Iterable

from .config_service import ConfigService
from .discovery_service import DiscoveryService
from .inference_service import InferenceService
from .introspection_service import IntrospectionService
from .pipeline_service import PipelineService
from .profiling_service import ProfilingService
from .plugin_service import PluginService
from .session_manager import SessionManager
from .session_service import SessionService
from .training_service import TrainingService
from .trainrun_service import TrainRunService
from .v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc


class CuvisAIService(cuvis_ai_pb2_grpc.CuvisAIServiceServicer):
    """Main gRPC service class delegating to specialized service components."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self.session_manager = session_manager or SessionManager()

        # Initialize service components
        self.session_service = SessionService(self.session_manager)
        self.config_service = ConfigService(self.session_manager)
        self.pipeline_service = PipelineService(self.session_manager)
        self.inference_service = InferenceService(self.session_manager)
        self.training_service = TrainingService(self.session_manager)
        self.trainrun_service = TrainRunService(self.session_manager)
        self.discovery_service = DiscoveryService()
        self.introspection_service = IntrospectionService(self.session_manager)
        self.plugin_service = PluginService(self.session_manager)
        self.profiling_service = ProfilingService(self.session_manager)

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------
    def CreateSession(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
        return self.session_service.create_session(request, context)

    def CloseSession(self, request, context) -> cuvis_ai_pb2.CloseSessionResponse:
        return self.session_service.close_session(request, context)

    def SetSessionSearchPaths(
        self, request, context
    ) -> cuvis_ai_pb2.SetSessionSearchPathsResponse:
        return self.session_service.set_session_search_paths(request, context)

    # ------------------------------------------------------------------
    # Config resolution and validation
    # ------------------------------------------------------------------
    def ResolveConfig(self, request, context) -> cuvis_ai_pb2.ResolveConfigResponse:
        return self.config_service.resolve_config(request, context)

    def GetParameterSchema(
        self, request, context
    ) -> cuvis_ai_pb2.GetParameterSchemaResponse:
        return self.config_service.get_parameter_schema(request, context)

    def ValidateConfig(self, request, context) -> cuvis_ai_pb2.ValidateConfigResponse:
        return self.config_service.validate_config(request, context)

    # ------------------------------------------------------------------
    # Pipeline construction and training configuration
    # ------------------------------------------------------------------

    def LoadPipelineWeights(
        self, request, context
    ) -> cuvis_ai_pb2.LoadPipelineWeightsResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_load_pipeline_weights(
            self.session_manager, request, context
        )

    def SetTrainRunConfig(
        self, request, context
    ) -> cuvis_ai_pb2.SetTrainRunConfigResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_set_train_run_config(
            self.session_manager, request, context
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def Inference(self, request, context) -> cuvis_ai_pb2.InferenceResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_inference(
            self.session_manager, request, context
        )

    # ------------------------------------------------------------------
    # Pipeline Introspection
    # ------------------------------------------------------------------
    def GetPipelineInputs(
        self, request, context
    ) -> cuvis_ai_pb2.GetPipelineInputsResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_get_pipeline_inputs(
            self.session_manager, request, context
        )

    def GetPipelineOutputs(
        self, request, context
    ) -> cuvis_ai_pb2.GetPipelineOutputsResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_get_pipeline_outputs(
            self.session_manager, request, context
        )

    def GetPipelineVisualization(
        self, request, context
    ) -> cuvis_ai_pb2.GetPipelineVisualizationResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_get_pipeline_visualization(
            self.session_manager, request, context
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def Train(self, request, context) -> Iterable[cuvis_ai_pb2.TrainResponse]:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_train(
            self.session_manager, request, context
        )

    def GetTrainStatus(self, request, context) -> cuvis_ai_pb2.GetTrainStatusResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_get_train_status(
            self.session_manager, request, context
        )

    # ------------------------------------------------------------------
    # Pipeline Management (Model Deployment)
    # ------------------------------------------------------------------
    def SavePipeline(self, request, context) -> cuvis_ai_pb2.SavePipelineResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_save_pipeline(
            self.session_manager, request, context
        )

    def LoadPipeline(self, request, context) -> cuvis_ai_pb2.LoadPipelineResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_load_pipeline(
            self.session_manager, request, context
        )

    # ------------------------------------------------------------------
    # Experiment Management (Reproducibility)
    # ------------------------------------------------------------------
    def SaveTrainRun(self, request, context) -> cuvis_ai_pb2.SaveTrainRunResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_save_train_run(
            self.session_manager, request, context
        )

    def RestoreTrainRun(self, request, context) -> cuvis_ai_pb2.RestoreTrainRunResponse:
        from . import orchestrator_bridge

        return orchestrator_bridge.forward_restore_train_run(
            self.session_manager, request, context
        )

    # ------------------------------------------------------------------
    # Pipeline Discovery
    # ------------------------------------------------------------------
    def ListAvailablePipelines(
        self, request, context
    ) -> cuvis_ai_pb2.ListAvailablePipelinesResponse:
        return self.discovery_service.list_available_pipelines(request, context)

    def GetPipelineInfo(self, request, context) -> cuvis_ai_pb2.GetPipelineInfoResponse:
        return self.discovery_service.get_pipeline_info(request, context)

    # ------------------------------------------------------------------
    # Training Capabilities
    # ------------------------------------------------------------------
    def GetTrainingCapabilities(
        self, request, context
    ) -> cuvis_ai_pb2.GetTrainingCapabilitiesResponse:
        return self.training_service.get_training_capabilities(request, context)

    # ------------------------------------------------------------------
    # Plugin Management
    # ------------------------------------------------------------------
    def LoadPlugins(self, request, context) -> cuvis_ai_pb2.LoadPluginsResponse:
        return self.plugin_service.load_plugins(request, context)

    def ListLoadedPlugins(
        self, request, context
    ) -> cuvis_ai_pb2.ListLoadedPluginsResponse:
        return self.plugin_service.list_loaded_plugins(request, context)

    def GetPluginInfo(self, request, context) -> cuvis_ai_pb2.GetPluginInfoResponse:
        return self.plugin_service.get_plugin_info(request, context)

    def ListAvailableNodes(
        self, request, context
    ) -> cuvis_ai_pb2.ListAvailableNodesResponse:
        return self.plugin_service.list_available_nodes(request, context)

    def ClearPluginCache(
        self, request, context
    ) -> cuvis_ai_pb2.ClearPluginCacheResponse:
        return self.plugin_service.clear_plugin_cache(request, context)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------
    def SetProfiling(self, request, context) -> cuvis_ai_pb2.SetProfilingResponse:
        return self.profiling_service.set_profiling(request, context)

    def GetProfilingSummary(
        self, request, context
    ) -> cuvis_ai_pb2.GetProfilingSummaryResponse:
        return self.profiling_service.get_profiling_summary(request, context)


__all__ = ["CuvisAIService"]
