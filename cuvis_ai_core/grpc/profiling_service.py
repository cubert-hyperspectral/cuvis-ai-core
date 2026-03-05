"""Profiling service component for runtime node profiling via gRPC."""

from __future__ import annotations

import grpc

from cuvis_ai_schemas.enums import ExecutionStage

from .error_handling import get_session_or_error, grpc_handler, require_pipeline
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2

# Mapping from Python ExecutionStage string values to proto enum integers.
# Proto enum names are EXECUTION_STAGE_<UPPER>, Python StrEnum values are lowercase.
_STAGE_STR_TO_PROTO: dict[str, int] = {
    "train": cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
    "val": cuvis_ai_pb2.EXECUTION_STAGE_VAL,
    "test": cuvis_ai_pb2.EXECUTION_STAGE_TEST,
    "inference": cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE,
}

_STAGE_PROTO_TO_STR: dict[int, str] = {v: k for k, v in _STAGE_STR_TO_PROTO.items()}


class ProfilingService:
    """Runtime profiling operations for Cuvis AI pipelines."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    @grpc_handler("SetProfiling failed")
    def set_profiling(
        self,
        request: cuvis_ai_pb2.SetProfilingRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SetProfilingResponse:
        """Enable, disable, or reconfigure pipeline profiling."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.SetProfilingResponse()

        if not require_pipeline(session, context):
            return cuvis_ai_pb2.SetProfilingResponse()

        # Map optional proto fields → Python defaults for absent fields
        synchronize_cuda = (
            request.synchronize_cuda if request.HasField("synchronize_cuda") else False
        )
        reset = request.reset if request.HasField("reset") else False
        skip_first_n = request.skip_first_n if request.HasField("skip_first_n") else 0

        session.pipeline.set_profiling(
            enabled=request.enabled,
            synchronize_cuda=synchronize_cuda,
            reset=reset,
            skip_first_n=skip_first_n,
        )

        return cuvis_ai_pb2.SetProfilingResponse(
            profiling_enabled=request.enabled,
        )

    @grpc_handler("GetProfilingSummary failed")
    def get_profiling_summary(
        self,
        request: cuvis_ai_pb2.GetProfilingSummaryRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetProfilingSummaryResponse:
        """Retrieve accumulated per-node profiling statistics."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.GetProfilingSummaryResponse()

        if not require_pipeline(session, context):
            return cuvis_ai_pb2.GetProfilingSummaryResponse()

        # Map optional proto stage enum → Python ExecutionStage or None
        stage = None
        if request.HasField("stage"):
            proto_stage = request.stage
            stage_str = _STAGE_PROTO_TO_STR.get(proto_stage)
            if stage_str is not None:
                stage = ExecutionStage(stage_str)

        stats = session.pipeline.get_profiling_summary(stage=stage)

        # Map Python NodeProfilingStats → proto messages
        proto_stats = []
        for s in stats:
            proto_stage_val = _STAGE_STR_TO_PROTO.get(
                s.stage, cuvis_ai_pb2.EXECUTION_STAGE_UNSPECIFIED
            )
            proto_stats.append(
                cuvis_ai_pb2.NodeProfilingStats(
                    node_name=s.node_name,
                    stage=proto_stage_val,
                    count=s.count,
                    mean_ms=s.mean_ms,
                    median_ms=s.median_ms,
                    std_ms=s.std_ms,
                    min_ms=s.min_ms,
                    max_ms=s.max_ms,
                    total_ms=s.total_ms,
                    last_ms=s.last_ms,
                )
            )

        return cuvis_ai_pb2.GetProfilingSummaryResponse(node_stats=proto_stats)


__all__ = ["ProfilingService"]
