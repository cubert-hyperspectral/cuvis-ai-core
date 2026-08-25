"""Profiling over the orchestrator seam.

Profiling state lives on the live pipeline inside the child runtime; the
parent forwards SetProfiling / GetProfilingSummary through the orchestrator
bridge. This drives the full parent -> bridge -> child -> pipeline path that
the old parent-local implementation could never serve (the parent holds no
pipeline), which is exactly how the regression shipped unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from tests.fixtures.sessions import _register_pipeline_plugins

DEFAULT_CHANNELS = 61
_WEIGHTS = Path("configs/pipeline/gradient_based.pt")


def test_profiling_end_to_end_through_child(grpc_stub, create_test_cube):
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id

    # Build the pipeline with its shipped weights: no data module needed, so
    # this regression gate runs everywhere (CI has no local test data).
    config_response = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path="pipeline/gradient_based",
        )
    )
    _register_pipeline_plugins(grpc_stub, session_id, config_response.config_bytes)
    load_response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=config_response.config_bytes
            ),
        )
    )
    assert load_response.success
    weights_response = grpc_stub.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id,
            weights_path=str(_WEIGHTS.resolve()),
            strict=True,
        )
    )
    assert weights_response.success

    set_resp = grpc_stub.SetProfiling(
        cuvis_ai_pb2.SetProfilingRequest(session_id=session_id, enabled=True)
    )
    assert set_resp.profiling_enabled is True

    cube, wavelengths = create_test_cube(
        batch_size=1, height=2, width=2, num_channels=DEFAULT_CHANNELS, mode="random"
    )
    wavelengths_2d = np.tile(wavelengths, (cube.shape[0], 1)).astype(np.int32)
    grpc_stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube.numpy()),
                wavelengths=helpers.numpy_to_proto(wavelengths_2d),
            ),
        )
    )

    summary = grpc_stub.GetProfilingSummary(
        cuvis_ai_pb2.GetProfilingSummaryRequest(session_id=session_id)
    )
    assert len(summary.node_stats) > 0
    for stat in summary.node_stats:
        assert stat.count >= 1
        assert stat.stage == cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE
        assert stat.mean_ms >= 0.0

    # A stage that never ran after enabling yields no stats.
    train_only = grpc_stub.GetProfilingSummary(
        cuvis_ai_pb2.GetProfilingSummaryRequest(
            session_id=session_id, stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN
        )
    )
    assert len(train_only.node_stats) == 0

    # reset=True wipes the accumulated stats.
    grpc_stub.SetProfiling(
        cuvis_ai_pb2.SetProfilingRequest(
            session_id=session_id, enabled=True, reset=True
        )
    )
    wiped = grpc_stub.GetProfilingSummary(
        cuvis_ai_pb2.GetProfilingSummaryRequest(session_id=session_id)
    )
    assert len(wiped.node_stats) == 0

    off = grpc_stub.SetProfiling(
        cuvis_ai_pb2.SetProfilingRequest(session_id=session_id, enabled=False)
    )
    assert off.profiling_enabled is False

    close_response = grpc_stub.CloseSession(
        cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
    )
    assert close_response.success
