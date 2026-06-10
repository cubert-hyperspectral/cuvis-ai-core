import copy

import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.training.config import (
    DataConfig,
    PipelineConfig,
    TrainingConfig,
    TrainRunConfig,
)


def _build_pipeline(stub, session_id: str, pipeline_dict: dict) -> None:
    """Helper to build pipeline via RPC using LoadPipeline."""

    import json

    payload = copy.deepcopy(pipeline_dict)
    payload.pop("version", None)

    response = stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=json.dumps(payload).encode("utf-8")
            ),
        )
    )
    assert response.success


@pytest.mark.slow
def test_set_train_run_config_rejects_embedded_pipeline(
    grpc_stub, minimal_pipeline_dict, tmp_path
):
    """A trainrun config that carries a pipeline section is rejected.

    SetTrainRunConfig has a single job — attach data/training config to
    a session whose pipeline was built explicitly via LoadPipeline. An
    embedded pipeline section would re-introduce a second pipeline
    creation entry point, so it is refused with FAILED_PRECONDITION.
    """
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    _build_pipeline(grpc_stub, session_id, minimal_pipeline_dict)

    embedded = copy.deepcopy(minimal_pipeline_dict)
    embedded.pop("version", None)
    embedded["metadata"]["name"] = "anything"

    trainrun = TrainRunConfig(
        name="with-pipeline",
        pipeline=PipelineConfig.from_dict(embedded),
        data=DataConfig(cu3s_file_path="/tmp/dummy.cu3s"),
        training=TrainingConfig(),
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.SetTrainRunConfig(
            cuvis_ai_pb2.SetTrainRunConfigRequest(
                session_id=session_id,
                config=trainrun.to_proto(),
            )
        )

    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


@pytest.mark.slow
def test_set_train_run_config_requires_existing_pipeline(
    grpc_stub, minimal_pipeline_dict
):
    """SetTrainRunConfig refuses to run when no pipeline is attached.

    The caller must build the pipeline first (LoadPipeline or
    RestoreTrainRun). SetTrainRunConfig no longer builds pipelines
    implicitly from an embedded ``pipeline:`` section, so a fresh
    session can't reach a usable training state in one call.
    """
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id

    clean_pipeline = copy.deepcopy(minimal_pipeline_dict)
    clean_pipeline.pop("version", None)

    trainrun = TrainRunConfig(
        name="no-prior-pipeline",
        pipeline=PipelineConfig.from_dict(clean_pipeline),
        data=DataConfig(cu3s_file_path="/tmp/dummy.cu3s"),
        training=TrainingConfig(),
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.SetTrainRunConfig(
            cuvis_ai_pb2.SetTrainRunConfigRequest(
                session_id=session_id,
                config=trainrun.to_proto(),
            )
        )

    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
