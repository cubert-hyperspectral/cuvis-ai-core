"""Shared fixtures for workflow testing to avoid redundant operations."""

from collections.abc import Generator
from pathlib import Path

import pytest

from cuvis_ai_core.grpc.v1 import cuvis_ai_core_pb2 as cuvis_ai_pb2
from tests.fixtures.grpc import resolve_and_load_pipeline


@pytest.fixture
def pretrained_pipeline(
    grpc_stub, tmp_path, data_config_factory
) -> Generator[Path, None, None]:
    """Create a pretrained pipeline for testing.

    Creates a trainable pipeline, performs statistical training, and saves it
    to a temporary directory. Uses the gradient_based pipeline from configs.

    Yields:
        Path: Path to the saved pretrained pipeline YAML file
    """
    temp_dir = tmp_path / "pretrained_pipelines"
    temp_dir.mkdir(exist_ok=True)
    pipeline_path = temp_dir / "pretrained_trainable.yaml"

    session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
    session_id = session_response.session_id

    try:
        resolve_and_load_pipeline(grpc_stub, session_id, path="pipeline/gradient_based")

        stat_train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config_factory(),
        )

        for _ in grpc_stub.Train(stat_train_request):
            pass

        save_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=str(pipeline_path),
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name="Pretrained Trainable Pipeline",
                    description="Pretrained on test data for workflow testing",
                    tags=["pretrained", "trainable"],
                ),
            )
        )

        if not save_response.success:
            raise RuntimeError("Failed to save pretrained pipeline")

        yield pipeline_path
    finally:
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


@pytest.fixture
def shared_workflow_setup(
    grpc_stub, tmp_path, data_config_factory
) -> Generator[dict, None, None]:
    """Shared setup for workflow tests that need multiple pipeline types.

    Creates and saves multiple pipeline types that can be reused across tests.
    Uses the gradient_based pipeline from configs.

    Yields:
        dict: Dictionary containing paths to different pipeline types
    """
    setup_data = {}
    temp_dir = tmp_path / "shared_workflows"
    temp_dir.mkdir(exist_ok=True)

    # Create and save a trainable pipeline
    session1 = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())

    try:
        # Load pipeline using proper proto construction
        resolve_and_load_pipeline(
            grpc_stub, session1.session_id, path="pipeline/gradient_based"
        )

        # Perform statistical training
        stat_train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session1.session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config_factory(),
        )

        for _ in grpc_stub.Train(stat_train_request):
            pass

        pipeline1_path = temp_dir / "trained_pipeline_1.yaml"
        grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session1.session_id,
                pipeline_path=str(pipeline1_path),
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name="Trained Pipeline 1",
                    description="Statistical training",
                    tags=["trained", "statistical"],
                ),
            )
        )
        setup_data["pipeline_1"] = pipeline1_path

        # Create and save a second trainable pipeline
        session2 = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())

        try:
            # Load pipeline using proper proto construction
            resolve_and_load_pipeline(
                grpc_stub, session2.session_id, path="pipeline/gradient_based"
            )

            # Perform statistical training
            stat_train_request = cuvis_ai_pb2.TrainRequest(
                session_id=session2.session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
                data=data_config_factory(),
            )

            for _ in grpc_stub.Train(stat_train_request):
                pass

            pipeline2_path = temp_dir / "trained_pipeline_2.yaml"
            grpc_stub.SavePipeline(
                cuvis_ai_pb2.SavePipelineRequest(
                    session_id=session2.session_id,
                    pipeline_path=str(pipeline2_path),
                    metadata=cuvis_ai_pb2.PipelineMetadata(
                        name="Trained Pipeline 2",
                        description="Statistical training variant",
                        tags=["trained", "statistical"],
                    ),
                )
            )
            setup_data["pipeline_2"] = pipeline2_path

        finally:
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session2.session_id)
            )

        yield setup_data

    finally:
        grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=session1.session_id)
        )
