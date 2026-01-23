"""Shared fixtures for workflow testing to avoid redundant operations."""

from collections.abc import Generator
from pathlib import Path

import pytest

from cuvis_ai_core.grpc.v1 import cuvis_ai_core_pb2 as cuvis_ai_pb2
from tests.fixtures.grpc import resolve_and_load_pipeline

# Cache for pretrained pipelines to avoid recomputation
_pretrained_pipeline_cache = {}


@pytest.fixture(scope="function")
def pretrained_pipeline(
    grpc_stub, tmp_path, data_config_factory
) -> Generator[Path, None, None]:
    """Function-scoped fixture that creates and caches a pretrained pipeline.

    This fixture creates a trainable pipeline, performs statistical training,
    and saves it to a temporary directory. Uses the gradient_based pipeline
    from configs which contains trainable nodes.

    Yields:
        Path: Path to the saved pretrained pipeline YAML file
    """
    cache_key = "pretrained_trainable"

    # Check if we already have a cached pipeline
    if cache_key in _pretrained_pipeline_cache:
        yield _pretrained_pipeline_cache[cache_key]
        return

    # Create temporary directory for this session
    temp_dir = tmp_path / "pretrained_pipelines"
    temp_dir.mkdir(exist_ok=True)
    pipeline_path = temp_dir / "pretrained_trainable.yaml"

    # Create session and setup pipeline
    session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
    session_id = session_response.session_id

    try:
        # Load the trainable pipeline using proper proto construction
        resolve_and_load_pipeline(grpc_stub, session_id, path="pipeline/gradient_based")

        # Perform statistical training to initialize nodes
        stat_train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config_factory(),
        )

        # Consume training progress messages
        for _ in grpc_stub.Train(stat_train_request):
            pass

        # Save the pretrained pipeline
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

        # Cache the pipeline path for reuse
        _pretrained_pipeline_cache[cache_key] = pipeline_path

        # Yield the pipeline path for test usage
        yield pipeline_path
    finally:
        # Clean up the session
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
