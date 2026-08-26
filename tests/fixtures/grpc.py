"""gRPC testing fixtures."""

import json
import logging
from collections.abc import Generator
from concurrent import futures
from pathlib import Path
from typing import Any

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc import CuvisAIService, cuvis_ai_pb2, cuvis_ai_pb2_grpc
from cuvis_ai_core.grpc.session_manager import SessionManager

# Keep a handle on the service so tests can introspect the live SessionManager
SERVICE_INSTANCE: CuvisAIService | None = None

# Bundled per-plugin manifests this repo's own pipelines reference (e.g.
# cuvis_ai_test_nodes for the mock-node configs under configs/pipeline).
_PLUGINS_DIR = Path(__file__).resolve().parents[2] / "configs" / "plugins"


def register_pipeline_plugins(
    grpc_stub: Any, session_id: str, pipeline_config_bytes: bytes
) -> None:
    """Register the plugins a pipeline declares, as a real client would.

    ``LoadPipeline`` resolves a pipeline's ``plugins:`` against the session's
    client-pushed catalog (``LoadPlugin``), not a server-side directory scan,
    so every declared plugin must be registered first. Each bundled manifest
    in ``configs/plugins`` is sent with its local ``path`` resolved to absolute
    (the server rejects a client-relative path over gRPC).
    """
    config = json.loads(pipeline_config_bytes)
    for name in config.get("plugins") or []:
        manifest_path = _PLUGINS_DIR / f"{name}.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and "repo" not in manifest:
            raw_path = Path(manifest.get("path", "."))
            if not raw_path.is_absolute():
                manifest["path"] = str((manifest_path.parent / raw_path).resolve())
        grpc_stub.LoadPlugin(
            cuvis_ai_pb2.LoadPluginRequest(
                session_id=session_id,
                manifest=cuvis_ai_pb2.PluginManifest(
                    config_bytes=json.dumps(manifest).encode("utf-8")
                ),
            )
        )


def restore_trainrun_into_prepared_session(
    grpc_stub: Any, trainrun_path: str, **request_fields: Any
):
    """Create a session, register the trainrun pipeline's plugins, restore into it.

    A server-created session has an empty plugin catalog, so restoring a
    trainrun whose pipeline declares ``plugins:`` requires the client-owned
    flow: CreateSession -> LoadPlugin each manifest -> RestoreTrainRun with
    ``session_id``. The pipeline reference is resolved against the trainrun
    file's directory when relative. Extra ``request_fields`` (``weights_path``,
    ``strict``) are forwarded to the request. Returns the restore response.
    """
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    trainrun = yaml.safe_load(Path(trainrun_path).read_text(encoding="utf-8"))
    reference = trainrun.get("pipeline") if isinstance(trainrun, dict) else None
    if isinstance(reference, str) and reference:
        pipeline_path = Path(reference)
        if not pipeline_path.is_absolute():
            pipeline_path = (Path(trainrun_path).parent / pipeline_path).resolve()
        if pipeline_path.is_file():
            register_pipeline_plugins(
                grpc_stub, session_id, pipeline_bytes_from_path(str(pipeline_path))
            )
    return grpc_stub.RestoreTrainRun(
        cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=trainrun_path, session_id=session_id, **request_fields
        )
    )


# Session-scoped gRPC server to avoid repeated startup/shutdown overhead
@pytest.fixture(scope="session")
def grpc_server() -> Generator[str, None, None]:
    """Session-scoped gRPC server fixture.

    Creates a single gRPC server that is shared across all tests in the session,
    significantly reducing startup/shutdown overhead.

    Yields:
        str: The server address (e.g., "localhost:port")
    """
    # Configure logging for gRPC server
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    # Create server with increased thread pool for better parallelism
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    global SERVICE_INSTANCE
    SERVICE_INSTANCE = CuvisAIService()
    service = SERVICE_INSTANCE
    cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(service, server)

    # Bind to available port
    port = server.add_insecure_port("localhost:0")
    if port == 0:
        raise RuntimeError("Failed to bind gRPC server to port")

    logger.info(f"Starting session-scoped gRPC server on port {port}")
    server.start()

    try:
        yield f"localhost:{port}"
    finally:
        logger.info("Stopping session-scoped gRPC server")
        server.stop(grace=None)
        SERVICE_INSTANCE = None


@pytest.fixture
def grpc_stub(grpc_server: str) -> Generator:
    """Create gRPC client stub using shared session server.

    Uses the session-scoped gRPC server to avoid creating new servers for each test.

    Args:
        grpc_server: Session-scoped server address from grpc_server fixture

    Yields:
        CuvisAIServiceStub: gRPC client stub
    """
    channel = grpc.insecure_channel(grpc_server)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        yield stub
    finally:
        channel.close()


@pytest.fixture(scope="session")
def grpc_session_manager(grpc_server: str) -> SessionManager:
    """Expose the live SessionManager used by the in-process gRPC server."""
    del grpc_server  # server is already started by the fixture dependency
    if SERVICE_INSTANCE is None:
        raise RuntimeError("CuvisAIService instance is not initialized")
    return SERVICE_INSTANCE.session_manager


# ------------------------------------------------------------------
# Shared gRPC Test Helper Functions
# ------------------------------------------------------------------


def resolve_and_load_pipeline(
    grpc_stub, session_id: str, path: str = "pipeline/gradient_based"
) -> cuvis_ai_pb2.LoadPipelineResponse:
    """Resolve and load pipeline structure via bytes-based API.

    This is a shared test helper to avoid duplicate implementations across test files.

    Args:
        grpc_stub: gRPC stub for making API calls
        session_id: Session ID to load pipeline into
        path: Pipeline path (e.g., "pipeline/gradient_based")

    Returns:
        LoadPipelineResponse from the gRPC call

    Raises:
        AssertionError: If the pipeline loading fails
    """
    config_response = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path=path,
        )
    )
    # Register the pipeline's declared plugins before loading it: the
    # orchestrator resolves plugins from the session's pushed catalog.
    register_pipeline_plugins(grpc_stub, session_id, config_response.config_bytes)
    response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=config_response.config_bytes
            ),
        )
    )
    assert response.success
    return response


def pipeline_bytes_from_path(pipeline_path: str | Path) -> bytes:
    """Convert a pipeline YAML file into JSON bytes for the LoadPipeline RPC.

    Shared helper — eliminates the identical copies in test_pipeline_management.py,
    test_experiment_management.py, and this module's load_pipeline_from_file.

    Args:
        pipeline_path: Path to pipeline YAML file

    Returns:
        JSON-encoded bytes ready for PipelineConfig.config_bytes
    """
    pipeline_dict = yaml.safe_load(Path(pipeline_path).read_text())
    return json.dumps(pipeline_dict).encode("utf-8")


def manifest_config_bytes(manifest: dict) -> bytes:
    """Encode a bare plugin manifest dict as the LoadPlugin wire payload.

    Shared helper — eliminates the identical copies in test_plugin_management.py
    and test_plugin_service.py.

    Args:
        manifest: Bare plugin manifest dict (name + source + capabilities)

    Returns:
        JSON-encoded bytes ready for manifest.config_bytes
    """
    return json.dumps(manifest).encode()


def load_pipeline_from_file(
    grpc_stub, session_id: str, pipeline_file: str | Path
) -> cuvis_ai_pb2.LoadPipelineResponse:
    """Load a pipeline from a saved YAML by converting to JSON bytes.

    This is a shared test helper to avoid duplicate implementations across test files.

    Args:
        grpc_stub: gRPC stub for making API calls
        session_id: Session ID to load pipeline into
        pipeline_file: Path to pipeline YAML file

    Returns:
        LoadPipelineResponse from the gRPC call

    Raises:
        AssertionError: If the pipeline loading fails
    """
    config_bytes = pipeline_bytes_from_path(pipeline_file)
    # Saved pipelines re-emit their `plugins:` list; register those plugins in
    # the target session first, exactly like resolve_and_load_pipeline.
    register_pipeline_plugins(grpc_stub, session_id, config_bytes)
    response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_bytes),
        )
    )
    assert response.success
    return response
