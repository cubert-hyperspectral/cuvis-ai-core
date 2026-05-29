"""Orchestrator-bridge tests.

The bridge is the only dispatch path: every LoadPipeline / Inference
/ Train / RestoreTrainRun call routes through it. The tests below
exercise the injection seams (composer + spawner) and the
in-memory mode used by the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

from cuvis_ai_core.grpc import orchestrator_bridge
from cuvis_ai_core.grpc.orchestrator_bridge import (
    _InMemoryChildHandle,
    _InMemoryRpcError,
    _InMemorySpawner,
    _InMemoryStub,
    _noop_composer,
    install_in_memory_orchestrator,
    reset_orchestrator,
)


# ---------------------------------------------------------------------------
# Composer + spawner injection seams
# ---------------------------------------------------------------------------


def test_set_composer_overrides_default():
    fake = MagicMock()
    try:
        orchestrator_bridge.set_composer(fake)
        assert orchestrator_bridge.get_composer() is fake
    finally:
        orchestrator_bridge.reset_composer()


def test_reset_composer_restores_default():
    from cuvis_ai_core.orchestrator.composer import compose_env as _real_compose_env

    orchestrator_bridge.set_composer(MagicMock())
    orchestrator_bridge.reset_composer()
    assert orchestrator_bridge.get_composer() is _real_compose_env


def test_set_spawner_overrides_default():
    fake = MagicMock(spec=_InMemorySpawner)
    try:
        orchestrator_bridge.set_spawner(fake)
        assert orchestrator_bridge.get_spawner() is fake
    finally:
        orchestrator_bridge.reset_spawner()


# ---------------------------------------------------------------------------
# detect_core_source: editable cuvis-ai-core (the dev case)
# ---------------------------------------------------------------------------


def test_detect_core_source_for_editable_install_returns_local():
    source = orchestrator_bridge.detect_core_source()
    assert source.kind in ("local", "pypi")
    if source.kind == "local":
        assert Path(source.identity).exists()


# ---------------------------------------------------------------------------
# In-memory mode: install_in_memory_orchestrator()
# ---------------------------------------------------------------------------


def test_install_in_memory_orchestrator_swaps_composer_and_spawner():
    install_in_memory_orchestrator()
    try:
        assert orchestrator_bridge.get_composer() is _noop_composer
        assert isinstance(orchestrator_bridge.get_spawner(), _InMemorySpawner)
    finally:
        reset_orchestrator()


def test_inmemory_spawner_returns_inmemory_child_handle(tmp_path):
    from cuvis_ai_core.orchestrator.spawner import DeclaredPaths

    declared = DeclaredPaths(output_dir=tmp_path / "o", scratch_dir=tmp_path / "s")
    (tmp_path / "o").mkdir()
    (tmp_path / "s").mkdir()
    spawner = _InMemorySpawner()
    handle = spawner.spawn(
        venv_path=tmp_path / "venv",
        cwd=tmp_path,
        declared_paths=declared,
    )
    assert isinstance(handle, _InMemoryChildHandle)
    assert handle.endpoint == "in-memory"
    assert handle.returncode == 0


def test_inmemory_handle_terminate_and_kill_idempotent(tmp_path):
    from cuvis_ai_core.orchestrator.spawner import DeclaredPaths

    declared = DeclaredPaths(output_dir=tmp_path / "o", scratch_dir=tmp_path / "s")
    (tmp_path / "o").mkdir()
    (tmp_path / "s").mkdir()
    spawner = _InMemorySpawner()
    handle = spawner.spawn(
        venv_path=tmp_path / "venv",
        cwd=tmp_path,
        declared_paths=declared,
    )
    assert handle.terminate(grace_s=1.0) == 0
    assert handle.kill() == 0


# ---------------------------------------------------------------------------
# _InMemoryStub: error propagation
# ---------------------------------------------------------------------------


def test_inmemory_stub_propagates_grpc_error_codes():
    """When the servicer sets a non-OK code, the stub raises an RpcError."""
    from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

    servicer = RunRuntimeServicer()
    stub = _InMemoryStub(servicer)
    # InitializeSession with empty session_id sets INVALID_ARGUMENT.
    with pytest.raises(_InMemoryRpcError) as excinfo:
        stub.InitializeSession(cuvis_ai_pb2.InitializeSessionRequest(session_id=""))
    import grpc

    assert excinfo.value.code() is grpc.StatusCode.INVALID_ARGUMENT


def test_inmemory_stub_returns_response_on_success():
    from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

    servicer = RunRuntimeServicer()
    stub = _InMemoryStub(servicer)
    response = stub.HealthCheck(cuvis_ai_pb2.HealthCheckRequest())
    assert (
        response.status
        == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
    )
