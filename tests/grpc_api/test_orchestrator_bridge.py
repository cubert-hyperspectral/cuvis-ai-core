"""Orchestrator-bridge tests.

The bridge is the only dispatch path: every LoadPipeline / Inference
/ Train / RestoreTrainRun call routes through it. The tests below
exercise the injection seams (composer + spawner) and the
in-memory mode used by the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

from cuvis_ai_core.grpc import orchestrator_bridge
from cuvis_ai_core.grpc.orchestrator_bridge import (
    _InMemoryChildHandle,
    _InMemoryContext,
    _InMemoryRpcError,
    _InMemorySpawner,
    _InMemoryStub,
    _noop_composer,
    install_in_memory_orchestrator,
    reset_orchestrator,
)
from cuvis_ai_core.grpc.session_manager import SessionManager


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
    assert response.status == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING


# ---------------------------------------------------------------------------
# ensure_child_for_session: idempotency (runs through the autouse in-memory mode)
# ---------------------------------------------------------------------------


def test_ensure_child_for_session_is_idempotent():
    sm = SessionManager()
    sid = sm.create_session()
    cfg = SimpleNamespace(plugins=None)

    first = orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, [])
    second = orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, [])

    assert first is second
    assert sm.get_session(sid).child_handle is first


# ---------------------------------------------------------------------------
# forward_* guards: missing session / missing child / bad request
# ---------------------------------------------------------------------------


def test_forward_load_pipeline_unknown_session_returns_failure():
    sm = SessionManager()
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, cuvis_ai_pb2.LoadPipelineRequest(session_id="nope"), ctx
    )
    assert resp.success is False


def test_forward_load_pipeline_missing_config_bytes_is_invalid_argument():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, cuvis_ai_pb2.LoadPipelineRequest(session_id=sid), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert "config_bytes" in ctx.details()


def test_forward_inference_without_child_is_failed_precondition():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    orchestrator_bridge.forward_inference(
        sm, cuvis_ai_pb2.InferenceRequest(session_id=sid), ctx
    )
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "child runtime" in ctx.details().lower()


def test_forward_train_without_child_yields_empty():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    gen = orchestrator_bridge.forward_train(
        sm, cuvis_ai_pb2.TrainRequest(session_id=sid), ctx
    )
    assert list(gen) == []
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION


def test_forward_pipeline_op_without_child_is_failed_precondition():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    # Exercises the shared _forward_pipeline_op no-child branch.
    resp = orchestrator_bridge.forward_get_pipeline_inputs(
        sm, cuvis_ai_pb2.GetPipelineInputsRequest(session_id=sid), ctx
    )
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert resp == cuvis_ai_pb2.GetPipelineInputsResponse()


def test_forward_set_train_run_config_missing_config_bytes_is_invalid_argument():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_set_train_run_config(
        sm, cuvis_ai_pb2.SetTrainRunConfigRequest(session_id=sid), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# _initialize_child_session: child rejects the handshake
# ---------------------------------------------------------------------------


def test_initialize_child_session_rejection_terminates_and_raises(tmp_path):
    from cuvis_ai_core.orchestrator.spawner import DeclaredPaths

    handle = MagicMock()
    handle.stub.return_value.InitializeSession.return_value = SimpleNamespace(ok=False)
    session = SimpleNamespace(search_paths=[])
    declared = DeclaredPaths(output_dir=tmp_path, scratch_dir=tmp_path)

    with pytest.raises(RuntimeError, match="rejected InitializeSession"):
        orchestrator_bridge._initialize_child_session(
            handle, "sid-x", session, {}, declared
        )
    handle.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# _propagate_rpc_error / _call_child_with_error_propagation
# ---------------------------------------------------------------------------


def test_propagate_rpc_error_copies_code_and_details():
    ctx = _InMemoryContext()
    err = _InMemoryRpcError(grpc.StatusCode.NOT_FOUND, "missing thing")
    orchestrator_bridge._propagate_rpc_error(err, ctx)
    assert ctx.code() is grpc.StatusCode.NOT_FOUND
    assert ctx.details() == "missing thing"


def test_propagate_rpc_error_defaults_to_unknown_when_code_absent():
    ctx = _InMemoryContext()
    # A bare RpcError has no code()/details() methods → UNKNOWN + str(exc).
    orchestrator_bridge._propagate_rpc_error(grpc.RpcError(), ctx)
    assert ctx.code() is grpc.StatusCode.UNKNOWN


def test_call_child_with_error_propagation_surfaces_rpc_error():
    class _RaisingStub:
        def SomeMethod(self, request):
            raise _InMemoryRpcError(grpc.StatusCode.INTERNAL, "boom")

    ctx = _InMemoryContext()
    factory = MagicMock(return_value="empty")
    out = orchestrator_bridge._call_child_with_error_propagation(
        _RaisingStub(), "SomeMethod", object(), ctx, factory
    )
    assert out == "empty"
    assert ctx.code() is grpc.StatusCode.INTERNAL
    assert ctx.details() == "boom"


# ---------------------------------------------------------------------------
# forward_restore_train_run: cleans up the allocated session on resolver error
# ---------------------------------------------------------------------------


def test_forward_restore_train_run_cleans_up_session_on_resolver_error(
    monkeypatch, tmp_path
):
    sm = SessionManager()
    fake_cfg = SimpleNamespace(pipeline=SimpleNamespace(plugins=["needs_this"]))
    monkeypatch.setattr(
        "cuvis_ai_core.grpc.trainrun_service.TrainRunService.parse_trainrun_yaml",
        lambda path: (fake_cfg, None),
    )
    monkeypatch.setattr(
        orchestrator_bridge,
        "ensure_child_for_session",
        MagicMock(side_effect=ValueError("unresolved plugins")),
    )
    ctx = _InMemoryContext()

    resp = orchestrator_bridge.forward_restore_train_run(
        sm,
        cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=str(tmp_path / "tr.yaml")),
        ctx,
    )

    assert ctx.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert resp == cuvis_ai_pb2.RestoreTrainRunResponse()
    # The parent session allocated before the failure was dropped.
    assert sm._sessions == {}
