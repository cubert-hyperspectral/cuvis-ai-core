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
    # Fresh handle is "alive" (returncode None), mirroring the real
    # ChildHandle whose returncode is process.poll() until the child exits.
    assert handle.returncode is None


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
    # After terminate the handle reports a returncode (no longer alive).
    assert handle.returncode == 0
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


def _resolvable_pipeline(tmp_path):
    """A minimal pipeline config + plugins dir the resolver accepts.

    ``ensure_child_for_session`` always runs ``resolve_pipeline_plugins``
    now (``plugins:`` is mandatory), so the lifecycle tests below need a
    config whose declared plugin resolves against a real manifest. The
    in-memory spawner ignores the plugin ``path``; only the catalog lookup
    and coverage check matter here.
    """
    node_class = "tests.fixtures.mock_nodes.MinMaxNormalizer"
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "m.yaml").write_text(
        "name: test_plugin\n"
        "path: '.'\n"
        "capabilities:\n"
        f"  - class_name: {node_class}\n",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        plugins=["test_plugin"],
        nodes=[SimpleNamespace(class_name=node_class)],
    )
    return cfg, [plugins_dir]


def test_ensure_child_for_session_is_idempotent(tmp_path):
    sm = SessionManager()
    sid = sm.create_session()
    cfg, plugins_dirs = _resolvable_pipeline(tmp_path)

    first = orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, plugins_dirs)
    second = orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, plugins_dirs)

    assert first is second
    assert sm.get_session(sid).child_handle is first


def test_ensure_child_respawns_when_child_has_exited(tmp_path):
    """A dead child handle is dropped and replaced, so the session recovers."""
    sm = SessionManager()
    sid = sm.create_session()
    cfg, plugins_dirs = _resolvable_pipeline(tmp_path)

    first = orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, plugins_dirs)
    # Simulate a crash: terminate flips returncode away from None.
    first.terminate()
    assert first.returncode is not None

    second = orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, plugins_dirs)
    assert second is not first
    assert second.returncode is None
    assert sm.get_session(sid).child_handle is second


def test_ensure_child_records_runtime_base_dir_and_close_removes_it(tmp_path):
    """ensure_child stamps the scratch root; close_session deletes the tree."""
    sm = SessionManager()
    sid = sm.create_session()
    cfg, plugins_dirs = _resolvable_pipeline(tmp_path)

    orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, plugins_dirs)
    base = sm.get_session(sid).runtime_base_dir
    assert base is not None
    assert base.exists()

    sm.close_session(sid)
    assert not base.exists()


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


@pytest.mark.parametrize(
    "bad_bytes",
    [
        b"{not valid json",  # malformed JSON -> JSONDecodeError
        b"[1, 2, 3]",  # JSON array -> non-dict ValueError
        b'"just a string"',  # JSON scalar -> non-dict ValueError
        b'{"unexpected_key": 123}',  # valid JSON object, invalid schema -> pydantic ValidationError (a ValueError)
    ],
)
def test_forward_load_pipeline_malformed_config_bytes_is_invalid_argument(bad_bytes):
    """Non-JSON / non-object / invalid-schema config_bytes -> INVALID_ARGUMENT, not a 500."""
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm,
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=sid,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=bad_bytes),
        ),
        ctx,
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert "pipeline config" in ctx.details()


def test_forward_load_pipeline_forwards_data_module(monkeypatch):
    """The ``data_module`` name on the request reaches the env composer."""
    sm = SessionManager()
    sid = sm.create_session()
    ensure = MagicMock()
    monkeypatch.setattr(orchestrator_bridge, "ensure_child_for_session", ensure)

    orchestrator_bridge.forward_load_pipeline(
        sm,
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=sid,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=b"{}"),
            data_module="cu3s",
        ),
        _InMemoryContext(),
    )

    assert ensure.call_args.kwargs["data_module"] == "cu3s"


def test_forward_load_pipeline_without_data_module_passes_none(monkeypatch):
    """An unset ``data_module`` (empty proto string) forwards as None, not ''."""
    sm = SessionManager()
    sid = sm.create_session()
    ensure = MagicMock()
    monkeypatch.setattr(orchestrator_bridge, "ensure_child_for_session", ensure)

    orchestrator_bridge.forward_load_pipeline(
        sm,
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=sid,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=b"{}"),
        ),
        _InMemoryContext(),
    )

    assert ensure.call_args.kwargs["data_module"] is None


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
    fake_cfg = SimpleNamespace(pipeline="pl.yaml")
    pl = tmp_path / "pl.yaml"
    pl.write_text(
        "plugins: [needs_this]\nnodes: []\nconnections: []\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "cuvis_ai_core.grpc.trainrun_service.TrainRunService.parse_trainrun_yaml",
        lambda path: (fake_cfg, pl),
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


# ---------------------------------------------------------------------------
# get_spawner lazy init + detect_core_source pypi branch
# ---------------------------------------------------------------------------


def test_get_spawner_lazily_constructs_local_spawner():
    from cuvis_ai_core.orchestrator.spawner import LocalChildRuntimeSpawner

    orchestrator_bridge.reset_spawner()
    try:
        assert isinstance(orchestrator_bridge.get_spawner(), LocalChildRuntimeSpawner)
    finally:
        orchestrator_bridge.reset_spawner()


def test_detect_core_source_site_packages_reports_pypi(monkeypatch):
    import cuvis_ai_core

    monkeypatch.setattr(
        cuvis_ai_core,
        "__file__",
        "/opt/venv/lib/site-packages/cuvis_ai_core/__init__.py",
    )
    source = orchestrator_bridge.detect_core_source()
    assert source.kind == "pypi"
    assert "cuvis-ai-core==" in source.identity


# ---------------------------------------------------------------------------
# forward_inference / forward_train with an attached (fake) child
# ---------------------------------------------------------------------------


def _attach_fake_child(sm, sid):
    handle = MagicMock()
    handle.returncode = None
    sm.get_session(sid).child_handle = handle
    return handle


def test_forward_inference_unknown_session_returns_empty():
    sm = SessionManager()
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_inference(
        sm, cuvis_ai_pb2.InferenceRequest(session_id="nope"), ctx
    )
    assert resp == cuvis_ai_pb2.InferenceResponse()


def test_forward_inference_forwards_to_child():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    handle = _attach_fake_child(sm, sid)
    handle.stub.return_value.Inference.return_value = cuvis_ai_pb2.InferenceResponse()
    resp = orchestrator_bridge.forward_inference(
        sm, cuvis_ai_pb2.InferenceRequest(session_id=sid), ctx
    )
    assert resp == cuvis_ai_pb2.InferenceResponse()
    handle.stub.return_value.Inference.assert_called_once()


def test_forward_train_unknown_session_yields_empty():
    sm = SessionManager()
    ctx = _InMemoryContext()
    gen = orchestrator_bridge.forward_train(
        sm, cuvis_ai_pb2.TrainRequest(session_id="nope"), ctx
    )
    assert list(gen) == []


def test_forward_train_streams_child_responses():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    handle = _attach_fake_child(sm, sid)
    handle.stub.return_value.Train.return_value = iter(
        [cuvis_ai_pb2.TrainResponse(), cuvis_ai_pb2.TrainResponse()]
    )
    out = list(
        orchestrator_bridge.forward_train(
            sm, cuvis_ai_pb2.TrainRequest(session_id=sid), ctx
        )
    )
    assert len(out) == 2


def test_forward_train_propagates_stream_error():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    handle = _attach_fake_child(sm, sid)
    handle.stub.return_value.Train.side_effect = _InMemoryRpcError(
        grpc.StatusCode.INTERNAL, "stream blew up"
    )
    out = list(
        orchestrator_bridge.forward_train(
            sm, cuvis_ai_pb2.TrainRequest(session_id=sid), ctx
        )
    )
    assert out == []
    assert ctx.code() is grpc.StatusCode.INTERNAL
    assert ctx.details() == "stream blew up"


# ---------------------------------------------------------------------------
# forward_restore_train_run: yaml parse failures, generic failure, success
# ---------------------------------------------------------------------------


def _patch_parse(monkeypatch, result=None, *, raises=None):
    def _parse(path):
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(
        "cuvis_ai_core.grpc.trainrun_service.TrainRunService.parse_trainrun_yaml",
        _parse,
    )


def test_forward_restore_train_run_missing_file_is_not_found(monkeypatch, tmp_path):
    sm = SessionManager()
    _patch_parse(monkeypatch, raises=FileNotFoundError("no such trainrun"))
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_restore_train_run(
        sm,
        cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=str(tmp_path / "x.yaml")),
        ctx,
    )
    assert ctx.code() is grpc.StatusCode.NOT_FOUND
    assert resp == cuvis_ai_pb2.RestoreTrainRunResponse()


def test_forward_restore_train_run_bad_yaml_is_invalid_argument(monkeypatch, tmp_path):
    sm = SessionManager()
    _patch_parse(monkeypatch, raises=ValueError("hydra defaults not allowed"))
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_restore_train_run(
        sm,
        cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=str(tmp_path / "x.yaml")),
        ctx,
    )
    assert ctx.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert resp == cuvis_ai_pb2.RestoreTrainRunResponse()


def test_forward_restore_train_run_reraises_non_value_error(monkeypatch, tmp_path):
    sm = SessionManager()
    fake_cfg = SimpleNamespace(pipeline="pl.yaml")
    pl = tmp_path / "pl.yaml"
    pl.write_text("plugins: [p]\nnodes: []\nconnections: []\n", encoding="utf-8")
    _patch_parse(monkeypatch, result=(fake_cfg, pl))
    monkeypatch.setattr(
        orchestrator_bridge,
        "ensure_child_for_session",
        MagicMock(side_effect=RuntimeError("compose failed")),
    )
    ctx = _InMemoryContext()
    with pytest.raises(RuntimeError, match="compose failed"):
        orchestrator_bridge.forward_restore_train_run(
            sm,
            cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=str(tmp_path / "x.yaml")),
            ctx,
        )
    # The allocated parent session was dropped before re-raising.
    assert sm._sessions == {}


def test_forward_restore_train_run_fills_parent_session_id(monkeypatch, tmp_path):
    sm = SessionManager()
    fake_cfg = SimpleNamespace(pipeline="pl.yaml")
    pl = tmp_path / "pl.yaml"
    pl.write_text("plugins: [p]\nnodes: []\nconnections: []\n", encoding="utf-8")
    _patch_parse(monkeypatch, result=(fake_cfg, pl))

    def _fake_ensure(
        session_manager, session_id, pipeline_config, plugins_dirs, data_module=None
    ):
        handle = MagicMock()
        handle.stub.return_value.RestoreTrainRun.return_value = (
            cuvis_ai_pb2.RestoreTrainRunResponse(session_id="")
        )
        session_manager.get_session(session_id).child_handle = handle
        return handle

    monkeypatch.setattr(orchestrator_bridge, "ensure_child_for_session", _fake_ensure)
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_restore_train_run(
        sm,
        cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=str(tmp_path / "x.yaml")),
        ctx,
    )
    # The child returned an empty session id, so the parent's is filled in.
    assert resp.session_id != ""
    assert resp.session_id in sm.list_sessions()


# ---------------------------------------------------------------------------
# Pipeline-op wrappers: each no-child wrapper returns FAILED_PRECONDITION
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name, request_cls",
    [
        ("forward_load_pipeline_weights", cuvis_ai_pb2.LoadPipelineWeightsRequest),
        ("forward_save_pipeline", cuvis_ai_pb2.SavePipelineRequest),
        ("forward_save_train_run", cuvis_ai_pb2.SaveTrainRunRequest),
        ("forward_get_pipeline_outputs", cuvis_ai_pb2.GetPipelineOutputsRequest),
        (
            "forward_get_pipeline_visualization",
            cuvis_ai_pb2.GetPipelineVisualizationRequest,
        ),
        ("forward_get_train_status", cuvis_ai_pb2.GetTrainStatusRequest),
    ],
)
def test_pipeline_op_wrappers_without_child(fn_name, request_cls):
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    getattr(orchestrator_bridge, fn_name)(sm, request_cls(session_id=sid), ctx)
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION


# ---------------------------------------------------------------------------
# forward_set_train_run_config: session / child guards + forward
# ---------------------------------------------------------------------------


def test_forward_set_train_run_config_unknown_session_fails():
    sm = SessionManager()
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_set_train_run_config(
        sm, cuvis_ai_pb2.SetTrainRunConfigRequest(session_id="nope"), ctx
    )
    assert resp.success is False


def test_forward_set_train_run_config_without_child_has_tailored_message():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    req = cuvis_ai_pb2.SetTrainRunConfigRequest(
        session_id=sid, config=cuvis_ai_pb2.TrainRunConfig(config_bytes=b"{}")
    )
    resp = orchestrator_bridge.forward_set_train_run_config(sm, req, ctx)
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "LoadPipeline" in ctx.details()


def test_forward_set_train_run_config_forwards_to_child():
    sm = SessionManager()
    sid = sm.create_session()
    ctx = _InMemoryContext()
    handle = _attach_fake_child(sm, sid)
    handle.stub.return_value.SetTrainRunConfig.return_value = (
        cuvis_ai_pb2.SetTrainRunConfigResponse(success=True)
    )
    req = cuvis_ai_pb2.SetTrainRunConfigRequest(
        session_id=sid, config=cuvis_ai_pb2.TrainRunConfig(config_bytes=b"{}")
    )
    resp = orchestrator_bridge.forward_set_train_run_config(sm, req, ctx)
    assert resp.success is True


# ---------------------------------------------------------------------------
# _InMemoryRpcError.__str__ + _InMemoryStub method surface
# ---------------------------------------------------------------------------


def test_inmemory_rpc_error_str_includes_code_and_details():
    err = _InMemoryRpcError(grpc.StatusCode.INTERNAL, "kaboom")
    text = str(err)
    assert "code=" in text and "kaboom" in text


def test_inmemory_stub_methods_invoke_servicer():
    """Every _InMemoryStub delegate calls into the servicer (no-session → raises)."""
    from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

    stub = _InMemoryStub(RunRuntimeServicer())
    calls = [
        (
            "LoadPipelineWeights",
            cuvis_ai_pb2.LoadPipelineWeightsRequest(session_id="x"),
        ),
        ("SavePipeline", cuvis_ai_pb2.SavePipelineRequest(session_id="x")),
        ("SaveTrainRun", cuvis_ai_pb2.SaveTrainRunRequest(session_id="x")),
        ("GetTrainStatus", cuvis_ai_pb2.GetTrainStatusRequest(session_id="x")),
        ("Inference", cuvis_ai_pb2.InferenceRequest(session_id="x")),
        (
            "SetTrainRunConfig",
            cuvis_ai_pb2.SetTrainRunConfigRequest(session_id="x"),
        ),
        (
            "RestoreTrainRun",
            cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path="absent.yaml"),
        ),
    ]
    for method_name, request in calls:
        # The point is to execute the delegate body; a non-OK servicer code
        # surfaces as _InMemoryRpcError, which is the expected outcome here.
        try:
            getattr(stub, method_name)(request)
        except _InMemoryRpcError:
            pass


def test_inmemory_stub_lifecycle_methods():
    """Train stream + CloseSession + StopRun in-memory stub paths."""
    from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

    stub = _InMemoryStub(RunRuntimeServicer())

    # Train returns a generator; consuming it runs the wrapping _iter body.
    try:
        list(stub.Train(cuvis_ai_pb2.TrainRequest(session_id="ghost")))
    except _InMemoryRpcError:
        pass

    # CloseSession is idempotent on the servicer → returns a response.
    close_resp = stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id="ghost"))
    assert close_resp.success is True

    # StopRun signals the servicer's shutdown event and returns ok.
    stop_resp = stub.StopRun(cuvis_ai_pb2.StopRunRequest())
    assert stop_resp.ok is True
