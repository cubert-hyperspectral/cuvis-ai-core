"""RunRuntimeServicer unit tests — in-process, no subprocess.

Drives the servicer directly with a fake context so the assertions
run in milliseconds. The end-to-end smoke (spawned child process)
lives in ``tests/orchestrator/test_spawner.py``.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import grpc
import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

from cuvis_ai_core.run_runtime.service import (
    RunRuntimeServicer,
    _decode_resolved_plugins,
)
from cuvis_ai_schemas.plugin import GitPluginSource, LocalPluginSource


class FakeContext:
    """Minimal grpc.ServicerContext stand-in for unary handler tests."""

    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details


# ---------------------------------------------------------------------------
# _decode_resolved_plugins
# ---------------------------------------------------------------------------


def test_decode_resolved_plugins_empty_bytes_returns_empty_dict():
    assert _decode_resolved_plugins(b"") == {}


def test_decode_resolved_plugins_discriminates_git_vs_local(tmp_path):
    # The wire format is a JSON LIST of bare single-plugin manifests; each
    # manifest carries its own ``name`` and the decoder re-keys by it.
    payload = json.dumps(
        [
            {
                "name": "from_git",
                "repo": "https://example.com/repo.git",
                "tag": "v1.2.3",
                "capabilities": [{"class_name": "pkg.mod.Cls"}],
            },
            {
                "name": "from_local",
                "path": str(tmp_path),
                "capabilities": [{"class_name": "pkg2.mod.Cls"}],
            },
        ]
    ).encode("utf-8")
    out = _decode_resolved_plugins(payload)
    assert isinstance(out["from_git"], GitPluginSource)
    assert isinstance(out["from_local"], LocalPluginSource)


def test_decode_resolved_plugins_missing_source_raises():
    # A bare manifest with neither 'repo' nor 'path' fails union validation.
    payload = json.dumps(
        [{"name": "bad", "capabilities": [{"class_name": "x.mod.Y"}]}]
    ).encode("utf-8")
    with pytest.raises(ValueError):
        _decode_resolved_plugins(payload)


def test_decode_resolved_plugins_top_level_not_list_raises():
    # The payload must decode to a LIST; an empty list is valid (-> {}),
    # but a JSON object at the top level is rejected.
    assert _decode_resolved_plugins(b"[]") == {}
    with pytest.raises(TypeError, match="must decode to a list"):
        _decode_resolved_plugins(b"{}")


def test_decode_resolved_plugins_entry_not_dict_raises():
    payload = json.dumps(["not-a-dict"]).encode("utf-8")
    with pytest.raises(TypeError, match="must be a dict"):
        _decode_resolved_plugins(payload)


# ---------------------------------------------------------------------------
# HealthCheck — trivial but exercised
# ---------------------------------------------------------------------------


def test_health_check_always_serving():
    servicer = RunRuntimeServicer()
    response = servicer.HealthCheck(cuvis_ai_pb2.HealthCheckRequest(), FakeContext())
    assert response.status == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING


# ---------------------------------------------------------------------------
# InitializeSession
# ---------------------------------------------------------------------------


def test_initialize_session_rejects_empty_session_id():
    servicer = RunRuntimeServicer()
    ctx = FakeContext()
    response = servicer.InitializeSession(
        cuvis_ai_pb2.InitializeSessionRequest(session_id=""),
        ctx,
    )
    assert response.ok is False
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


def test_initialize_session_rejects_malformed_plugin_json():
    servicer = RunRuntimeServicer()
    ctx = FakeContext()
    response = servicer.InitializeSession(
        cuvis_ai_pb2.InitializeSessionRequest(
            session_id="s1",
            resolved_plugins_json=b"not-json",
        ),
        ctx,
    )
    assert response.ok is False
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


def test_initialize_session_with_no_plugins_succeeds():
    servicer = RunRuntimeServicer()
    ctx = FakeContext()
    response = servicer.InitializeSession(
        cuvis_ai_pb2.InitializeSessionRequest(
            session_id="s1",
            search_paths=["/some/path"],
            resolved_plugins_json=b"",
        ),
        ctx,
    )
    assert response.ok is True
    state = servicer._session_manager.get_session("s1")
    assert state.search_paths == ["/some/path"]


class _FakeTestNode:
    """Importable dummy class for plugin-registration tests."""

    pass


def test_initialize_session_imports_class_and_registers_it():
    """End-to-end through the real ``import_plugin_nodes`` helper.

    Uses a class defined in this test module so the import resolves
    without any plugin packaging machinery. The point is that the
    servicer registers the imported class on the session's
    NodeRegistry just like the legacy in-process loader did.
    """
    servicer = RunRuntimeServicer()
    ctx = FakeContext()
    payload = json.dumps(
        [
            {
                "name": "fake_plugin",
                "path": "/tmp/does-not-need-to-exist",
                "capabilities": [{"class_name": f"{__name__}._FakeTestNode"}],
            }
        ]
    ).encode("utf-8")
    response = servicer.InitializeSession(
        cuvis_ai_pb2.InitializeSessionRequest(
            session_id="s2",
            resolved_plugins_json=payload,
        ),
        ctx,
    )
    assert response.ok is True
    state = servicer._session_manager.get_session("s2")
    assert "_FakeTestNode" in state.node_registry.loaded_plugin_nodes
    assert state.node_registry.loaded_plugin_nodes["_FakeTestNode"] is _FakeTestNode
    assert "fake_plugin" in state.registered_plugins


# ---------------------------------------------------------------------------
# Lifecycle: StopRun, CloseSession
# ---------------------------------------------------------------------------


def test_stop_run_sets_shutdown_event_and_closes_session():
    servicer = RunRuntimeServicer()
    servicer._session_manager.create_session_with_id("s1")
    response = servicer.StopRun(
        cuvis_ai_pb2.StopRunRequest(session_id="s1", grace_seconds=2),
        FakeContext(),
    )
    assert response.ok is True
    assert servicer.shutdown_event.is_set()
    # Session is closed (close_session pops it from the manager).
    assert "s1" not in servicer._session_manager.list_sessions()


def test_stop_run_no_session_id_still_signals_shutdown():
    servicer = RunRuntimeServicer()
    response = servicer.StopRun(cuvis_ai_pb2.StopRunRequest(), FakeContext())
    assert response.ok is True
    assert servicer.shutdown_event.is_set()


def test_close_session_is_idempotent_for_unknown_id():
    servicer = RunRuntimeServicer()
    response = servicer.CloseSession(
        cuvis_ai_pb2.CloseSessionRequest(session_id="never-existed"),
        FakeContext(),
    )
    assert response.success is True


def test_restore_train_run_requires_initialize_session_first():
    """Without an InitializeSession call, RestoreTrainRun has no session_id to attach to."""
    servicer = RunRuntimeServicer()
    ctx = FakeContext()
    servicer.RestoreTrainRun(cuvis_ai_pb2.RestoreTrainRunRequest(), ctx)
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "InitializeSession" in (ctx.details or "")


# ---------------------------------------------------------------------------
# LoadPipeline — missing session / missing config bytes
# ---------------------------------------------------------------------------


def test_load_pipeline_rejects_unknown_session():
    servicer = RunRuntimeServicer()
    ctx = FakeContext()
    request = cuvis_ai_pb2.LoadPipelineRequest(session_id="ghost")
    response = servicer.LoadPipeline(request, ctx)
    assert response.success is False
    assert ctx.code == grpc.StatusCode.NOT_FOUND


def test_load_pipeline_rejects_missing_config_bytes():
    servicer = RunRuntimeServicer()
    servicer._session_manager.create_session_with_id("s1")
    ctx = FakeContext()
    request = cuvis_ai_pb2.LoadPipelineRequest(session_id="s1")
    response = servicer.LoadPipeline(request, ctx)
    assert response.success is False
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# InitializeSession: get_session failure → INTERNAL
# ---------------------------------------------------------------------------


def test_initialize_session_internal_error_when_session_unattachable(monkeypatch):
    servicer = RunRuntimeServicer()
    ctx = FakeContext()
    monkeypatch.setattr(
        servicer._session_manager,
        "get_session",
        Mock(side_effect=ValueError("could not attach")),
    )
    response = servicer.InitializeSession(
        cuvis_ai_pb2.InitializeSessionRequest(
            session_id="s1", resolved_plugins_json=b""
        ),
        ctx,
    )
    assert response.ok is False
    assert ctx.code == grpc.StatusCode.INTERNAL


# ---------------------------------------------------------------------------
# Unary delegates forward to the underlying service objects
# ---------------------------------------------------------------------------


def test_unary_delegates_forward_to_services():
    """Every pipeline/training delegate runs against an unknown session.

    The point is that the servicer's one-line delegate body executes; the
    underlying service reports NOT_FOUND, which is the expected outcome.
    """
    servicer = RunRuntimeServicer()
    cases = [
        (
            "LoadPipelineWeights",
            cuvis_ai_pb2.LoadPipelineWeightsRequest(session_id="ghost"),
        ),
        ("SavePipeline", cuvis_ai_pb2.SavePipelineRequest(session_id="ghost")),
        (
            "SaveTrainRun",
            cuvis_ai_pb2.SaveTrainRunRequest(
                session_id="ghost", trainrun_path="x.yaml"
            ),
        ),
        (
            "GetPipelineInputs",
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id="ghost"),
        ),
        (
            "GetPipelineOutputs",
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id="ghost"),
        ),
        (
            "GetPipelineVisualization",
            cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id="ghost"),
        ),
        (
            "SetTrainRunConfig",
            cuvis_ai_pb2.SetTrainRunConfigRequest(session_id="ghost"),
        ),
        ("GetTrainStatus", cuvis_ai_pb2.GetTrainStatusRequest(session_id="ghost")),
        ("Inference", cuvis_ai_pb2.InferenceRequest(session_id="ghost")),
    ]
    for handler, request in cases:
        getattr(servicer, handler)(request, FakeContext())


def test_train_delegate_yields_from_training_service():
    servicer = RunRuntimeServicer()
    gen = servicer.Train(cuvis_ai_pb2.TrainRequest(session_id="ghost"), FakeContext())
    # Consuming the generator runs the `yield from` delegate body.
    try:
        list(gen)
    except Exception:
        pass


def test_restore_train_run_delegates_after_initialize(tmp_path):
    servicer = RunRuntimeServicer()
    servicer.InitializeSession(
        cuvis_ai_pb2.InitializeSessionRequest(
            session_id="s1", resolved_plugins_json=b""
        ),
        FakeContext(),
    )
    ctx = FakeContext()
    servicer.RestoreTrainRun(
        cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(tmp_path / "absent.yaml")
        ),
        ctx,
    )
    # Delegated to the trainrun service, whose parse step reports the missing file.
    assert ctx.code == grpc.StatusCode.NOT_FOUND


def test_stop_run_tolerates_unknown_session():
    servicer = RunRuntimeServicer()
    response = servicer.StopRun(
        cuvis_ai_pb2.StopRunRequest(session_id="ghost", grace_seconds=1), FakeContext()
    )
    assert response.ok is True
    assert servicer.shutdown_event.is_set()
