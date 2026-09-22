"""A session's child runtime is replaced when it cannot serve the requested pipeline.

The bridge used to hand every later LoadPipeline to whatever child the session
already had, composed for the FIRST pipeline's plugin set. A pipeline of another
plugin family then failed inside that child with a module import error until the
session was closed. These tests pin the replacement contract: reuse while the
child can serve the request (same or subset plugin set, same data module),
otherwise compose the new env first, retire the old child with a confirmed exit,
spawn the replacement, and keep the old child on any failure before the retire.
Loads on one session serialise for the whole operation; a request that reaches a
retired child is answered as a replacement, not a crash.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

from cuvis_ai_core.grpc import orchestrator_bridge
from cuvis_ai_core.grpc.orchestrator_bridge import (
    _InMemoryChildHandle,
    _InMemoryContext,
    _InMemoryRpcError,
    _InMemorySpawner,
)
from cuvis_ai_core.grpc.session_manager import ChildStillRunning, SessionManager
from cuvis_ai_core.orchestrator import leases as leases_mod

NODE_A = "tests.fixtures.mock_nodes.MinMaxNormalizer"
NODE_B = "tests.fixtures.mock_nodes.MockBinaryDecider"


def _register(sm, sid, name, *, class_name=None, data_module=None, path="."):
    """Register a plugin into the session catalog (metadata only, like LoadPlugin)."""
    capabilities = []
    if class_name is not None:
        capabilities.append({"class_name": class_name})
    if data_module is not None:
        capabilities.append(
            {
                "kind": "data_module",
                "class_name": "tests.fixtures.fake_data_modules.FakeCu3sDataModule",
                "data_module_name": data_module,
                "extras": [data_module],
            }
        )
    sm.get_session(sid).registered_plugins[name] = {
        "name": name,
        "path": path,
        "capabilities": capabilities,
    }


def _pipeline(plugins, class_names=()):
    return SimpleNamespace(
        plugins=list(plugins),
        nodes=[SimpleNamespace(class_name=name) for name in class_names],
    )


def _pipeline_bytes(plugins):
    return json.dumps(
        {"plugins": list(plugins), "nodes": [], "connections": []}
    ).encode("utf-8")


def _load_request(sid, plugins, data_module=None):
    request = cuvis_ai_pb2.LoadPipelineRequest(
        session_id=sid,
        pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=_pipeline_bytes(plugins)),
    )
    if data_module:
        request.data_module = data_module
    return request


class _RecordingSpawner(_InMemorySpawner):
    """In-memory spawner that keeps every handle it produced."""

    def __init__(self) -> None:
        self.handles: list[_InMemoryChildHandle] = []

    def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
        handle = super().spawn(
            venv_path, cwd=cwd, declared_paths=declared_paths, request_gpu=request_gpu
        )
        self.handles.append(handle)
        return handle


@pytest.fixture
def two_plugins():
    sm = SessionManager()
    sid = sm.create_session()
    _register(sm, sid, "plugin_a", class_name=NODE_A)
    _register(sm, sid, "plugin_b", class_name=NODE_B)
    spawner = _RecordingSpawner()
    orchestrator_bridge.set_spawner(spawner)
    yield sm, sid, spawner
    orchestrator_bridge.reset_spawner()


# ---------------------------------------------------------------------------
# child_can_serve: the reuse rule
# ---------------------------------------------------------------------------


def _resolved(sm, sid, plugins, class_names=(), data_module=None):
    return orchestrator_bridge._resolve_plugins(
        _pipeline(plugins, class_names), sm.get_session(sid), data_module
    )


def test_child_can_serve_table(two_plugins):
    sm, sid, _spawner = two_plugins
    _register(sm, sid, "dl", data_module="cu3s")
    session = sm.get_session(sid)

    child_set = _resolved(sm, sid, ["plugin_a", "plugin_b"], [NODE_A, NODE_B])
    session.resolved_plugins = dict(child_set)
    session.child_data_module = None

    same = _resolved(sm, sid, ["plugin_a", "plugin_b"], [NODE_A, NODE_B])
    subset = _resolved(sm, sid, ["plugin_a"], [NODE_A])
    superset = _resolved(sm, sid, ["plugin_a", "plugin_b", "dl"], [NODE_A])
    assert orchestrator_bridge.child_can_serve(session, same, None)
    assert orchestrator_bridge.child_can_serve(session, subset, None)
    assert not orchestrator_bridge.child_can_serve(session, superset, None)

    # The same name pointing at another manifest is another plugin.
    _register(sm, sid, "plugin_a", class_name=NODE_A, path="./elsewhere")
    changed = _resolved(sm, sid, ["plugin_a"], [NODE_A])
    assert not orchestrator_bridge.child_can_serve(session, changed, None)

    # A data module the child was not composed for needs a new env even when
    # the plugin set is unchanged; a child composed WITH it serves both.
    assert not orchestrator_bridge.child_can_serve(session, same, "cu3s")
    session.child_data_module = "cu3s"
    assert orchestrator_bridge.child_can_serve(session, same, "cu3s")
    assert orchestrator_bridge.child_can_serve(session, same, None)
    assert not orchestrator_bridge.child_can_serve(session, same, "tiff_paired")


# ---------------------------------------------------------------------------
# ensure_child_for_session: reuse vs swap
# ---------------------------------------------------------------------------


def test_ensure_child_swaps_child_when_pipeline_needs_plugin_the_child_lacks(
    two_plugins,
):
    sm, sid, spawner = two_plugins

    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    second = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_b"], [NODE_B])
    )

    assert second is not first
    assert first.returncode is not None
    assert first.retired_by_parent is True
    assert second.returncode is None
    assert second.retired_by_parent is False
    session = sm.get_session(sid)
    assert session.child_handle is second
    assert sorted(session.resolved_plugins) == ["plugin_b"]
    assert len(spawner.handles) == 2


def test_ensure_child_reuses_child_for_subset_plugin_set(two_plugins):
    sm, sid, spawner = two_plugins

    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a", "plugin_b"], [NODE_A, NODE_B])
    )
    again = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )

    assert again is first
    assert first.returncode is None
    assert len(spawner.handles) == 1
    # The child keeps the plugin set it was composed for, not the subset.
    assert sorted(sm.get_session(sid).resolved_plugins) == ["plugin_a", "plugin_b"]


def test_ensure_child_swaps_when_data_module_is_added(two_plugins):
    sm, sid, spawner = two_plugins
    _register(sm, sid, "dl", data_module="cu3s")
    cfg = _pipeline(["plugin_a", "dl"], [NODE_A])

    without = orchestrator_bridge.ensure_child_for_session(
        sm, sid, cfg, data_module=None
    )
    with_cu3s = orchestrator_bridge.ensure_child_for_session(
        sm, sid, cfg, data_module="cu3s"
    )
    assert with_cu3s is not without
    assert sm.get_session(sid).child_data_module == "cu3s"

    # The reverse direction reuses: an env composed with the extras serves a
    # run that needs none.
    plain = orchestrator_bridge.ensure_child_for_session(sm, sid, cfg, data_module=None)
    assert plain is with_cu3s
    assert len(spawner.handles) == 2


def test_compose_failure_on_swap_keeps_the_old_child_and_lease(two_plugins, tmp_path):
    sm, sid, spawner = two_plugins
    root = _lease_root(tmp_path)
    calls = {"n": 0}

    def composer(plugins, *, core_source, active_data_module=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return root / "digest0001" / ".venv"
        raise RuntimeError("uv lock failed: no network")

    orchestrator_bridge.set_composer(composer)
    lease_spawner = _LeaseAwareRecordingSpawner(root)
    orchestrator_bridge.set_spawner(lease_spawner)

    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    with pytest.raises(RuntimeError, match="no network"):
        orchestrator_bridge.ensure_child_for_session(
            sm, sid, _pipeline(["plugin_b"], [NODE_B])
        )

    session = sm.get_session(sid)
    assert session.child_handle is first
    assert first.returncode is None
    assert first.retired_by_parent is False
    assert sorted(session.resolved_plugins) == ["plugin_a"]
    ((_path, lease),) = leases_mod.read_leases(root)
    assert lease is not None and lease.entry_digest == "digest0001"
    assert len(lease_spawner.handles) == 1


def test_spawn_failure_after_detach_leaves_session_childless_and_next_load_recovers(
    two_plugins,
):
    sm, sid, spawner = two_plugins
    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )

    class _BoomOnce(_RecordingSpawner):
        def __init__(self) -> None:
            super().__init__()
            self.armed = True

        def spawn(self, *args, **kwargs):
            if self.armed:
                self.armed = False
                raise RuntimeError("spawn exploded")
            return super().spawn(*args, **kwargs)

    boom = _BoomOnce()
    orchestrator_bridge.set_spawner(boom)
    with pytest.raises(RuntimeError, match="spawn exploded"):
        orchestrator_bridge.ensure_child_for_session(
            sm, sid, _pipeline(["plugin_b"], [NODE_B])
        )

    session = sm.get_session(sid)
    assert first.returncode is not None  # the old child was retired before the spawn
    assert session.child_handle is None
    assert session.resolved_plugins is None

    recovered = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_b"], [NODE_B])
    )
    assert recovered is session.child_handle
    assert sorted(session.resolved_plugins) == ["plugin_b"]


# ---------------------------------------------------------------------------
# Leases across a swap
# ---------------------------------------------------------------------------


def _lease_root(tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    for digest in ("digest0001", "digest0002"):
        (root / digest / ".venv").mkdir(parents=True)
    leases_mod.ensure_root_marker(root)
    return root


class _LeaseAwareRecordingSpawner(_RecordingSpawner):
    """Handles borrow this process's pid so finalize_lease accepts them."""

    def __init__(self, lease_root: Path) -> None:
        super().__init__()
        self.lease_root = lease_root

    def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
        handle = super().spawn(
            venv_path, cwd=cwd, declared_paths=declared_paths, request_gpu=request_gpu
        )
        handle.process = SimpleNamespace(pid=os.getpid())
        return handle


def _counting_composer(root: Path):
    calls = {"n": 0}

    def composer(plugins, *, core_source, active_data_module=None):
        calls["n"] += 1
        return root / f"digest000{calls['n']}" / ".venv"

    return composer


def test_swap_replaces_the_session_lease(two_plugins, tmp_path):
    sm, sid, _spawner = two_plugins
    root = _lease_root(tmp_path)
    orchestrator_bridge.set_composer(_counting_composer(root))
    spawner = _LeaseAwareRecordingSpawner(root)
    orchestrator_bridge.set_spawner(spawner)

    orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_b"], [NODE_B])
    )

    ((_path, lease),) = leases_mod.read_leases(root)
    assert lease is not None
    assert lease.phase == "final"
    assert lease.entry_digest == "digest0002"
    assert lease.session_id == sid
    sm.close_session(sid)
    assert leases_mod.read_leases(root) == []


def test_close_session_after_swap_terminates_only_the_live_child(two_plugins):
    sm, sid, spawner = two_plugins
    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    second = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_b"], [NODE_B])
    )
    base = sm.get_session(sid).runtime_base_dir
    assert base is not None and base.exists()

    terminations: list[str] = []
    real_terminate = second.terminate

    def spying_terminate(grace_s=5.0):
        terminations.append("second")
        return real_terminate(grace_s)

    second.terminate = spying_terminate  # type: ignore[method-assign]
    first_code = first.returncode

    sm.close_session(sid)

    assert terminations == ["second"]
    assert first.returncode == first_code
    assert not base.exists()


# ---------------------------------------------------------------------------
# Confirmed exit
# ---------------------------------------------------------------------------


class _SurvivorHandle(_InMemoryChildHandle):
    """A child that ignores terminate and kill, as a hung process would."""

    def terminate(self, grace_s: float = 5.0):
        return None

    def kill(self):
        return None


class _SurvivorFirstSpawner(_RecordingSpawner):
    def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
        from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

        if not self.handles:
            handle = _SurvivorHandle(RunRuntimeServicer())
            self.handles.append(handle)
            return handle  # type: ignore[return-value]
        return super().spawn(
            venv_path, cwd=cwd, declared_paths=declared_paths, request_gpu=request_gpu
        )


def test_detach_child_refuses_a_survivor(two_plugins):
    sm, sid, _spawner = two_plugins
    spawner = _SurvivorFirstSpawner()
    orchestrator_bridge.set_spawner(spawner)

    survivor = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    with pytest.raises(ChildStillRunning):
        orchestrator_bridge.ensure_child_for_session(
            sm, sid, _pipeline(["plugin_b"], [NODE_B])
        )

    session = sm.get_session(sid)
    assert session.child_handle is survivor
    assert sorted(session.resolved_plugins) == ["plugin_a"]
    assert len(spawner.handles) == 1

    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_b"]), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.INTERNAL
    assert "could not be stopped" in ctx.details()


# ---------------------------------------------------------------------------
# A request that reaches the retired child
# ---------------------------------------------------------------------------


def test_inference_racing_a_swap_reports_replacement_not_crash(two_plugins):
    sm, sid, _spawner = two_plugins
    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_b"], [NODE_B])
    )
    assert first.retired_by_parent is True

    ctx = _InMemoryContext()
    orchestrator_bridge._propagate_child_failure(
        first,
        _InMemoryRpcError(grpc.StatusCode.UNAVAILABLE, "connection refused"),
        ctx,
        session=sm.get_session(sid),
    )

    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "replaced" in ctx.details()
    assert ctx.trailing_metadata() == ()
    assert sm.get_session(sid).crash_log_dir is None


def test_retired_child_never_preserves_crash_logs(two_plugins, monkeypatch):
    sm, sid, _spawner = two_plugins
    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_b"], [NODE_B])
    )

    def _must_not_run(*args, **kwargs):
        raise AssertionError("crash logs must not be preserved for a retired child")

    monkeypatch.setattr(orchestrator_bridge, "_preserve_crash_logs", _must_not_run)
    ctx = _InMemoryContext()
    orchestrator_bridge._propagate_child_failure(
        first,
        _InMemoryRpcError(grpc.StatusCode.INTERNAL, "stream reset"),
        ctx,
        session=None,
    )
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION


# ---------------------------------------------------------------------------
# forward_load_pipeline: compose failures, serialisation, close during a load
# ---------------------------------------------------------------------------


def test_compose_failure_maps_to_failed_precondition(two_plugins):
    sm, sid, _spawner = two_plugins
    from cuvis_ai_core.orchestrator.uv_runner import UvRunnerError

    def composer(plugins, *, core_source, active_data_module=None):
        raise UvRunnerError("uv sync failed: could not resolve torch")

    orchestrator_bridge.set_composer(composer)
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_a"]), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "could not resolve torch" in ctx.details()


def test_unclassified_compose_exception_stays_internal(two_plugins):
    sm, sid, _spawner = two_plugins

    def composer(plugins, *, core_source, active_data_module=None):
        raise KeyError("a bug, not a compose failure")

    orchestrator_bridge.set_composer(composer)
    with pytest.raises(KeyError):
        orchestrator_bridge.forward_load_pipeline(
            sm, _load_request(sid, ["plugin_a"]), _InMemoryContext()
        )


def test_competing_loads_on_one_session_serialise_whole_operation(
    two_plugins, monkeypatch
):
    sm, sid, spawner = two_plugins
    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    def slow_composer(plugins, *, core_source, active_data_module=None):
        with guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with guard:
            state["active"] -= 1
        return Path("in-memory-venv")

    orchestrator_bridge.set_composer(slow_composer)

    def slow_forward(
        child, stub_method, request, context, empty_response_factory, *, session=None
    ):
        with guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with guard:
            state["active"] -= 1
        return cuvis_ai_pb2.LoadPipelineResponse(success=True)

    monkeypatch.setattr(
        orchestrator_bridge, "_call_child_with_error_propagation", slow_forward
    )

    results: dict[str, cuvis_ai_pb2.LoadPipelineResponse] = {}

    def load(plugin):
        results[plugin] = orchestrator_bridge.forward_load_pipeline(
            sm, _load_request(sid, [plugin]), _InMemoryContext()
        )

    threads = [
        threading.Thread(target=load, args=(p,)) for p in ("plugin_a", "plugin_b")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(r.success for r in results.values())
    assert state["max_active"] == 1
    assert len(spawner.handles) == 2
    live = sm.get_session(sid).child_handle
    assert live is spawner.handles[-1]
    assert spawner.handles[0].retired_by_parent is True


def test_close_during_spawn_leaves_no_orphan(two_plugins, tmp_path):
    sm, sid, _spawner = two_plugins
    root = _lease_root(tmp_path)
    orchestrator_bridge.set_composer(_counting_composer(root))

    class _ClosingSpawner(_LeaseAwareRecordingSpawner):
        def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
            handle = super().spawn(
                venv_path,
                cwd=cwd,
                declared_paths=declared_paths,
                request_gpu=request_gpu,
            )
            # The close arrives while the spawn is in flight (same thread here:
            # the per-session lock is re-entrant, another thread would block
            # until the load finished and then find the child to tear down).
            sm.close_session(sid)
            return handle

    spawner = _ClosingSpawner(root)
    orchestrator_bridge.set_spawner(spawner)

    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_a"]), ctx
    )

    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.NOT_FOUND
    assert len(spawner.handles) == 1
    assert spawner.handles[0].returncode is not None
    assert leases_mod.read_leases(root) == []
    assert not sm.has_session(sid)
