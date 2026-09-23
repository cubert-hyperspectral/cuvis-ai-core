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
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

from cuvis_ai_core.grpc import orchestrator_bridge
from cuvis_ai_core.grpc import session_manager as session_manager_mod
from cuvis_ai_core.grpc.orchestrator_bridge import (
    _InMemoryChildHandle,
    _InMemoryContext,
    _InMemoryRpcError,
    _InMemorySpawner,
    _InMemoryStub,
)
from cuvis_ai_core.grpc.session_manager import ChildStillRunning, SessionManager
from cuvis_ai_core.orchestrator import leases as leases_mod
from cuvis_ai_core.orchestrator.spawner import SpawnError
from cuvis_ai_core.orchestrator.uv_runner import UvRunnerError


def _scratch_tree(sid: str) -> Path:
    """The per-session scratch tree the bridge declares for a child."""
    return Path(tempfile.gettempdir()) / "cuvis_runtime_sessions" / sid


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

    # The same source with regenerated metadata (tags, an icon, another capability
    # order) is the same install: the venv does not change, so the warm child stays.
    _register(sm, sid, "plugin_a", class_name=NODE_A)
    catalog = sm.get_session(sid).registered_plugins
    catalog["plugin_a"]["capabilities"][0]["tags"] = ["regenerated"]
    regenerated = _resolved(sm, sid, ["plugin_a"], [NODE_A])
    assert regenerated["plugin_a"].model_dump() != child_set["plugin_a"].model_dump()
    assert orchestrator_bridge.child_can_serve(session, regenerated, None)

    # A data module the child was not composed for needs a new env even when
    # the plugin set is unchanged; a child composed WITH it serves both.
    assert not orchestrator_bridge.child_can_serve(session, same, "cu3s")
    session.child_data_module = "cu3s"
    assert orchestrator_bridge.child_can_serve(session, same, "cu3s")
    assert orchestrator_bridge.child_can_serve(session, same, None)
    assert not orchestrator_bridge.child_can_serve(session, same, "tiff_paired")

    # A child the parent already told to stop serves nothing, whatever its set.
    session.child_handle = SimpleNamespace(retired_by_parent=True, returncode=None)
    assert not orchestrator_bridge.child_can_serve(session, same, "cu3s")


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
        raise UvRunnerError("uv lock failed: no network")

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

    # Through the RPC path the same failure is the server's answer, not a
    # fault, and the old child keeps serving.
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_b"]), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "no network" in ctx.details()
    assert session.child_handle is first
    assert first.returncode is None
    ((_path, lease),) = leases_mod.read_leases(root)
    assert lease is not None and lease.entry_digest == "digest0001"


def test_spawn_failure_after_retire_leaves_session_childless_and_next_load_recovers(
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
    # The scratch tree declared for the failed spawn is recorded, so a close
    # removes it even though no child ever attached.
    assert session.runtime_base_dir == _scratch_tree(sid)
    assert session.runtime_base_dir.exists()

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

    # Spy both handles: the in-memory terminate always reports 0, so the
    # retired child's return code alone could not show a second stop.
    terminations: list[str] = []
    for name, handle in (("first", first), ("second", second)):

        def spying_terminate(grace_s=5.0, _name=name, _real=handle.terminate):
            terminations.append(_name)
            return _real(grace_s)

        def spying_kill(_name=name, _real=handle.kill):
            terminations.append(_name)
            return _real()

        handle.terminate = spying_terminate  # type: ignore[method-assign]
        handle.kill = spying_kill  # type: ignore[method-assign]

    sm.close_session(sid)

    assert terminations == ["second"]
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


def test_retire_child_refuses_a_survivor(two_plugins):
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
    # The survivor stays attached for the retry but MARKED: StopRun, terminate
    # and kill were delivered, so it serves nothing any more.
    assert survivor.retired_by_parent is True

    # The server's answer, not a fault: a client that retried INTERNAL on a
    # fresh session would spawn a runtime beside the survivor.
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_b"]), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "could not be stopped" in ctx.details()

    # A request that still reaches the stopping child is a replacement (ABORTED,
    # repeat it once the stop went through), not a crash.
    ctx = _InMemoryContext()
    orchestrator_bridge._propagate_child_failure(
        survivor,
        _InMemoryRpcError(grpc.StatusCode.UNAVAILABLE, "connection refused"),
        ctx,
        session=session,
    )
    assert ctx.code() is grpc.StatusCode.ABORTED
    assert ctx.trailing_metadata() == ()

    # The kill lands late. The next load sees an exited child the parent had
    # stopped: no crash logs, a fresh child, and the pipeline it asked for.
    survivor._returncode = 1
    recovered = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_b"], [NODE_B])
    )
    assert recovered is not survivor
    assert session.child_handle is recovered
    assert sorted(session.resolved_plugins) == ["plugin_b"]
    assert session.crash_log_dir is None
    assert len(spawner.handles) == 2


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

    # Through the forwarding wrapper, the production path: the retired child's
    # servicer refuses the pipeline-less Inference, the RpcError reaches the
    # marker check first.
    ctx = _InMemoryContext()
    resp = orchestrator_bridge._call_child_with_error_propagation(
        first,
        "Inference",
        cuvis_ai_pb2.InferenceRequest(session_id=sid),
        ctx,
        lambda: cuvis_ai_pb2.InferenceResponse(),
        session=sm.get_session(sid),
    )

    assert resp == cuvis_ai_pb2.InferenceResponse()
    # ABORTED, not FAILED_PRECONDITION: the answer means "repeat the request",
    # and the client shows precondition failures once without retrying.
    assert ctx.code() is grpc.StatusCode.ABORTED
    assert "replaced" in ctx.details()
    assert ctx.trailing_metadata() == ()
    assert sm.get_session(sid).crash_log_dir is None


def test_dead_child_recovery_keeps_crash_status_for_racing_requests(
    two_plugins, monkeypatch, tmp_path
):
    """A child that died on its own is not marked; a racing request keeps the crash."""
    sm, sid, _spawner = two_plugins
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    first = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    # The child crashed: its exit code is set and a stop cannot change it (the
    # in-memory terminate would otherwise report 0 for a process that is gone).
    first._returncode = 137
    first.terminate = lambda grace_s=5.0: 137  # type: ignore[method-assign]

    second = orchestrator_bridge.ensure_child_for_session(
        sm, sid, _pipeline(["plugin_a"], [NODE_A])
    )
    assert second is not first
    assert first.retired_by_parent is False

    preserved: list[object] = []
    monkeypatch.setattr(
        orchestrator_bridge,
        "_preserve_crash_logs",
        lambda child, session: preserved.append(child) or None,
    )
    ctx = _InMemoryContext()
    orchestrator_bridge._propagate_child_failure(
        first,
        _InMemoryRpcError(grpc.StatusCode.UNAVAILABLE, "connection refused"),
        ctx,
        session=sm.get_session(sid),
    )
    assert ctx.code() is grpc.StatusCode.INTERNAL
    assert "exited unexpectedly" in ctx.details()
    assert preserved == [first]
    trailers = dict(ctx.trailing_metadata())
    assert trailers[orchestrator_bridge.TRAILER_CHILD_EXIT_CODE] == "137"


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
    assert ctx.code() is grpc.StatusCode.ABORTED


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


@pytest.mark.parametrize("bug", [KeyError, NotImplementedError, RecursionError])
def test_unclassified_compose_exception_propagates_to_the_servicer(two_plugins, bug):
    """A bug is not mapped: it escapes the bridge (the undecorated servicer
    reports it as UNKNOWN) instead of being dressed up as a precondition.
    NotImplementedError and RecursionError are RuntimeErrors, so the mapping
    has to name the compose / spawn failure classes, not the base class."""
    sm, sid, _spawner = two_plugins

    def composer(plugins, *, core_source, active_data_module=None):
        raise bug("a bug, not a compose failure")

    orchestrator_bridge.set_composer(composer)
    with pytest.raises(bug):
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
        threading.Thread(target=load, args=(p,), daemon=True)
        for p in ("plugin_a", "plugin_b")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not any(t.is_alive() for t in threads), (
        "load threads did not finish (lock held?)"
    )
    assert set(results) == {"plugin_a", "plugin_b"}
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
    # The scratch tree recreated for the fresh child went with it.
    assert not _scratch_tree(sid).exists()


def test_close_from_another_thread_waits_for_the_load_then_tears_down(
    two_plugins, tmp_path, monkeypatch
):
    """A CloseSession on another thread blocks on the session's lock while the load is
    spawning; the load then finds the session closing and disposes of the child it
    spawned: no orphan, no lease, and no child attached to a closed session."""
    sm, sid, _spawner = two_plugins
    root = _lease_root(tmp_path)
    orchestrator_bridge.set_composer(_counting_composer(root))
    monkeypatch.setattr(
        orchestrator_bridge,
        "_call_child_with_error_propagation",
        lambda *a, **k: cuvis_ai_pb2.LoadPipelineResponse(success=True),
    )
    entered, release = threading.Event(), threading.Event()

    class _GatedSpawner(_LeaseAwareRecordingSpawner):
        def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
            entered.set()
            assert release.wait(timeout=10)
            return super().spawn(
                venv_path,
                cwd=cwd,
                declared_paths=declared_paths,
                request_gpu=request_gpu,
            )

    spawner = _GatedSpawner(root)
    orchestrator_bridge.set_spawner(spawner)
    ctx = _InMemoryContext()
    result: dict[str, cuvis_ai_pb2.LoadPipelineResponse] = {}

    def load():
        result["resp"] = orchestrator_bridge.forward_load_pipeline(
            sm, _load_request(sid, ["plugin_a"]), ctx
        )

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    assert entered.wait(timeout=5)
    closer = threading.Thread(target=lambda: sm.close_session(sid), daemon=True)
    closer.start()
    closer.join(timeout=0.3)
    assert closer.is_alive(), "close_session did not wait for the load's lock"

    release.set()
    loader.join(timeout=10)
    closer.join(timeout=10)
    assert not loader.is_alive() and not closer.is_alive()
    # The close had marked the session before it waited; the load, back from the
    # spawn, finds the mark, stops the fresh child itself and answers NOT_FOUND
    # instead of attaching a child to a session that is going away.
    assert result["resp"].success is False
    assert ctx.code() is grpc.StatusCode.NOT_FOUND
    handle = spawner.handles[0]
    assert handle.returncode is not None
    assert leases_mod.read_leases(root) == []
    assert not sm.has_session(sid)
    assert not _scratch_tree(sid).exists()


class _HungLoadHandle(_InMemoryChildHandle):
    """A child whose LoadPipeline never answers until the parent stops it."""

    def __init__(self, servicer) -> None:
        super().__init__(servicer)
        self.entered = threading.Event()
        self.released = threading.Event()

    def stub(self):
        outer = self

        class _HungStub(_InMemoryStub):
            def LoadPipeline(self, request, timeout=None):
                outer.entered.set()
                outer.released.wait(timeout=10)
                raise _InMemoryRpcError(
                    grpc.StatusCode.UNAVAILABLE, "connection refused"
                )

        return _HungStub(self._servicer)

    def terminate(self, grace_s: float = 5.0):
        self.released.set()
        return super().terminate(grace_s)


def test_close_session_breaks_a_hung_forwarded_load(two_plugins, monkeypatch):
    """The forwarded LoadPipeline has no deadline. A child that hangs inside it holds
    the session's lock; after the grace the close stops the child, the load unwinds
    with a replacement answer, and the teardown proceeds."""
    sm, sid, _spawner = two_plugins
    monkeypatch.setattr(session_manager_mod, "CLOSE_LOCK_GRACE_SECONDS", 0.2)

    class _HungSpawner(_RecordingSpawner):
        def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
            from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

            handle = _HungLoadHandle(RunRuntimeServicer())
            self.handles.append(handle)
            return handle  # type: ignore[return-value]

    spawner = _HungSpawner()
    orchestrator_bridge.set_spawner(spawner)
    ctx = _InMemoryContext()
    result: dict[str, cuvis_ai_pb2.LoadPipelineResponse] = {}

    def load():
        result["resp"] = orchestrator_bridge.forward_load_pipeline(
            sm, _load_request(sid, ["plugin_a"]), ctx
        )

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    deadline = time.monotonic() + 5
    while not spawner.handles and time.monotonic() < deadline:
        time.sleep(0.01)
    hung = spawner.handles[0]
    assert hung.entered.wait(timeout=5)

    started = time.monotonic()
    closer = threading.Thread(target=lambda: sm.close_session(sid), daemon=True)
    closer.start()
    closer.join(timeout=5)
    assert not closer.is_alive(), "close_session hung behind the forwarded load"
    loader.join(timeout=5)
    assert not loader.is_alive()
    assert time.monotonic() - started < 4

    assert hung.returncode is not None
    assert hung.retired_by_parent is True
    assert result["resp"].success is False
    assert ctx.code() is grpc.StatusCode.ABORTED
    assert "stopped by the server" in ctx.details()
    assert ctx.trailing_metadata() == ()
    assert not sm.has_session(sid)


# ---------------------------------------------------------------------------
# A close that races a compose or a spawn does not wait for them
# ---------------------------------------------------------------------------


def _gated_composer(inner, entered: threading.Event, release: threading.Event):
    """A composer that blocks like a cold ``uv sync`` until the test releases it."""

    def composer(plugins, *, core_source, active_data_module=None):
        entered.set()
        assert release.wait(timeout=10)
        return inner(
            plugins, core_source=core_source, active_data_module=active_data_module
        )

    return composer


def _short_close_grace(monkeypatch) -> None:
    monkeypatch.setattr(session_manager_mod, "CLOSE_LOCK_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(session_manager_mod, "CLOSE_LOCK_RETRY_SECONDS", 0.1)


def test_close_during_a_slow_compose_returns_within_the_grace(
    two_plugins, tmp_path, monkeypatch
):
    """A CloseSession while the load is still inside the composer (a cold build takes
    minutes) must not wait for it: after the grace the close tears down without the lock,
    and the load, back from the compose, finds the session closed and spawns nothing."""
    sm, sid, _spawner = two_plugins
    _short_close_grace(monkeypatch)
    root = _lease_root(tmp_path)
    entered, release = threading.Event(), threading.Event()
    orchestrator_bridge.set_composer(
        _gated_composer(_counting_composer(root), entered, release)
    )
    spawner = _LeaseAwareRecordingSpawner(root)
    orchestrator_bridge.set_spawner(spawner)
    ctx = _InMemoryContext()
    result: dict[str, cuvis_ai_pb2.LoadPipelineResponse] = {}

    def load():
        result["resp"] = orchestrator_bridge.forward_load_pipeline(
            sm, _load_request(sid, ["plugin_a"]), ctx
        )

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    assert entered.wait(timeout=5)

    started = time.monotonic()
    closer = threading.Thread(target=lambda: sm.close_session(sid), daemon=True)
    closer.start()
    closer.join(timeout=5)
    assert not closer.is_alive(), "close_session waited for the compose"
    assert time.monotonic() - started < 2
    assert not sm.has_session(sid)

    release.set()
    loader.join(timeout=10)
    assert not loader.is_alive()
    assert result["resp"].success is False
    assert ctx.code() is grpc.StatusCode.NOT_FOUND
    assert spawner.handles == [], "a child was spawned for a closed session"
    assert leases_mod.read_leases(root) == []
    assert not _scratch_tree(sid).exists()


def test_close_during_a_spawn_stops_the_child_before_anything_is_forwarded(
    two_plugins, monkeypatch
):
    """The close hits while the load is spawning (health poll, InitializeSession); the
    grace finds no child to stop. The load must notice the close once the spawn returns
    and stop the fresh child before forwarding into it: a LoadPipeline that hangs there
    would otherwise hold the lock, and the session, forever."""
    sm, sid, _spawner = two_plugins
    _short_close_grace(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    class _GatedHungSpawner(_RecordingSpawner):
        def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
            from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

            entered.set()
            assert release.wait(timeout=10)
            handle = _HungLoadHandle(RunRuntimeServicer())
            self.handles.append(handle)
            return handle  # type: ignore[return-value]

    spawner = _GatedHungSpawner()
    orchestrator_bridge.set_spawner(spawner)
    ctx = _InMemoryContext()
    result: dict[str, cuvis_ai_pb2.LoadPipelineResponse] = {}

    def load():
        result["resp"] = orchestrator_bridge.forward_load_pipeline(
            sm, _load_request(sid, ["plugin_a"]), ctx
        )

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    assert entered.wait(timeout=5)

    started = time.monotonic()
    closer = threading.Thread(target=lambda: sm.close_session(sid), daemon=True)
    closer.start()
    closer.join(timeout=5)
    assert not closer.is_alive(), "close_session waited for the spawn"
    assert time.monotonic() - started < 2
    assert not sm.has_session(sid)

    release.set()
    loader.join(timeout=10)
    assert not loader.is_alive()
    hung = spawner.handles[0]
    assert not hung.entered.is_set(), "LoadPipeline was forwarded into a closed session"
    assert hung.returncode is not None
    assert result["resp"].success is False
    assert ctx.code() is grpc.StatusCode.NOT_FOUND
    assert not _scratch_tree(sid).exists()


def test_no_child_during_a_load_is_aborted_not_a_precondition(two_plugins):
    """Inference that finds no child while a load owns the session is racing a switch:
    ABORTED, repeat the request. Without a load in flight the same answer is a real
    precondition failure: no pipeline was ever loaded."""
    sm, sid, _spawner = two_plugins
    session = sm.get_session(sid)
    request = cuvis_ai_pb2.InferenceRequest(session_id=sid)

    ctx = _InMemoryContext()
    orchestrator_bridge.forward_inference(sm, request, ctx)
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION

    session.load_in_flight = True
    ctx = _InMemoryContext()
    orchestrator_bridge.forward_inference(sm, request, ctx)
    assert ctx.code() is grpc.StatusCode.ABORTED
    assert "load" in ctx.details().lower()
    session.load_in_flight = False

    # The flag is the load's own: set for the whole forwarded operation, clear after.
    orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_a"]), _InMemoryContext()
    )
    assert session.load_in_flight is False


# ---------------------------------------------------------------------------
# Spawn-window failures are the server's answer
# ---------------------------------------------------------------------------


def test_initialize_session_rpc_error_maps_to_failed_precondition(two_plugins):
    """A plugin import failure inside the child's InitializeSession arrives as an
    RpcError (the child servicer is undecorated). It is a spawn failure: one
    FAILED_PRECONDITION with the cause, the child stopped, the scratch tree recorded."""
    sm, sid, _spawner = two_plugins

    class _RefusingInitHandle(_InMemoryChildHandle):
        def stub(self):
            class _Stub(_InMemoryStub):
                def InitializeSession(self, request, timeout=None):
                    raise _InMemoryRpcError(
                        grpc.StatusCode.UNKNOWN,
                        "Exception calling application: No module named 'cuvis_ai_sam3'",
                    )

            return _Stub(self._servicer)

    class _RefusingSpawner(_RecordingSpawner):
        def spawn(self, venv_path, *, cwd, declared_paths, request_gpu=False):
            from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

            handle = _RefusingInitHandle(RunRuntimeServicer())
            self.handles.append(handle)
            return handle  # type: ignore[return-value]

    spawner = _RefusingSpawner()
    orchestrator_bridge.set_spawner(spawner)
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_a"]), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "InitializeSession" in ctx.details()
    assert "No module named" in ctx.details()
    assert spawner.handles[0].returncode is not None
    session = sm.get_session(sid)
    assert session.child_handle is None
    assert session.runtime_base_dir == _scratch_tree(sid)
    assert session.runtime_base_dir.exists()

    sm.close_session(sid)
    assert not _scratch_tree(sid).exists()


def test_os_error_from_spawn_maps_to_failed_precondition(two_plugins):
    """Popen / mkdtemp failures below the spawner's own SpawnError are spawn failures too."""
    sm, sid, _spawner = two_plugins

    class _OsErrorSpawner(_RecordingSpawner):
        def spawn(self, *args, **kwargs):
            raise PermissionError("[WinError 5] Access is denied: child.stderr.log")

    orchestrator_bridge.set_spawner(_OsErrorSpawner())
    ctx = _InMemoryContext()
    resp = orchestrator_bridge.forward_load_pipeline(
        sm, _load_request(sid, ["plugin_a"]), ctx
    )
    assert resp.success is False
    assert ctx.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "Could not start the child runtime" in ctx.details()
    assert "Access is denied" in ctx.details()
    assert sm.get_session(sid).child_handle is None
    with pytest.raises(SpawnError):
        orchestrator_bridge.ensure_child_for_session(
            sm, sid, _pipeline(["plugin_a"], [NODE_A])
        )
