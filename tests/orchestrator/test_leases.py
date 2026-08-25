"""Lease lifecycle, liveness guards, and reaper behaviour.

Everything runs against fabricated pids / snapshots — no real child
processes are spawned and nothing outside pytest tmp dirs is touched.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psutil
import pytest

from cuvis_ai_core.orchestrator import leases as leases_mod

# A pid that certainly does not exist (Windows pids are DWORDs but real
# ones stay far below this; psutil raises NoSuchProcess).
DEAD_PID = 0x7FFF_FFF0


def _marked_root(tmp_path: Path) -> Path:
    """A cache root carrying the composer marker (maintenance allowed)."""
    root = tmp_path / "cache"
    root.mkdir()
    leases_mod.ensure_root_marker(root)
    return root


@pytest.fixture(autouse=True)
def _isolate_scratch_sweep(tmp_path: Path, monkeypatch):
    """Point the reaper's scratch sweep at a private temp dir.

    Every reap pass sweeps ``<tempdir>/cuvis_runtime_sessions``; without
    this, the suite would walk the developer's real temp dir. Tests that
    exercise the sweep override the location themselves.
    """
    monkeypatch.setattr(
        leases_mod.tempfile, "gettempdir", lambda: str(tmp_path / "isolated_temp")
    )


@pytest.fixture(autouse=True)
def _isolate_crash_store(tmp_path: Path, monkeypatch):
    """Point the crash-log store at a private temp dir.

    Reaping a dead orphan preserves its logs via ``crash_logs``, whose
    default store lives under the developer's real run-cache root.
    """
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crash_store"))


# ---------------------------------------------------------------------------
# Lease files: write / finalize / read / remove / corruption
# ---------------------------------------------------------------------------


def test_intent_then_finalize_roundtrip(tmp_path: Path):
    root = _marked_root(tmp_path)
    leases_mod.write_intent_lease(root, "sid-1", "digest-a")
    ((path, lease),) = leases_mod.read_leases(root)
    assert lease is not None
    assert lease.phase == "intent"
    assert lease.entry_digest == "digest-a"
    assert lease.child_pid is None
    assert leases_mod.pid_alive(lease.parent_pid, lease.parent_create_time)

    session_root = tmp_path / "cuvis_runtime_sessions" / "sid-1"
    leases_mod.finalize_lease(
        root, "sid-1", "digest-a", child_pid=os.getpid(), session_root=session_root
    )
    ((path, lease),) = leases_mod.read_leases(root)
    assert lease is not None
    assert lease.phase == "final"
    assert lease.child_pid == os.getpid()
    assert lease.session_root == str(session_root)
    assert leases_mod.pid_alive(lease.child_pid, lease.child_create_time)


def test_lease_writes_leave_no_tmp_files(tmp_path: Path):
    root = _marked_root(tmp_path)
    leases_mod.write_intent_lease(root, "sid-1", "digest-a")
    leftovers = list(leases_mod.leases_dir(root).glob("*.tmp"))
    assert leftovers == []


def test_remove_lease_missing_is_fine(tmp_path: Path):
    root = _marked_root(tmp_path)
    leases_mod.remove_lease(root, "never-written")  # must not raise


def test_corrupt_lease_reported_and_quarantined(tmp_path: Path):
    root = _marked_root(tmp_path)
    directory = leases_mod.leases_dir(root)
    directory.mkdir(parents=True)
    bad = directory / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    ((path, lease),) = leases_mod.read_leases(root)
    assert lease is None
    leases_mod.quarantine_lease(path)
    assert not bad.exists()
    assert bad.with_suffix(".corrupt").exists()


def test_finalize_dead_child_raises(tmp_path: Path):
    root = _marked_root(tmp_path)
    with pytest.raises(psutil.NoSuchProcess):
        leases_mod.finalize_lease(
            root, "sid-1", "digest-a", child_pid=DEAD_PID, session_root=tmp_path
        )


# ---------------------------------------------------------------------------
# Liveness + root guards
# ---------------------------------------------------------------------------


def test_pid_alive_matches_create_time():
    me = psutil.Process()
    assert leases_mod.pid_alive(me.pid, me.create_time())
    assert not leases_mod.pid_alive(me.pid, me.create_time() - 10_000)  # recycled
    assert not leases_mod.pid_alive(DEAD_PID, time.time())
    assert not leases_mod.pid_alive(None, None)


def test_root_guard_requires_marker(tmp_path: Path):
    bare = tmp_path / "not_a_cache"
    bare.mkdir()
    assert not leases_mod.root_guard_ok(bare)
    leases_mod.ensure_root_marker(bare)
    assert leases_mod.root_guard_ok(bare)


def test_root_guard_refuses_unc_paths():
    assert not leases_mod.root_guard_ok(Path(r"\\server\share\cuvis_runs"))


def test_adopt_root_requires_locks_dir(tmp_path: Path):
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    leases_mod.adopt_root_if_composer_shaped(stranger)
    assert not (stranger / leases_mod.ROOT_MARKER_NAME).exists()

    legacy = tmp_path / "legacy_cache"
    (legacy / ".locks").mkdir(parents=True)
    leases_mod.adopt_root_if_composer_shaped(legacy)
    assert (legacy / leases_mod.ROOT_MARKER_NAME).exists()


def _error_messages(action) -> list[str]:
    """Run ``action`` and collect every ERROR-level loguru message it emits."""
    from loguru import logger

    messages: list[str] = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="ERROR")
    try:
        action()
    finally:
        logger.remove(handler_id)
    return messages


def test_root_guard_is_quiet_before_the_first_compose(tmp_path: Path):
    """A missing or empty root is a fresh install, not a mispointed directory."""
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()
    errors = _error_messages(
        lambda: (leases_mod.root_guard_ok(missing), leases_mod.root_guard_ok(empty))
    )
    assert not leases_mod.root_guard_ok(missing)
    assert not leases_mod.root_guard_ok(empty)
    assert errors == []


def test_root_guard_stays_loud_for_unmarked_root_with_content(tmp_path: Path):
    stranger = tmp_path / "stranger"
    (stranger / "somebody_elses_data").mkdir(parents=True)
    errors = _error_messages(lambda: leases_mod.root_guard_ok(stranger))
    assert len(errors) == 1
    assert leases_mod.ROOT_MARKER_NAME in errors[0]


def test_reap_adopts_legacy_root_before_guarding(tmp_path: Path):
    """The first post-upgrade pass reaps a pre-marker root instead of skipping it."""
    legacy = tmp_path / "legacy_cache"
    (legacy / ".locks").mkdir(parents=True)
    path = _write_raw_lease(
        legacy, "sid-i", phase="intent", parent_pid=DEAD_PID, parent_create_time=1.0
    )
    leases_mod.reap_orphans(legacy)
    assert (legacy / leases_mod.ROOT_MARKER_NAME).exists()
    assert not path.exists()


# ---------------------------------------------------------------------------
# Process snapshot path predicates
# ---------------------------------------------------------------------------


def _info(**kwargs) -> leases_mod.ProcessInfo:
    """A ProcessInfo with harmless defaults, overridable per test."""
    defaults = dict(
        pid=1234,
        create_time=time.time(),
        ppid=1,
        exe=None,
        cwd=None,
        is_run_runtime=False,
    )
    defaults.update(kwargs)
    return leases_mod.ProcessInfo(**defaults)


def test_snapshot_exe_and_cwd_under(tmp_path: Path):
    entry = tmp_path / "abcd1234"
    snapshot = leases_mod.ProcessSnapshot(
        infos=(
            _info(exe=str(entry / ".venv" / "python.exe")),
            _info(cwd=str(entry / "work")),
        )
    )
    assert snapshot.exe_under(entry)
    assert snapshot.cwd_under(entry)
    assert not snapshot.exe_under(tmp_path / "other")
    assert not snapshot.cwd_under(tmp_path / "other")


def test_take_process_snapshot_sees_this_test_process():
    snapshot = leases_mod.take_process_snapshot()
    assert any(info.pid == os.getpid() for info in snapshot.infos)


# ---------------------------------------------------------------------------
# _kill_runtime_tree never-kill guards
# ---------------------------------------------------------------------------


def test_kill_tree_refuses_cmdline_mismatch():
    """This test process is not a child runtime — it must never be killed."""
    me = psutil.Process()
    assert leases_mod._kill_runtime_tree(me.pid, me.create_time()) is False
    assert psutil.pid_exists(me.pid)


def test_kill_tree_refuses_recycled_pid():
    me = psutil.Process()
    assert leases_mod._kill_runtime_tree(me.pid, me.create_time() - 10_000) is False
    assert psutil.pid_exists(me.pid)


def test_kill_tree_reports_already_dead():
    assert leases_mod._kill_runtime_tree(DEAD_PID, time.time()) is True


# ---------------------------------------------------------------------------
# Guarded rmtree
# ---------------------------------------------------------------------------


def test_guarded_rmtree_refuses_outside_scratch_root(tmp_path: Path):
    victim = tmp_path / "precious"
    victim.mkdir()
    (victim / "data.txt").write_text("keep me", encoding="utf-8")
    leases_mod._guarded_rmtree(victim)
    assert victim.exists()


def test_guarded_rmtree_refuses_the_scratch_root_itself(tmp_path: Path):
    scratch_root = tmp_path / leases_mod.SCRATCH_ROOT_NAME
    scratch_root.mkdir()
    leases_mod._guarded_rmtree(scratch_root)
    assert scratch_root.exists()


def test_guarded_rmtree_removes_session_dirs(tmp_path: Path):
    session = tmp_path / leases_mod.SCRATCH_ROOT_NAME / "sid-1"
    session.mkdir(parents=True)
    leases_mod._guarded_rmtree(session)
    assert not session.exists()


# ---------------------------------------------------------------------------
# reap_orphans
# ---------------------------------------------------------------------------


def _write_raw_lease(root: Path, session_id: str, **fields) -> Path:
    """Write a lease file with full control over every field."""
    payload = dict(
        phase="final",
        entry_digest="digest-a",
        session_id=session_id,
        parent_pid=os.getpid(),
        parent_create_time=psutil.Process().create_time(),
        child_pid=None,
        child_create_time=None,
        session_root=None,
        created_at=time.time(),
    )
    payload.update(fields)
    directory = leases_mod.leases_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _session_root_with_logs(tmp_path: Path, sid: str) -> Path:
    """A fake per-session scratch tree containing child logs."""
    session_root = tmp_path / leases_mod.SCRATCH_ROOT_NAME / sid
    runtime = session_root / "scratch" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "child.stderr.log").write_text("boom", encoding="utf-8")
    (runtime / "child.stdout.log").write_text("out", encoding="utf-8")
    return session_root


def test_reap_refuses_unmarked_root(tmp_path: Path):
    bare = tmp_path / "bare"
    bare.mkdir()
    _write_raw_lease(bare, "sid-x", parent_pid=DEAD_PID, parent_create_time=1.0)
    leases_mod.reap_orphans(bare)
    assert (leases_mod.leases_dir(bare) / "sid-x.json").exists()  # untouched


def test_reap_keeps_live_parent_leases(tmp_path: Path):
    root = _marked_root(tmp_path)
    path = _write_raw_lease(root, "sid-live")  # parent = this test process
    leases_mod.reap_orphans(root)
    assert path.exists()


def test_reap_removes_stale_intent_lease(tmp_path: Path):
    root = _marked_root(tmp_path)
    path = _write_raw_lease(
        root, "sid-i", phase="intent", parent_pid=DEAD_PID, parent_create_time=1.0
    )
    leases_mod.reap_orphans(root)
    assert not path.exists()


def test_reap_quarantines_corrupt_lease(tmp_path: Path):
    root = _marked_root(tmp_path)
    directory = leases_mod.leases_dir(root)
    directory.mkdir(parents=True)
    (directory / "junk.json").write_text("]]]", encoding="utf-8")
    leases_mod.reap_orphans(root)
    assert not (directory / "junk.json").exists()
    assert (directory / "junk.corrupt").exists()


def test_reap_cleans_dead_orphan_and_preserves_logs(tmp_path: Path):
    root = _marked_root(tmp_path)
    session_root = _session_root_with_logs(tmp_path, "sid-d")
    path = _write_raw_lease(
        root,
        "sid-d",
        parent_pid=DEAD_PID,
        parent_create_time=1.0,
        child_pid=DEAD_PID,
        child_create_time=1.0,
        session_root=str(session_root),
    )
    leases_mod.reap_orphans(root)
    assert not path.exists()
    assert not session_root.exists()
    preserved = list((tmp_path / "crash_store").glob("*-sid-d/*"))
    names = {p.name for p in preserved}
    assert "child.stderr.log" in names
    assert "child.stdout.log" in names
    # The reaper has no exit code for a child whose parent is gone.
    assert "crash_info.txt" in names
    marker = next(p for p in preserved if p.name == "crash_info.txt")
    assert "exit_code: unknown" in marker.read_text(encoding="utf-8")


def test_reap_keeps_lease_when_kill_fails(tmp_path: Path, monkeypatch):
    root = _marked_root(tmp_path)
    session_root = _session_root_with_logs(tmp_path, "sid-k")
    me = psutil.Process()
    path = _write_raw_lease(
        root,
        "sid-k",
        parent_pid=DEAD_PID,
        parent_create_time=1.0,
        child_pid=me.pid,
        child_create_time=me.create_time(),
        session_root=str(session_root),
    )
    monkeypatch.setattr(leases_mod, "_kill_runtime_tree", lambda pid, ct: False)
    leases_mod.reap_orphans(root)
    assert path.exists()  # protection record survives a failed kill
    assert session_root.exists()


def test_reap_cleans_after_confirmed_kill(tmp_path: Path, monkeypatch):
    root = _marked_root(tmp_path)
    session_root = _session_root_with_logs(tmp_path, "sid-c")
    me = psutil.Process()
    path = _write_raw_lease(
        root,
        "sid-c",
        parent_pid=DEAD_PID,
        parent_create_time=1.0,
        child_pid=me.pid,
        child_create_time=me.create_time(),
        session_root=str(session_root),
    )
    killed = []
    monkeypatch.setattr(
        leases_mod, "_kill_runtime_tree", lambda pid, ct: killed.append(pid) or True
    )
    leases_mod.reap_orphans(root)
    assert killed == [me.pid]
    assert not path.exists()
    assert not session_root.exists()


def test_reap_catch_all_kills_leaseless_runtime_orphan(tmp_path: Path, monkeypatch):
    root = _marked_root(tmp_path)
    orphan = _info(
        pid=4242,
        ppid=1,  # POSIX-reparented / dead parent
        exe=str(root / "abcd" / ".venv" / "python"),
        is_run_runtime=True,
    )
    bystander = _info(pid=4343, ppid=1, exe=str(root / "abcd" / ".venv" / "python"))
    monkeypatch.setattr(
        leases_mod,
        "take_process_snapshot",
        lambda: leases_mod.ProcessSnapshot(infos=(orphan, bystander)),
    )
    # The catch-all re-checks liveness against the (stale) snapshot.
    monkeypatch.setattr(leases_mod, "pid_alive", lambda pid, ct: pid == 4242)
    killed = []
    monkeypatch.setattr(
        leases_mod, "_kill_runtime_tree", lambda pid, ct: killed.append(pid) or True
    )
    leases_mod.reap_orphans(root)
    assert killed == [4242]  # the non-runtime bystander is never touched


def test_reap_catch_all_skips_pids_the_lease_pass_handled(tmp_path: Path, monkeypatch):
    """A child killed via its lease is not reported again as a leaseless orphan."""
    root = _marked_root(tmp_path)
    _write_raw_lease(
        root,
        "sid-o",
        parent_pid=DEAD_PID,
        parent_create_time=1.0,
        child_pid=4242,
        child_create_time=1.0,
    )
    orphan = _info(
        pid=4242,
        ppid=DEAD_PID,
        create_time=1.0,
        exe=str(root / "abcd" / ".venv" / "python"),
        is_run_runtime=True,
    )
    monkeypatch.setattr(
        leases_mod,
        "take_process_snapshot",
        lambda: leases_mod.ProcessSnapshot(infos=(orphan,)),
    )
    monkeypatch.setattr(leases_mod, "pid_alive", lambda pid, ct: pid == 4242)
    killed = []
    monkeypatch.setattr(
        leases_mod, "_kill_runtime_tree", lambda pid, ct: killed.append(pid) or True
    )
    leases_mod.reap_orphans(root)
    assert killed == [4242]  # once, via the lease; the catch-all skipped it


def test_reap_catch_all_ignores_processes_gone_since_snapshot(
    tmp_path: Path, monkeypatch
):
    root = _marked_root(tmp_path)
    ghost = _info(
        pid=4242,
        ppid=1,
        exe=str(root / "abcd" / ".venv" / "python"),
        is_run_runtime=True,
    )
    monkeypatch.setattr(
        leases_mod,
        "take_process_snapshot",
        lambda: leases_mod.ProcessSnapshot(infos=(ghost,)),
    )
    killed = []
    monkeypatch.setattr(
        leases_mod, "_kill_runtime_tree", lambda pid, ct: killed.append(pid) or True
    )
    leases_mod.reap_orphans(root)  # real pid_alive: 4242 does not exist
    assert killed == []


def test_reap_catch_all_skips_live_parent(tmp_path: Path, monkeypatch):
    root = _marked_root(tmp_path)
    me = psutil.Process()
    child = _info(
        pid=4242,
        ppid=me.pid,  # this test process is the live parent
        create_time=time.time(),
        exe=str(root / "abcd" / ".venv" / "python"),
        is_run_runtime=True,
    )
    monkeypatch.setattr(
        leases_mod,
        "take_process_snapshot",
        lambda: leases_mod.ProcessSnapshot(infos=(child,)),
    )
    killed = []
    monkeypatch.setattr(
        leases_mod, "_kill_runtime_tree", lambda pid, ct: killed.append(pid) or True
    )
    leases_mod.reap_orphans(root)
    assert killed == []


# ---------------------------------------------------------------------------
# Lease-less scratch sweep
# ---------------------------------------------------------------------------


def _aged(path: Path, seconds: float) -> None:
    """Backdate a directory's mtime by ``seconds``."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_sweep_removes_only_old_unreferenced_scratch(tmp_path: Path, monkeypatch):
    root = _marked_root(tmp_path)
    scratch_root = tmp_path / leases_mod.SCRATCH_ROOT_NAME
    old = scratch_root / "old-session"
    fresh = scratch_root / "fresh-session"
    old.mkdir(parents=True)
    fresh.mkdir(parents=True)
    _aged(old, leases_mod._LEASELESS_SCRATCH_AGE_SECONDS + 3600)

    monkeypatch.setattr(leases_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    leases_mod.reap_orphans(root)
    assert not old.exists()
    assert fresh.exists()


def test_sweep_keeps_dir_with_live_cwd(tmp_path: Path, monkeypatch):
    root = _marked_root(tmp_path)
    scratch_root = tmp_path / leases_mod.SCRATCH_ROOT_NAME
    busy = scratch_root / "busy-session"
    busy.mkdir(parents=True)
    _aged(busy, leases_mod._LEASELESS_SCRATCH_AGE_SECONDS + 3600)

    monkeypatch.setattr(leases_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        leases_mod,
        "take_process_snapshot",
        lambda: leases_mod.ProcessSnapshot(infos=(_info(cwd=str(busy / "work")),)),
    )
    leases_mod.reap_orphans(root)
    assert busy.exists()


def test_sweep_keeps_lease_referenced_scratch(tmp_path: Path, monkeypatch):
    root = _marked_root(tmp_path)
    scratch_root = tmp_path / leases_mod.SCRATCH_ROOT_NAME
    referenced = scratch_root / "referenced-session"
    referenced.mkdir(parents=True)
    _aged(referenced, leases_mod._LEASELESS_SCRATCH_AGE_SECONDS + 3600)
    # A LIVE server (this test process) holds a lease naming that root.
    _write_raw_lease(root, "sid-r", session_root=str(referenced))

    monkeypatch.setattr(leases_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    leases_mod.reap_orphans(root)
    assert referenced.exists()


def test_sweep_finds_roots_from_lease_paths_when_tempdir_differs(
    tmp_path: Path, monkeypatch
):
    """TEMP moved between contexts: lease session_roots still locate the tree."""
    root = _marked_root(tmp_path)
    other_temp = tmp_path / "other_temp" / leases_mod.SCRATCH_ROOT_NAME
    referenced = other_temp / "kept"
    stale = other_temp / "stale"
    referenced.mkdir(parents=True)
    stale.mkdir(parents=True)
    _aged(stale, leases_mod._LEASELESS_SCRATCH_AGE_SECONDS + 3600)
    _write_raw_lease(root, "sid-r", session_root=str(referenced))

    monkeypatch.setattr(
        leases_mod.tempfile, "gettempdir", lambda: str(tmp_path / "current_temp")
    )
    leases_mod.reap_orphans(root)
    assert referenced.exists()
    assert not stale.exists()
