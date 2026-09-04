"""Unit tests for the crash-log store (``crash_logs``).

A crashed child's stdout/stderr logs are its only postmortem evidence,
and every teardown path ``rmtree``-s the session tree that holds them.
These tests pin the preserve-before-delete helper: best-effort copies,
the marker file, the env-overridable store location, the newest-N bound
on the store, and idempotence per session id (the failing RPC and the
later teardown must not produce two directories for one crash).
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import pytest

from cuvis_ai_core.orchestrator import crash_logs
from cuvis_ai_core.orchestrator.crash_logs import (
    crash_dir_root,
    preserve_child_logs,
    preserve_session_logs,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _forget_preserved_sessions():
    """The preserved-per-session map is module state; start every test clean."""
    reset_for_tests()
    yield
    reset_for_tests()


# ---------------------------------------------------------------------------
# crash_dir_root
# ---------------------------------------------------------------------------


def test_crash_dir_root_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    assert crash_dir_root() == tmp_path / "crashes"


def test_crash_dir_root_defaults_under_cache_root(monkeypatch, tmp_path):
    monkeypatch.delenv("CUVIS_RUNTIME_CRASH_DIR", raising=False)
    monkeypatch.setenv("CUVIS_RUN_CACHE_DIR", str(tmp_path / "runs"))
    assert crash_dir_root() == tmp_path / "runs" / ".crash_logs"


# ---------------------------------------------------------------------------
# preserve_child_logs
# ---------------------------------------------------------------------------


def _write_logs(tmp_path):
    stdout_log = tmp_path / "child.stdout.log"
    stderr_log = tmp_path / "child.stderr.log"
    stdout_log.write_text("registered 152 nodes", encoding="utf-8")
    stderr_log.write_text("Fatal Python error: Aborted", encoding="utf-8")
    return stdout_log, stderr_log


def test_preserve_child_logs_copies_both_logs_and_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    stdout_log, stderr_log = _write_logs(tmp_path)

    dest = preserve_child_logs(
        (stdout_log, stderr_log),
        session_id="sess-1",
        exit_code=3221226505,
        endpoint="127.0.0.1:51973",
    )

    assert dest is not None and dest.is_dir()
    assert (dest / "child.stdout.log").read_text(encoding="utf-8") == (
        "registered 152 nodes"
    )
    assert (dest / "child.stderr.log").exists()
    marker = (dest / "crash_info.txt").read_text(encoding="utf-8")
    assert "sess-1" in marker
    assert "0xC0000409" in marker
    assert "127.0.0.1:51973" in marker


def test_preserve_child_logs_skips_missing_files(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    assert (
        preserve_child_logs(
            (None, tmp_path / "never_written.log"), session_id="s", exit_code=1
        )
        is None
    )
    # Nothing to preserve -> the store is not even created.
    assert not (tmp_path / "crashes").exists()


def test_preserve_child_logs_survives_unreadable_file(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    stdout_log, stderr_log = _write_logs(tmp_path)

    def _locked(src, dst, **kwargs):
        raise OSError("file is locked")

    monkeypatch.setattr(shutil, "copy2", _locked)
    # Every copy fails (the Windows lock hazard) -> None, but no raise.
    assert (
        preserve_child_logs((stdout_log, stderr_log), session_id="s", exit_code=9)
        is None
    )


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------


def test_prune_keeps_newest_dirs(monkeypatch, tmp_path):
    root = tmp_path / "crashes"
    root.mkdir()
    for i in range(crash_logs._MAX_CRASH_DIRS + 3):
        entry = root / f"20260801-00000{i}-sess"
        entry.mkdir()
        stamp = 1_000_000 + i
        os.utime(entry, (stamp, stamp))

    crash_logs._prune_crash_dirs(root)

    survivors = sorted(p.name for p in root.iterdir())
    assert len(survivors) == crash_logs._MAX_CRASH_DIRS
    # The oldest three (indices 0-2) are gone, the newest remain.
    assert all(not name.startswith("20260801-000000") for name in survivors)
    assert all(not name.startswith("20260801-000001-") for name in survivors)


def test_preserve_prunes_the_store(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    root = tmp_path / "crashes"
    root.mkdir()
    for i in range(crash_logs._MAX_CRASH_DIRS + 2):
        entry = root / f"old-{i}"
        entry.mkdir()
        stamp = 1_000_000 + i
        os.utime(entry, (stamp, stamp))
    stdout_log, stderr_log = _write_logs(tmp_path)

    dest = preserve_child_logs(
        (stdout_log, stderr_log), session_id="fresh", exit_code=1
    )

    assert dest is not None and dest.exists()
    assert len(list(root.iterdir())) == crash_logs._MAX_CRASH_DIRS


# ---------------------------------------------------------------------------
# preserve_session_logs (orphan reaper entry point)
# ---------------------------------------------------------------------------


def test_preserve_session_logs_globs_nested_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    session_root = tmp_path / "cuvis_runtime_sessions" / "sess-7"
    runtime = session_root / "scratch" / "runtime"
    runtime.mkdir(parents=True)
    _write_logs(runtime)
    (runtime / "unrelated.txt").write_text("not a child log", encoding="utf-8")

    dest = preserve_session_logs(session_root, session_id="sess-7")

    assert dest is not None and dest.is_dir()
    assert dest.name.endswith("-sess-7")
    assert (dest / "child.stdout.log").exists()
    assert (dest / "child.stderr.log").exists()
    assert not (dest / "unrelated.txt").exists()
    # No exit code is known for a reaped orphan.
    assert "exit_code: unknown" in (dest / "crash_info.txt").read_text(encoding="utf-8")


def test_preserve_session_logs_without_logs_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    empty_root = tmp_path / "cuvis_runtime_sessions" / "sess-empty"
    empty_root.mkdir(parents=True)

    assert preserve_session_logs(empty_root, session_id="sess-empty") is None
    assert preserve_session_logs(tmp_path / "missing", session_id="gone") is None
    assert not (tmp_path / "crashes").exists()


def test_preserve_session_logs_survives_unscannable_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    session_root = tmp_path / "cuvis_runtime_sessions" / "sess-locked"
    runtime = session_root / "scratch" / "runtime"
    runtime.mkdir(parents=True)
    _write_logs(runtime)

    def _denied(self, pattern, *args, **kwargs):
        # Like the real rglob, a lazy generator: the error surfaces while the
        # result is consumed, so the sorted() call must sit inside the try.
        yield from ()
        raise PermissionError(f"cannot scan {self}")

    monkeypatch.setattr(Path, "rglob", _denied)
    # The scan itself fails (unreadable tree) -> warn and return None, no raise.
    assert preserve_session_logs(session_root, session_id="sess-locked") is None
    # Logs exist on disk, but they were never reached -> store not created.
    assert not (tmp_path / "crashes").exists()


# ---------------------------------------------------------------------------
# Idempotence per session id
# ---------------------------------------------------------------------------


def test_preserve_child_logs_is_idempotent_per_session(monkeypatch, tmp_path):
    """The second call for one session returns the first directory, unchanged."""
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    stdout_log, stderr_log = _write_logs(tmp_path)

    first = preserve_child_logs(
        (stdout_log, stderr_log), session_id="sess-dup", exit_code=9
    )
    second = preserve_child_logs(
        (stdout_log, stderr_log), session_id="sess-dup", exit_code=9
    )

    assert first is not None
    assert second == first
    assert [p.name for p in (tmp_path / "crashes").iterdir()] == [first.name]


def test_preserve_child_logs_still_separates_distinct_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    stdout_log, stderr_log = _write_logs(tmp_path)

    one = preserve_child_logs((stdout_log, stderr_log), session_id="a", exit_code=1)
    two = preserve_child_logs((stdout_log, stderr_log), session_id="b", exit_code=1)

    assert one is not None and two is not None and one != two


def test_preserve_child_logs_two_threads_produce_one_directory(monkeypatch, tmp_path):
    """close_session and the failing RPC can race; the lock leaves one folder."""
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    stdout_log, stderr_log = _write_logs(tmp_path)
    start = threading.Barrier(2)
    results: list[Path | None] = [None, None]

    def _preserve(index: int) -> None:
        start.wait(timeout=5)
        results[index] = preserve_child_logs(
            (stdout_log, stderr_log), session_id="sess-race", exit_code=9
        )

    threads = [threading.Thread(target=_preserve, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert results[0] is not None
    assert results[0] == results[1]
    assert len(list((tmp_path / "crashes").iterdir())) == 1


def test_a_failed_preservation_is_not_remembered(monkeypatch, tmp_path):
    """Nothing was preserved, so a later attempt must be allowed to try again."""
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    stdout_log, stderr_log = _write_logs(tmp_path)

    def _locked(src, dst, **kwargs):
        raise OSError("file is locked")

    monkeypatch.setattr(shutil, "copy2", _locked)
    assert (
        preserve_child_logs(
            (stdout_log, stderr_log), session_id="sess-retry", exit_code=9
        )
        is None
    )

    monkeypatch.undo()
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    assert (
        preserve_child_logs(
            (stdout_log, stderr_log), session_id="sess-retry", exit_code=9
        )
        is not None
    )


def test_preserve_child_logs_ignores_non_path_entries(monkeypatch, tmp_path):
    """Log paths read off a child handle with getattr can be anything."""
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(tmp_path / "crashes"))
    _, stderr_log = _write_logs(tmp_path)

    dest = preserve_child_logs(
        (object(), stderr_log), session_id="sess-junk", exit_code=1
    )

    assert dest is not None
    assert [p.name for p in sorted(dest.iterdir())] == [
        "child.stderr.log",
        "crash_info.txt",
    ]
