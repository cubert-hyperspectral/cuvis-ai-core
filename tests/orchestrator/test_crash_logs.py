"""Unit tests for the crash-log store (``crash_logs``).

A crashed child's stdout/stderr logs are its only postmortem evidence,
and every teardown path ``rmtree``-s the session tree that holds them.
These tests pin the preserve-before-delete helper: best-effort copies,
the marker file, the env-overridable store location, and the newest-N
bound on the store.
"""

from __future__ import annotations

import os
import shutil

from cuvis_ai_core.orchestrator import crash_logs
from cuvis_ai_core.orchestrator.crash_logs import (
    crash_dir_root,
    preserve_child_logs,
)


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
