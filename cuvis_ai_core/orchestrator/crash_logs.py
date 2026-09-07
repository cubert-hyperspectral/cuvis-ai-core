"""Preserve child-runtime logs before their session tree is deleted.

A child that exits abnormally leaves its only postmortem evidence in
``child.stdout.log`` / ``child.stderr.log`` under the session scratch
tree — and every teardown path ends in ``rmtree`` of that tree. This
module copies those logs aside first, into ``.crash_logs/`` under the
composer cache root (dot-prefixed, never a cache entry), so a crash can
still be diagnosed after cleanup.

Preservation is idempotent per session id: the failing RPC preserves the
logs at the moment of death so the client learns where they are, and the
later ``close_session`` teardown asks again and is handed the same
directory instead of copying a second one.

The cache root comes from the stdlib-only ``cache_paths`` leaf, so importing
this module never pays the composer's (and its transitive) import cost.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from cuvis_ai_core.orchestrator.cache_paths import resolve_cache_root
from cuvis_ai_core.orchestrator.spawner import format_exit_code

# Operator override for where preserved logs land. When unset they live
# under the composer's cache root so an operator finds them next to the
# composed envs the crashed child ran from.
_CRASH_DIR_ENV = "CUVIS_RUNTIME_CRASH_DIR"
CRASH_LOGS_DIRNAME = ".crash_logs"

_MARKER_NAME = "crash_info.txt"
_MAX_CRASH_DIRS = 5

# One preserved directory per session id. Two callers race for a crashed
# session (the failing RPC and the later close_session teardown, plus the
# orphan reaper), and each would otherwise create its own timestamped copy.
# The lock is held across the copy so the loser of the race waits and then
# sees the winner's directory. Only successful preservations are recorded,
# so a failed attempt can still be retried. One Path per crashed session
# lives here for the life of the process.
_preserve_lock = threading.Lock()
_preserved: dict[str, Path] = {}


def reset_for_tests() -> None:
    """Forget which sessions have preserved logs (test isolation only)."""
    with _preserve_lock:
        _preserved.clear()


def crash_dir_root() -> Path:
    """Resolve where preserved child logs are stored.

    ``$CUVIS_RUNTIME_CRASH_DIR`` when set, else
    ``<composer cache root>/.crash_logs``.
    """
    override = os.environ.get(_CRASH_DIR_ENV)
    if override:
        return Path(override)
    return resolve_cache_root(None) / CRASH_LOGS_DIRNAME


def preserve_child_logs(
    log_paths: Iterable[Path | None],
    *,
    session_id: str,
    exit_code: int | None = None,
    endpoint: str | None = None,
) -> Path | None:
    """Copy a dead child's log files into the crash-log store.

    Returns the destination directory, or ``None`` when nothing could be
    preserved. Idempotent per ``session_id``: a repeat call for a session
    whose logs are already preserved returns the recorded directory
    without copying anything a second time. Best-effort by design:
    session teardown must never fail because a log file is missing, still
    locked (Windows), or the store is unwritable.
    """
    with _preserve_lock:
        recorded = _preserved.get(session_id)
        if recorded is not None:
            return recorded
        destination = _copy_child_logs(
            log_paths, session_id=session_id, exit_code=exit_code, endpoint=endpoint
        )
        if destination is not None:
            _preserved[session_id] = destination
        return destination


def _copy_child_logs(
    log_paths: Iterable[Path | None],
    *,
    session_id: str,
    exit_code: int | None,
    endpoint: str | None,
) -> Path | None:
    """Do the actual copy for :func:`preserve_child_logs` (called under the lock)."""
    try:
        # Only real path-likes: a caller reading log paths off a child handle
        # with getattr can hand us anything, and a stray object must not turn
        # into a half-populated crash directory.
        candidates = [Path(p) for p in log_paths if isinstance(p, (str, Path))]
        files = [p for p in candidates if p.exists()]
        if not files:
            return None
        destination = (
            crash_dir_root() / f"{time.strftime('%Y%m%d-%H%M%S')}-{session_id}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for log_file in files:
            try:
                shutil.copy2(log_file, destination / log_file.name)
                copied += 1
            except OSError as exc:
                logger.warning(f"Could not preserve {log_file}: {exc}")
        code_text = format_exit_code(exit_code) if exit_code is not None else "unknown"
        marker = (
            f"session_id: {session_id}\n"
            f"exit_code: {code_text}\n"
            f"endpoint: {endpoint or 'unknown'}\n"
            f"preserved_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )
        (destination / _MARKER_NAME).write_text(marker, encoding="utf-8")
        _prune_crash_dirs(destination.parent)
        return destination if copied else None
    except Exception as exc:
        logger.warning(f"Could not preserve child logs for {session_id}: {exc}")
        return None


def _prune_crash_dirs(root: Path) -> None:
    """Bound the crash-log store to the newest ``_MAX_CRASH_DIRS`` entries."""
    try:
        entries = sorted(
            (p for p in root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return
    for stale in entries[:-_MAX_CRASH_DIRS]:
        shutil.rmtree(stale, ignore_errors=True)


# The spawner writes the child's streams as ``child.stdout.log`` /
# ``child.stderr.log`` under the session scratch tree. The orphan reaper
# knows only that tree (from the lease), not the individual log paths.
_LOG_GLOB = "child.*.log"


def preserve_session_logs(session_root: Path, *, session_id: str) -> Path | None:
    """Preserve every child log found under a session's scratch tree.

    The orphan reaper's counterpart to :func:`preserve_child_logs`: it
    works from a lease, which records the session root but not the log
    paths, nor an exit code (the parent that spawned the child is gone,
    so the marker records ``unknown``). Best-effort like its sibling;
    returns the destination, or ``None`` when nothing was preserved.
    """
    try:
        log_files = sorted(session_root.rglob(_LOG_GLOB))
    except OSError as exc:
        logger.warning(f"Could not scan {session_root} for child logs: {exc}")
        return None
    if not log_files:
        return None
    return preserve_child_logs(log_files, session_id=session_id)
