"""Preserve child-runtime logs before their session tree is deleted.

A child that exits abnormally leaves its only postmortem evidence in
``child.stdout.log`` / ``child.stderr.log`` under the session scratch
tree — and every teardown path ends in ``rmtree`` of that tree. This
module copies those logs aside first, into ``.crash_logs/`` under the
composer cache root (dot-prefixed, never a cache entry), so a crash can
still be diagnosed after cleanup.

The composer cache-root lookup stays a lazy import inside
:func:`crash_dir_root`, so importing this module never pays the
composer's (and its transitive) import cost.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from cuvis_ai_core.orchestrator.spawner import format_exit_code

# Operator override for where preserved logs land. When unset they live
# under the composer's cache root so an operator finds them next to the
# composed envs the crashed child ran from.
_CRASH_DIR_ENV = "CUVIS_RUNTIME_CRASH_DIR"
CRASH_LOGS_DIRNAME = ".crash_logs"

_MARKER_NAME = "crash_info.txt"
_MAX_CRASH_DIRS = 5


def crash_dir_root() -> Path:
    """Resolve where preserved child logs are stored.

    ``$CUVIS_RUNTIME_CRASH_DIR`` when set, else
    ``<composer cache root>/.crash_logs``.
    """
    override = os.environ.get(_CRASH_DIR_ENV)
    if override:
        return Path(override)
    # Lazy import mirrors model_cache: light consumers never pay the
    # composer's (and its transitive) import cost.
    from cuvis_ai_core.orchestrator.composer import resolve_cache_root

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
    preserved. Best-effort by design: session teardown must never fail
    because a log file is missing, still locked (Windows), or the store
    is unwritable.
    """
    try:
        files = [p for p in log_paths if p is not None and p.exists()]
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
