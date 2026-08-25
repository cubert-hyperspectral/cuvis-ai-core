"""Thin subprocess wrapper around ``uv lock`` and ``uv sync``.

A single shim so tests can mock both invocations and the composer has
one place to surface errors.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from loguru import logger

_DEFAULT_TIMEOUT = 600  # uv lock + sync of HF + torch can take minutes
# uv announces a held cache lock on stderr the moment it starts waiting and
# prints nothing more until it acquires the lock. A prune whose most recent
# stderr line is still that announcement after this many seconds is abandoned
# instead of burning the full timeout.
_PRUNE_BUSY_GRACE_SECONDS = 15.0
# Wording verified against uv 0.8.24 ("Cache is currently in-use, waiting for
# other uv processes to finish (use `--force` to override)").
_UV_CACHE_BUSY_MARKER = "waiting for other uv processes"


class UvRunnerError(RuntimeError):
    """Raised when a uv subprocess fails or no uv executable can be found."""


class UvCacheBusyError(UvRunnerError):
    """``uv cache prune`` gave up because another uv process holds the cache lock."""


def _uv_executable() -> str:
    """Resolve the uv executable for compose subprocesses.

    Order: explicit ``CUVIS_UV`` override (a host app pins the exact
    binary — the cleanest cross-process contract), then ``shutil.which``
    over the inherited PATH, then the ``uv`` wheel's own locator when
    that package happens to be installed. uv installs to the *per-user*
    PATH on Windows, so whether a bare ``"uv"`` resolves depends on the
    host's launch context; failing here, with the tool named and the
    searched PATH shown, replaces an opaque ``[WinError 2]`` deep inside
    ``subprocess.run``.
    """
    override = os.environ.get("CUVIS_UV")
    if override:
        return override
    found = shutil.which("uv")
    if found:
        return found
    try:
        from uv import find_uv_bin  # present only when the uv wheel is installed

        return find_uv_bin()
    except (ImportError, FileNotFoundError):
        pass
    raise UvRunnerError(
        "'uv' was not found. Set CUVIS_UV to the uv executable or ensure uv is "
        f"on the server process PATH.\nPATH={os.environ.get('PATH', '')}"
    )


def uv_lock(project_dir: Path, *, timeout: int = _DEFAULT_TIMEOUT) -> None:
    """Run ``uv lock --project <project_dir>``.

    Writes ``uv.lock`` next to the runtime ``pyproject.toml``. A
    repeat invocation on the same project is a no-op once the lock
    exists.
    """
    _run_uv([_uv_executable(), "lock", "--project", str(project_dir)], timeout=timeout)


def uv_sync(project_dir: Path, *, timeout: int = _DEFAULT_TIMEOUT) -> None:
    """Run ``uv sync --project <project_dir>``.

    Materialises ``<project_dir>/.venv`` against the lockfile uv
    produced in :func:`uv_lock`.
    """
    _run_uv([_uv_executable(), "sync", "--project", str(project_dir)], timeout=timeout)


def uv_cache_prune(
    *, timeout: int = _DEFAULT_TIMEOUT, busy_grace: float = _PRUNE_BUSY_GRACE_SECONDS
) -> None:
    """Run ``uv cache prune`` to drop unreferenced blobs from uv's own cache.

    Under uv's default hardlink mode an evicted venv frees little disk
    until uv's cache releases its surviving links, so the eviction
    deleter calls this once per drained batch — strictly AFTER the
    rmtrees complete (pruning earlier would find the blobs still linked
    and keep them).

    uv blocks on its global cache lock for as long as any other uv
    process holds it, and a long-running ``uvx`` tool holds it
    indefinitely. uv announces the wait on stderr the moment it starts
    and prints "Pruning cache at: ..." once it acquires the lock, so a
    prune whose most recent stderr line is still the wait announcement
    after ``busy_grace`` seconds is still blocked: it is killed and
    reported as :class:`UvCacheBusyError` (a subclass of
    :class:`UvRunnerError`) instead of stalling until ``timeout``. A
    prune that acquired the lock mid-grace keeps running until
    ``timeout``. Callers treat both errors as non-fatal.
    """
    cmd = [_uv_executable(), "cache", "prune"]
    logger.debug(f"uv invocation: {' '.join(cmd)}")
    deadline = time.monotonic() + timeout
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # uv writes UTF-8 (cache paths included); the locale codec —
            # cp1252 on Windows — cannot decode it and would crash the
            # pump thread on the first non-ASCII path.
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        # FileNotFoundError for a vanished binary, PermissionError /
        # WinError 193 / Exec format error for a bad CUVIS_UV target.
        raise UvRunnerError(
            f"'{cmd[0]}' was not found or is not executable.\n"
            f"Command: {' '.join(cmd)}\n"
            f"PATH={os.environ.get('PATH', '')}"
        ) from exc

    stderr_lines: list[str] = []
    pump = threading.Thread(
        target=_pump_lines, args=(proc.stderr, stderr_lines), daemon=True
    )
    pump.start()
    try:
        proc.wait(timeout=min(busy_grace, timeout))
    except subprocess.TimeoutExpired:
        if stderr_lines and _UV_CACHE_BUSY_MARKER in stderr_lines[-1]:
            proc.kill()
            proc.wait()
            raise UvCacheBusyError(
                "uv cache is in use by another uv process; prune skipped. "
                f"Command: {' '.join(cmd)}"
            ) from None
        try:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.1))
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            raise UvRunnerError(
                f"uv timed out after {timeout}s. Command: {' '.join(cmd)}"
            ) from exc
    pump.join(timeout=5.0)
    if proc.returncode != 0:
        stderr = "".join(stderr_lines).strip()
        raise UvRunnerError(
            f"uv failed (exit {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {stderr or '<empty>'}"
        )


def _pump_lines(stream, sink: list[str]) -> None:
    """Drain ``stream`` into ``sink`` so the child never blocks on a full pipe."""
    with stream:
        for line in stream:
            sink.append(line)


def _run_uv(cmd: list[str], *, timeout: int) -> None:
    """Execute one uv command, mapping every failure to ``UvRunnerError``."""
    logger.debug(f"uv invocation: {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        # Belt and braces: _uv_executable() resolves before we get here, but a
        # stale CUVIS_UV / deleted binary still lands in this branch — name the
        # missing tool instead of leaking a bare [WinError 2].
        raise UvRunnerError(
            f"'{cmd[0]}' was not found or is not executable.\n"
            f"Command: {' '.join(cmd)}\n"
            f"PATH={os.environ.get('PATH', '')}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UvRunnerError(
            f"uv timed out after {timeout}s. Command: {' '.join(cmd)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise UvRunnerError(
            f"uv failed (exit {exc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {exc.stderr.strip() if exc.stderr else '<empty>'}"
        ) from exc
