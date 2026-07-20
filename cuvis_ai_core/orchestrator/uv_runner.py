"""Thin subprocess wrapper around ``uv lock`` and ``uv sync``.

A single shim so tests can mock both invocations and the composer has
one place to surface errors.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger

_DEFAULT_TIMEOUT = 600  # uv lock + sync of HF + torch can take minutes


class UvRunnerError(RuntimeError):
    """Raised when a uv subprocess fails or no uv executable can be found."""


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


def _run_uv(cmd: list[str], *, timeout: int) -> None:
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
