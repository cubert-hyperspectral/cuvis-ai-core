"""Thin subprocess wrapper around ``uv lock`` and ``uv sync``.

A single shim so tests can mock both invocations and the composer
has one place to surface errors. Mirrors the timeout + check +
loguru pattern in
``cuvis_ai_core/utils/git_and_os.py:_install_dependencies_with_uv``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

_DEFAULT_TIMEOUT = 600  # uv lock + sync of HF + torch can take minutes


class UvRunnerError(RuntimeError):
    """Raised when a uv subprocess returns a non-zero exit code."""


def uv_lock(project_dir: Path, *, timeout: int = _DEFAULT_TIMEOUT) -> None:
    """Run ``uv lock --project <project_dir>``.

    Writes ``uv.lock`` next to the runtime ``pyproject.toml``. A
    repeat invocation on the same project is a no-op once the lock
    exists.
    """
    _run_uv(["uv", "lock", "--project", str(project_dir)], timeout=timeout)


def uv_sync(project_dir: Path, *, timeout: int = _DEFAULT_TIMEOUT) -> None:
    """Run ``uv sync --project <project_dir>``.

    Materialises ``<project_dir>/.venv`` against the lockfile uv
    produced in :func:`uv_lock`.
    """
    _run_uv(["uv", "sync", "--project", str(project_dir)], timeout=timeout)


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
