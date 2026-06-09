"""uv_runner subprocess-error translation tests.

The composer mocks ``uv_lock`` / ``uv_sync`` wholesale, so the real
``_run_uv`` body — argv construction and the ``CalledProcessError`` /
``TimeoutExpired`` → ``UvRunnerError`` translation — is exercised only
here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cuvis_ai_core.orchestrator.uv_runner import UvRunnerError, uv_lock, uv_sync


def test_uv_lock_builds_expected_argv():
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_lock(Path("proj"))
    assert run.call_args.args[0] == ["uv", "lock", "--project", str(Path("proj"))]
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["capture_output"] is True


def test_uv_sync_builds_expected_argv():
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_sync(Path("proj"))
    assert run.call_args.args[0] == ["uv", "sync", "--project", str(Path("proj"))]


def test_uv_lock_translates_called_process_error_with_stderr():
    err = subprocess.CalledProcessError(returncode=2, cmd=["uv", "lock"], stderr="boom")
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_lock(Path("proj"))
    msg = str(excinfo.value)
    assert "exit 2" in msg
    assert "boom" in msg


def test_uv_sync_translates_called_process_error_with_empty_stderr():
    err = subprocess.CalledProcessError(returncode=1, cmd=["uv", "sync"], stderr=None)
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_sync(Path("proj"))
    assert "<empty>" in str(excinfo.value)


def test_uv_lock_translates_timeout():
    err = subprocess.TimeoutExpired(cmd=["uv", "lock"], timeout=5)
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_lock(Path("proj"), timeout=5)
    assert "timed out after 5s" in str(excinfo.value)
