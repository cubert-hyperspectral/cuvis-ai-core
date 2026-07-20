"""uv_runner subprocess-error translation and executable-resolution tests.

The composer mocks ``uv_lock`` / ``uv_sync`` wholesale, so the real
``_run_uv`` body — argv construction, the executable resolution chain
(``CUVIS_UV`` → ``shutil.which`` → ``uv.find_uv_bin``), and the
``FileNotFoundError`` / ``CalledProcessError`` / ``TimeoutExpired`` →
``UvRunnerError`` translation — is exercised only here.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from cuvis_ai_core.orchestrator.uv_runner import UvRunnerError, uv_lock, uv_sync


@pytest.fixture()
def pinned_uv(monkeypatch):
    """Pin resolution to a known binary so argv assertions are deterministic."""
    monkeypatch.setenv("CUVIS_UV", "/pinned/uv")
    return "/pinned/uv"


def test_uv_lock_builds_expected_argv(pinned_uv):
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_lock(Path("proj"))
    assert run.call_args.args[0] == [pinned_uv, "lock", "--project", str(Path("proj"))]
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["capture_output"] is True


def test_uv_sync_builds_expected_argv(pinned_uv):
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_sync(Path("proj"))
    assert run.call_args.args[0] == [pinned_uv, "sync", "--project", str(Path("proj"))]


def test_cuvis_uv_override_beats_path_lookup(monkeypatch):
    monkeypatch.setenv("CUVIS_UV", "/override/uv")
    with (
        patch(
            "cuvis_ai_core.orchestrator.uv_runner.shutil.which",
            return_value="/path/uv",
        ),
        patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run,
    ):
        uv_lock(Path("proj"))
    assert run.call_args.args[0][0] == "/override/uv"


def test_path_lookup_used_without_override(monkeypatch):
    monkeypatch.delenv("CUVIS_UV", raising=False)
    with (
        patch(
            "cuvis_ai_core.orchestrator.uv_runner.shutil.which",
            return_value="/usr/local/bin/uv",
        ),
        patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run,
    ):
        uv_sync(Path("proj"))
    assert run.call_args.args[0][0] == "/usr/local/bin/uv"


def test_uv_wheel_locator_is_last_resort(monkeypatch):
    monkeypatch.delenv("CUVIS_UV", raising=False)
    fake_uv = types.ModuleType("uv")
    fake_uv.find_uv_bin = lambda: "/venv/bin/uv"
    monkeypatch.setitem(sys.modules, "uv", fake_uv)
    with (
        patch("cuvis_ai_core.orchestrator.uv_runner.shutil.which", return_value=None),
        patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run,
    ):
        uv_lock(Path("proj"))
    assert run.call_args.args[0][0] == "/venv/bin/uv"


def test_unresolvable_uv_raises_with_tool_name_and_path(monkeypatch):
    monkeypatch.delenv("CUVIS_UV", raising=False)
    monkeypatch.setenv("PATH", "/nowhere")
    monkeypatch.delitem(sys.modules, "uv", raising=False)
    with (
        patch("cuvis_ai_core.orchestrator.uv_runner.shutil.which", return_value=None),
        patch.dict(sys.modules, {"uv": None}),
    ):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_lock(Path("proj"))
    msg = str(excinfo.value)
    assert "'uv' was not found" in msg
    assert "CUVIS_UV" in msg
    assert "PATH=/nowhere" in msg


def test_stale_override_translates_file_not_found(monkeypatch):
    # Resolution succeeds (CUVIS_UV set) but the binary is gone by exec time.
    monkeypatch.setenv("CUVIS_UV", "/stale/uv")
    err = FileNotFoundError(2, "No such file or directory")
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_lock(Path("proj"))
    msg = str(excinfo.value)
    assert "'/stale/uv' was not found or is not executable" in msg
    assert "PATH=" in msg


def test_uv_lock_translates_called_process_error_with_stderr(pinned_uv):
    err = subprocess.CalledProcessError(returncode=2, cmd=["uv", "lock"], stderr="boom")
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_lock(Path("proj"))
    msg = str(excinfo.value)
    assert "exit 2" in msg
    assert "boom" in msg


def test_uv_sync_translates_called_process_error_with_empty_stderr(pinned_uv):
    err = subprocess.CalledProcessError(returncode=1, cmd=["uv", "sync"], stderr=None)
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_sync(Path("proj"))
    assert "<empty>" in str(excinfo.value)


def test_uv_lock_translates_timeout(pinned_uv):
    err = subprocess.TimeoutExpired(cmd=["uv", "lock"], timeout=5)
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_lock(Path("proj"), timeout=5)
    assert "timed out after 5s" in str(excinfo.value)
