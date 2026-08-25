"""uv_runner subprocess-error translation and executable-resolution tests.

The composer mocks ``uv_lock`` / ``uv_sync`` wholesale, so the real
``_run_uv`` body — argv construction, the executable resolution chain
(``CUVIS_UV`` → ``shutil.which`` → ``uv.find_uv_bin``), and the
``FileNotFoundError`` / ``CalledProcessError`` / ``TimeoutExpired`` →
``UvRunnerError`` translation — is exercised only here.
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from cuvis_ai_core.orchestrator.uv_runner import (
    UvCacheBusyError,
    UvRunnerError,
    uv_cache_prune,
    uv_lock,
    uv_sync,
)


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


# ---------------------------------------------------------------------------
# uv cache prune — busy-lock detection on top of the plain timeout
# ---------------------------------------------------------------------------


class _FakePrune:
    """Stand-in for ``subprocess.Popen`` running ``uv cache prune``.

    ``stderr`` is what uv would print; ``blocked_waits`` is how many
    ``wait()`` calls time out before the process exits with ``returncode``.
    """

    def __init__(self, stderr: str = "", blocked_waits: int = 0, returncode: int = 0):
        self.stderr = io.StringIO(stderr)
        self._blocked_waits = blocked_waits
        self._exit_code = returncode
        self.returncode: int | None = None
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.killed:
            self.returncode = -9
            return self.returncode
        if self._blocked_waits > 0:
            self._blocked_waits -= 1
            # Give the stderr pump thread a beat, as a real wait() would;
            # the busy check right after this reads what the pump collected.
            time.sleep(0.05)
            raise subprocess.TimeoutExpired(
                cmd=["uv", "cache", "prune"], timeout=timeout
            )
        self.returncode = self._exit_code
        return self.returncode

    def kill(self):
        self.killed = True


def _patched_popen(fake: _FakePrune):
    return patch(
        "cuvis_ai_core.orchestrator.uv_runner.subprocess.Popen", return_value=fake
    )


def test_prune_completes_quietly(pinned_uv):
    fake = _FakePrune(stderr="Removed 3 files\n")
    with _patched_popen(fake):
        uv_cache_prune(busy_grace=0.01)
    assert not fake.killed


def test_prune_gives_up_when_cache_lock_is_held(pinned_uv):
    """uv's 'waiting for other uv processes' line ends the prune after the grace."""
    fake = _FakePrune(
        stderr=(
            "Cache is currently in-use, waiting for other uv processes to finish "
            "(use `--force` to override)\n"
        ),
        blocked_waits=99,
    )
    with _patched_popen(fake):
        with pytest.raises(UvCacheBusyError) as excinfo:
            uv_cache_prune(timeout=600, busy_grace=0.01)
    assert fake.killed
    assert "in use by another uv process" in str(excinfo.value)


def test_prune_slow_but_working_is_not_treated_as_busy(pinned_uv):
    """No busy line on stderr: a long prune simply keeps running to completion."""
    fake = _FakePrune(stderr="Pruning...\n", blocked_waits=1)
    with _patched_popen(fake):
        uv_cache_prune(timeout=600, busy_grace=0.01)
    assert not fake.killed
    assert fake.returncode == 0


def test_prune_translates_timeout(pinned_uv):
    fake = _FakePrune(stderr="Pruning...\n", blocked_waits=99)
    with _patched_popen(fake):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_cache_prune(timeout=5, busy_grace=0.01)
    assert not isinstance(excinfo.value, UvCacheBusyError)
    assert "timed out after 5s" in str(excinfo.value)
    assert fake.killed


def test_prune_translates_nonzero_exit_with_stderr(pinned_uv):
    fake = _FakePrune(stderr="error: cache corrupt\n", returncode=2)
    with _patched_popen(fake):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_cache_prune(busy_grace=0.01)
    msg = str(excinfo.value)
    assert "exit 2" in msg
    assert "cache corrupt" in msg


def test_prune_that_acquired_the_lock_mid_grace_is_not_killed(pinned_uv):
    """The busy line alone is history: once 'Pruning cache at' follows, let it run."""
    fake = _FakePrune(
        stderr=(
            "Cache is currently in-use, waiting for other uv processes to finish "
            "(use `--force` to override)\n"
            "Pruning cache at: C:\\Users\\x\\uv\\cache\n"
        ),
        blocked_waits=1,
    )
    with _patched_popen(fake):
        uv_cache_prune(timeout=600, busy_grace=0.01)
    assert not fake.killed
    assert fake.returncode == 0


def test_prune_timeout_smaller_than_grace_is_honored(pinned_uv):
    """timeout is the overall budget even when it undercuts the busy grace."""
    fake = _FakePrune(stderr="Pruning...\n", blocked_waits=99)
    with _patched_popen(fake):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_cache_prune(timeout=2, busy_grace=600.0)
    assert "timed out after 2s" in str(excinfo.value)
    assert fake.wait_timeouts[0] == 2  # first wait clamped to the budget


def test_prune_decodes_stderr_as_utf8(pinned_uv):
    """uv writes UTF-8; the locale codec (cp1252 on Windows) must not be used."""
    fake = _FakePrune(stderr="ok\n")
    with _patched_popen(fake) as popen:
        uv_cache_prune(busy_grace=0.01)
    assert popen.call_args.kwargs["encoding"] == "utf-8"
    assert popen.call_args.kwargs["errors"] == "replace"
