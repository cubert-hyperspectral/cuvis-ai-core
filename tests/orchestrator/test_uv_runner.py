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


@pytest.mark.parametrize(
    "err",
    [
        FileNotFoundError(2, "No such file or directory"),
        PermissionError(13, "Permission denied"),
        OSError(8, "Exec format error"),
    ],
    ids=["vanished", "not_executable", "bad_binary"],
)
def test_prune_translates_popen_os_error_naming_the_binary(pinned_uv, monkeypatch, err):
    # Popen itself fails (binary vanished / bad CUVIS_UV target after resolution):
    # the whole OSError family, not just FileNotFoundError, must be translated.
    monkeypatch.setenv("PATH", "/nowhere")
    with patch(
        "cuvis_ai_core.orchestrator.uv_runner.subprocess.Popen", side_effect=err
    ):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_cache_prune(busy_grace=0.01)
    msg = str(excinfo.value)
    assert not isinstance(excinfo.value, UvCacheBusyError)
    assert f"'{pinned_uv}' was not found or is not executable" in msg
    assert f"Command: {pinned_uv} cache prune" in msg
    assert "PATH=/nowhere" in msg
    assert excinfo.value.__cause__ is err


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


# ---------------------------------------------------------------------------
# Timeout ceiling: CUVIS_UV_TIMEOUT_SECONDS, default 1500 s per invocation
# ---------------------------------------------------------------------------


def test_uv_lock_and_sync_default_timeout_is_1500(pinned_uv, monkeypatch):
    monkeypatch.delenv("CUVIS_UV_TIMEOUT_SECONDS", raising=False)
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_lock(Path("proj"))
        uv_sync(Path("proj"))
    assert [c.kwargs["timeout"] for c in run.call_args_list] == [1500, 1500]


def test_uv_timeout_read_from_env_at_call_time(pinned_uv, monkeypatch):
    monkeypatch.setenv("CUVIS_UV_TIMEOUT_SECONDS", "42.5")
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_lock(Path("proj"))
        uv_sync(Path("proj"))
    assert [c.kwargs["timeout"] for c in run.call_args_list] == [42.5, 42.5]


def test_uv_timeout_env_zero_falls_back_to_default(pinned_uv, monkeypatch):
    monkeypatch.setenv("CUVIS_UV_TIMEOUT_SECONDS", "0")
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_sync(Path("proj"))
    assert run.call_args.kwargs["timeout"] == 1500


def test_explicit_timeout_beats_the_env_knob(pinned_uv, monkeypatch):
    monkeypatch.setenv("CUVIS_UV_TIMEOUT_SECONDS", "42.5")
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run") as run:
        uv_sync(Path("proj"), timeout=7)
    assert run.call_args.kwargs["timeout"] == 7


def test_uv_sync_logs_elapsed_seconds(pinned_uv):
    from loguru import logger

    messages: list[str] = []
    handler = logger.add(
        lambda msg: messages.append(msg.record["message"]), level="INFO"
    )
    try:
        with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run"):
            uv_sync(Path("proj"))
    finally:
        logger.remove(handler)
    assert any("uv sync" in m and "finished after" in m for m in messages)


def test_uv_timeout_message_names_the_knob(pinned_uv):
    err = subprocess.TimeoutExpired(cmd=["uv", "sync"], timeout=5)
    with patch("cuvis_ai_core.orchestrator.uv_runner.subprocess.run", side_effect=err):
        with pytest.raises(UvRunnerError) as excinfo:
            uv_sync(Path("proj"), timeout=5)
    assert "timed out after 5s" in str(excinfo.value)
    assert "CUVIS_UV_TIMEOUT_SECONDS" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Prune at a low I/O priority (the deleter runs it behind a cold child start)
# ---------------------------------------------------------------------------


def test_prune_low_priority_lowers_the_child_io_priority(pinned_uv, monkeypatch):
    from cuvis_ai_core.orchestrator import uv_runner as uv_runner_mod

    seen: list[int] = []
    monkeypatch.setattr(uv_runner_mod, "_lower_io_priority", seen.append)
    fake = _FakePrune(stderr="Removed 3 files\n")
    fake.pid = 4242
    with _patched_popen(fake):
        uv_cache_prune(busy_grace=0.01, low_priority=True)
    assert seen == [4242]


def test_prune_keeps_normal_priority_by_default(pinned_uv, monkeypatch):
    from cuvis_ai_core.orchestrator import uv_runner as uv_runner_mod

    seen: list[int] = []
    monkeypatch.setattr(uv_runner_mod, "_lower_io_priority", seen.append)
    fake = _FakePrune(stderr="Removed 3 files\n")
    fake.pid = 4242
    with _patched_popen(fake):
        uv_cache_prune(busy_grace=0.01)
    assert seen == []


def test_lower_io_priority_asks_psutil_for_the_lowest_class(monkeypatch):
    import psutil

    from cuvis_ai_core.orchestrator import uv_runner as uv_runner_mod

    calls: list[tuple] = []

    class _Proc:
        def __init__(self, pid):
            calls.append(("process", pid))

        def ionice(self, *args):
            calls.append(("ionice", args))

    monkeypatch.setattr(uv_runner_mod.psutil, "Process", _Proc)
    uv_runner_mod._lower_io_priority(7)
    lowest = getattr(
        psutil, "IOPRIO_LOW" if sys.platform == "win32" else "IOPRIO_CLASS_IDLE", None
    )
    if lowest is None:
        pytest.skip("psutil has no I/O priority class on this platform")
    assert calls == [("process", 7), ("ionice", (lowest,))]


def test_lower_io_priority_swallows_psutil_errors(monkeypatch):
    import psutil

    from cuvis_ai_core.orchestrator import uv_runner as uv_runner_mod

    class _Gone:
        def __init__(self, pid):
            raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(uv_runner_mod.psutil, "Process", _Gone)
    uv_runner_mod._lower_io_priority(7)
