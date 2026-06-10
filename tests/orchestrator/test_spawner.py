"""Spawner unit tests.

The full happy-path tests boot a real child runtime via
``python -m cuvis_ai_core.run_runtime``; they only need the parent's
own python (the child's venv is mocked by reusing the same
interpreter). Tests run in seconds because the runtime does no work
until ``InitializeSession`` is called.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import grpc
import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

from cuvis_ai_core.orchestrator.spawner import (
    ChildHandle,
    DeclaredPaths,
    LocalChildRuntimeSpawner,
    SpawnError,
    _prepend_path,
    _read_stderr_log,
    _timeout_from_env,
)


@pytest.fixture(autouse=True)
def _use_real_spawner():
    """Override the suite-wide in-memory orchestrator: this file tests the real spawner."""
    from cuvis_ai_core.grpc import orchestrator_bridge

    orchestrator_bridge.reset_orchestrator()
    yield
    orchestrator_bridge.install_in_memory_orchestrator()


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


def test_prepend_path_empty_pathenv_returns_new_entry_only():
    assert _prepend_path("", "/new") == "/new"


def test_prepend_path_prepends_with_os_sep(monkeypatch):
    out = _prepend_path("/old1" + os.pathsep + "/old2", "/new")
    parts = out.split(os.pathsep)
    assert parts[0] == "/new"
    assert "/old1" in parts and "/old2" in parts


def test_declared_paths_is_immutable():
    p = DeclaredPaths(output_dir=Path("/o"), scratch_dir=Path("/s"))
    with pytest.raises(Exception):
        p.output_dir = Path("/x")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Full subprocess spawn (uses parent's python; the child venv is mocked
# by symlinking / fake-tree because real compose_env is unit-tested
# separately).
# ---------------------------------------------------------------------------


def _find_active_venv() -> Path:
    """Return the venv that owns ``sys.executable``.

    ``compose_env`` would normally produce this path, but for fast
    unit tests we use the parent's own venv: it has ``cuvis_ai_core``
    editable-installed so the child can ``import cuvis_ai_core.run_runtime``
    without any composer / uv overhead. The spawner is given the real
    venv directory, hits ``venv_python`` for the launch path, and
    boots a normal child process.
    """
    if sys.platform == "win32":
        # Scripts/python.exe → parent dir Scripts → venv root
        return Path(sys.executable).parent.parent
    # bin/python → parent dir bin → venv root
    return Path(sys.executable).parent.parent


@pytest.fixture
def fake_venv() -> Path:
    return _find_active_venv()


@pytest.fixture
def declared_paths(tmp_path: Path) -> DeclaredPaths:
    out = tmp_path / "out"
    scratch = tmp_path / "scratch"
    out.mkdir()
    scratch.mkdir()
    return DeclaredPaths(output_dir=out, scratch_dir=scratch)


def test_spawn_rejects_missing_venv_python(tmp_path: Path, declared_paths):
    spawner = LocalChildRuntimeSpawner()
    with pytest.raises(SpawnError, match="no python"):
        spawner.spawn(
            tmp_path / "does_not_exist",
            cwd=declared_paths.output_dir,
            declared_paths=declared_paths,
        )


def test_spawn_boots_child_and_health_checks_ok(fake_venv: Path, declared_paths):
    spawner = LocalChildRuntimeSpawner()
    handle = spawner.spawn(
        fake_venv,
        cwd=declared_paths.output_dir,
        declared_paths=declared_paths,
    )
    try:
        response = handle.stub().HealthCheck(
            cuvis_ai_pb2.HealthCheckRequest(), timeout=5.0
        )
        assert (
            response.status == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
        )
        assert handle.endpoint.startswith("127.0.0.1:")
        assert handle.process.poll() is None
    finally:
        handle.terminate(grace_s=2.0)
    # After terminate, the process has exited.
    assert handle.process.poll() is not None


def test_spawn_curates_env_no_secret_passthrough(
    fake_venv: Path, declared_paths, monkeypatch
):
    """SSH agent / cloud tokens must NOT be inherited by the child."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "also-should-not-leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")

    spawner = LocalChildRuntimeSpawner()
    env = spawner._build_child_env(
        venv_path=fake_venv,
        declared_paths=declared_paths,
        request_gpu=False,
    )
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "SSH_AUTH_SOCK" not in env
    # HOME is faked into output_dir / .home
    assert env["HOME"] == str(declared_paths.output_dir / ".home")
    assert env["TEMP"] == str(declared_paths.scratch_dir)
    # PYTHONPATH must not be set — uv handles .pth files via the venv.
    assert "PYTHONPATH" not in env
    # PATH is prepended with the venv's bin/Scripts directory.
    first_path = env["PATH"].split(os.pathsep)[0]
    assert "Scripts" in first_path or "bin" in first_path


def test_spawn_gpu_flag_passes_cuda_vars_through(
    fake_venv: Path, declared_paths, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    spawner = LocalChildRuntimeSpawner()
    env = spawner._build_child_env(
        venv_path=fake_venv,
        declared_paths=declared_paths,
        request_gpu=True,
    )
    assert env.get("CUDA_VISIBLE_DEVICES") == "0,1"


def test_spawn_keeps_ld_library_path_without_gpu(
    fake_venv: Path, declared_paths, monkeypatch
):
    """LD_LIBRARY_PATH must survive curation even when GPU is not requested.

    It is the dynamic-linker search path the child interpreter may need to
    start; dropping it made the child exit 127 before Python ran.
    """
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/python/lib")
    spawner = LocalChildRuntimeSpawner()
    env = spawner._build_child_env(
        venv_path=fake_venv,
        declared_paths=declared_paths,
        request_gpu=False,
    )
    assert env.get("LD_LIBRARY_PATH") == "/opt/python/lib"


def test_spawn_terminate_returns_returncode(fake_venv: Path, declared_paths):
    spawner = LocalChildRuntimeSpawner()
    handle = spawner.spawn(
        fake_venv,
        cwd=declared_paths.output_dir,
        declared_paths=declared_paths,
    )
    rc = handle.terminate(grace_s=2.0)
    assert rc is not None
    assert handle.returncode == rc


def test_spawn_endpoint_polling_surfaces_child_death(
    tmp_path: Path, declared_paths, monkeypatch
):
    """A child that dies before writing the endpoint produces a clear SpawnError."""
    # Tighten the endpoint-poll timeout so the test runs quickly when the
    # child never writes its endpoint.
    monkeypatch.setattr(
        "cuvis_ai_core.orchestrator.spawner._ENDPOINT_POLL_TIMEOUT_SECONDS", 2.0
    )

    # Build a fake venv whose python is a stub script that exits non-zero.
    venv = tmp_path / "doomed_venv"
    if sys.platform == "win32":
        # On Windows we just symlink the real python and rely on the bad
        # bind address to make module import fail.
        return  # skip on Windows; behaviour exercised by the missing-venv test
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    target = bin_dir / "python"
    target.write_text("#!/bin/sh\nexit 7\n")
    target.chmod(0o755)

    spawner = LocalChildRuntimeSpawner()
    with pytest.raises(SpawnError):
        spawner.spawn(
            venv,
            cwd=declared_paths.output_dir,
            declared_paths=declared_paths,
        )


# ---------------------------------------------------------------------------
# _timeout_from_env
# ---------------------------------------------------------------------------


def test_timeout_from_env_unset_returns_default(monkeypatch):
    monkeypatch.delenv("CUVIS_TEST_TIMEOUT", raising=False)
    assert _timeout_from_env("CUVIS_TEST_TIMEOUT", 12.0) == 12.0


def test_timeout_from_env_valid_value_is_used(monkeypatch):
    monkeypatch.setenv("CUVIS_TEST_TIMEOUT", "3.5")
    assert _timeout_from_env("CUVIS_TEST_TIMEOUT", 12.0) == 3.5


def test_timeout_from_env_non_numeric_falls_back(monkeypatch):
    monkeypatch.setenv("CUVIS_TEST_TIMEOUT", "not-a-number")
    assert _timeout_from_env("CUVIS_TEST_TIMEOUT", 12.0) == 12.0


def test_timeout_from_env_non_positive_falls_back(monkeypatch):
    monkeypatch.setenv("CUVIS_TEST_TIMEOUT", "0")
    assert _timeout_from_env("CUVIS_TEST_TIMEOUT", 12.0) == 12.0
    monkeypatch.setenv("CUVIS_TEST_TIMEOUT", "-4")
    assert _timeout_from_env("CUVIS_TEST_TIMEOUT", 12.0) == 12.0


# ---------------------------------------------------------------------------
# _read_stderr_log
# ---------------------------------------------------------------------------


def test_read_stderr_log_none_or_missing_returns_empty(tmp_path: Path):
    assert _read_stderr_log(None) == ""
    assert _read_stderr_log(tmp_path / "absent.log") == ""


def test_read_stderr_log_reads_existing_file(tmp_path: Path):
    log = tmp_path / "stderr.log"
    log.write_text("traceback here", encoding="utf-8")
    assert _read_stderr_log(log) == "traceback here"


def test_read_stderr_log_surfaces_oserror(tmp_path: Path, monkeypatch):
    log = tmp_path / "stderr.log"
    log.write_text("x", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_text", _boom)
    out = _read_stderr_log(log)
    assert out.startswith("<unreadable stderr log")
    assert "disk gone" in out


# ---------------------------------------------------------------------------
# ChildHandle.terminate / kill, driven by a mocked Popen so the SIGTERM /
# kill escalation is deterministic and platform-independent.
# ---------------------------------------------------------------------------


def _fake_proc(*, poll, returncode=0):
    """Build a Popen stand-in. ``poll`` is a value or a side_effect list."""
    from unittest.mock import MagicMock

    proc = MagicMock(spec=subprocess.Popen)
    if isinstance(poll, list):
        proc.poll.side_effect = poll
    else:
        proc.poll.return_value = poll
    proc.returncode = returncode
    return proc


def _quiet_stub(monkeypatch, handle):
    """Replace handle.stub() with one whose StopRun is a no-op."""
    from unittest.mock import MagicMock

    stub = MagicMock()
    monkeypatch.setattr(handle, "stub", lambda: stub)
    return stub


def test_terminate_returns_returncode_when_already_dead():
    proc = _fake_proc(poll=4, returncode=4)
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    assert handle.terminate() == 4
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_terminate_graceful_stop_then_clean_exit(monkeypatch):
    proc = _fake_proc(poll=None)
    proc.wait.return_value = 0
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    stub = _quiet_stub(monkeypatch, handle)
    assert handle.terminate(grace_s=2.0) == 0
    stub.StopRun.assert_called_once()
    proc.terminate.assert_not_called()


def test_terminate_tolerates_stoprun_rpc_error(monkeypatch):
    proc = _fake_proc(poll=None)
    proc.wait.return_value = 0
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    stub = _quiet_stub(monkeypatch, handle)
    stub.StopRun.side_effect = grpc.RpcError("boom")
    assert handle.terminate(grace_s=2.0) == 0


def test_terminate_escalates_to_sigterm_then_succeeds(monkeypatch):
    proc = _fake_proc(poll=None)
    proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="child", timeout=2.0),
        0,
    ]
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    _quiet_stub(monkeypatch, handle)
    assert handle.terminate(grace_s=2.0) == 0
    proc.terminate.assert_called_once()


def test_terminate_escalates_to_kill_when_sigterm_ignored(monkeypatch):
    # poll: alive in terminate(), alive again inside kill().
    proc = _fake_proc(poll=[None, None], returncode=-9)
    proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="child", timeout=2.0),
        subprocess.TimeoutExpired(cmd="child", timeout=2.0),
        subprocess.TimeoutExpired(cmd="child", timeout=5.0),
    ]
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    _quiet_stub(monkeypatch, handle)
    assert handle.terminate(grace_s=2.0) == -9
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


def test_kill_noop_when_already_dead():
    proc = _fake_proc(poll=0, returncode=0)
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    assert handle.kill() == 0
    proc.kill.assert_not_called()


def test_close_channel_closes_open_channel():
    from unittest.mock import MagicMock

    proc = _fake_proc(poll=0, returncode=0)
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    channel = MagicMock()
    handle._channel = channel
    handle.kill()  # routes through _close_channel
    channel.close.assert_called_once()
    assert handle._channel is None


# ---------------------------------------------------------------------------
# _wait_for_endpoint / _wait_for_health: death and timeout branches, driven
# by a mocked Popen so they are deterministic on every platform.
# ---------------------------------------------------------------------------


def test_wait_for_endpoint_surfaces_child_death(tmp_path: Path):
    spawner = LocalChildRuntimeSpawner()
    stderr = tmp_path / "stderr.log"
    stderr.write_text("ImportError: boom", encoding="utf-8")
    proc = _fake_proc(poll=7, returncode=7)
    with pytest.raises(SpawnError, match="exited before reporting"):
        spawner._wait_for_endpoint(tmp_path / "endpoint.txt", proc, stderr_log=stderr)


def test_wait_for_endpoint_times_out(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "cuvis_ai_core.orchestrator.spawner._ENDPOINT_POLL_TIMEOUT_SECONDS", 0.2
    )
    spawner = LocalChildRuntimeSpawner()
    proc = _fake_proc(poll=None)
    with pytest.raises(SpawnError, match="did not write endpoint"):
        spawner._wait_for_endpoint(tmp_path / "endpoint.txt", proc)
    proc.terminate.assert_called_once()


def test_wait_for_health_surfaces_child_death(tmp_path: Path):
    spawner = LocalChildRuntimeSpawner()
    stderr = tmp_path / "stderr.log"
    stderr.write_text("segfault", encoding="utf-8")
    proc = _fake_proc(poll=1, returncode=1)
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc, stderr_log=stderr)
    with pytest.raises(SpawnError, match="died before HealthCheck"):
        spawner._wait_for_health(handle)


def test_wait_for_health_times_out_when_never_serving(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "cuvis_ai_core.orchestrator.spawner._HEALTH_POLL_TIMEOUT_SECONDS", 0.2
    )
    monkeypatch.setattr(
        "cuvis_ai_core.orchestrator.spawner._HEALTH_POLL_INTERVAL_SECONDS", 0.05
    )
    spawner = LocalChildRuntimeSpawner()
    proc = _fake_proc(poll=None)
    handle = ChildHandle(endpoint="127.0.0.1:1", process=proc)
    stub = MagicMock()
    stub.HealthCheck.side_effect = grpc.RpcError("not up yet")
    monkeypatch.setattr(handle, "stub", lambda: stub)
    with pytest.raises(SpawnError, match="did not become SERVING"):
        spawner._wait_for_health(handle)
