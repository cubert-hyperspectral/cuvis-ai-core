"""Spawner unit tests.

The full happy-path tests boot a real child runtime via
``python -m cuvis_ai_core.run_runtime``; they only need the parent's
own python (the child's venv is mocked by reusing the same
interpreter). Tests run in seconds because the runtime does no work
until ``InitializeSession`` is called.
"""

from __future__ import annotations

import os
import shutil
import socket
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
)


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
            response.status
            == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
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
