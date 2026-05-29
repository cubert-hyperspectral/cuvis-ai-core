"""Child runtime spawner — single seam item 06's future sandbox replaces.

The local implementation launches a ``python -m cuvis_ai_core.run_runtime``
subprocess inside the composed venv. The child binds an ephemeral
loopback TCP port and writes its endpoint to a file the parent reads;
that decouples the two so we don't race on port allocation. Once the
endpoint file appears, the parent ``HealthCheck``-polls the gRPC stub
until SERVING, then returns the handle.

The spawner is the single place that decides what env vars the child
inherits. Phase 3b implementation:

- ``PATH`` is prepended with the child venv's ``bin`` / ``Scripts`` so
  bundled scripts resolve first.
- ``PYTHONPATH`` is **left unset** — launching ``venv_python`` already
  gives the child the right ``sys.path`` via the venv's site-packages
  and any editable ``.pth`` files uv installed. Overriding
  ``PYTHONPATH`` risks hiding legitimately installed packages.
- ``HOME`` is pointed at a run-scoped scratch dir so libraries that
  assume ``HOME`` exists (``huggingface_hub``, ``matplotlib``,
  ``transformers``, Jupyter, pip cache) write there instead of the
  real user ``HOME``.
- CUDA-related vars pass through when GPU is requested.
- SSH agent socket, ``AWS_*``, ``GITHUB_*``, ``HUGGINGFACE_HUB_TOKEN``,
  ``.env`` bleed-through, and other implicit credentials are excluded.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import grpc
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from loguru import logger

from cuvis_ai_core.orchestrator.venv_paths import venv_bin_dir, venv_python

_ENDPOINT_POLL_TIMEOUT_SECONDS = 30.0
_ENDPOINT_POLL_INTERVAL_SECONDS = 0.05
_HEALTH_POLL_TIMEOUT_SECONDS = 30.0
_HEALTH_POLL_INTERVAL_SECONDS = 0.2
_CUDA_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
)

# Explicit deny-list for env vars that must never leak from the parent
# into the child. Anything matching one of the patterns below is
# stripped after the initial os.environ copy. Patterns ending in ``*``
# match by prefix.
_DENY_PATTERNS = (
    "PYTHONPATH",  # uv's .pth files handle site-packages; PYTHONPATH would shadow
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "AWS_*",
    "GITHUB_*",
    "GH_TOKEN",
    "GITLAB_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_*",
)


def _read_stderr_log(path: Path | None) -> str:
    """Read the child's captured stderr file for post-mortem display.

    Returns ``""`` if the file is missing — callers fall back to a
    generic crash message in that case.
    """
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unreadable stderr log {path}: {exc}>"


def _is_denied(name: str) -> bool:
    for pat in _DENY_PATTERNS:
        if pat.endswith("*"):
            if name.startswith(pat[:-1]):
                return True
        elif name == pat:
            return True
    return False


class SpawnError(RuntimeError):
    """Raised when the child runtime cannot be launched or reached."""


@dataclass(frozen=True)
class DeclaredPaths:
    """Filesystem paths the child is permitted to read / write.

    Forwarded into ``InitializeSession`` so a future FS sandbox knows
    the bind-mount set without rummaging through per-request payloads.
    """

    output_dir: Path
    scratch_dir: Path


@dataclass
class ChildHandle:
    """Live reference to a running child runtime process.

    ``endpoint`` is the ``host:port`` the parent's :func:`stub` talks
    to. ``terminate`` / ``kill`` are the graceful and forced shutdown
    paths the orchestrator calls on session close or crash recovery.
    ``stderr_log`` points at the file that captures the child's
    stderr — read on crash to surface a useful error.
    """

    endpoint: str
    process: subprocess.Popen
    stderr_log: Path | None = None
    _channel: grpc.Channel | None = field(default=None, init=False, repr=False)

    def stub(self) -> cuvis_ai_pb2_grpc.RunRuntimeStub:
        if self._channel is None:
            self._channel = grpc.insecure_channel(self.endpoint)
        return cuvis_ai_pb2_grpc.RunRuntimeStub(self._channel)

    def terminate(self, grace_s: float = 5.0) -> int | None:
        """Politely ask the child to exit; hard-kill after ``grace_s``.

        Returns the process exit code (``None`` if already dead).
        """
        if self.process.poll() is not None:
            self._close_channel()
            return self.process.returncode

        try:
            # Send a graceful StopRun if we can still talk to it.
            try:
                self.stub().StopRun(
                    cuvis_ai_pb2.StopRunRequest(grace_seconds=int(grace_s)),
                    timeout=min(grace_s, 5.0),
                )
            except grpc.RpcError as exc:
                logger.debug(f"StopRun RPC failed (continuing with terminate): {exc}")

            try:
                return self.process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Child runtime at {self.endpoint} did not exit within "
                    f"{grace_s}s; sending SIGTERM."
                )
        finally:
            pass

        self.process.terminate()
        try:
            return self.process.wait(timeout=max(grace_s, 1.0))
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Child runtime at {self.endpoint} still alive after SIGTERM; killing."
            )
        return self.kill()

    def kill(self) -> int | None:
        """Send the platform's hard-kill signal and wait for the process to die."""
        if self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.error(f"Child at {self.endpoint} did not die after kill().")
        self._close_channel()
        return self.process.returncode

    @property
    def returncode(self) -> int | None:
        return self.process.poll()

    def _close_channel(self) -> None:
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(f"channel close error: {exc}")
            self._channel = None


class ChildRuntimeSpawner(ABC):
    """Abstract spawn surface.

    Item 06 will add concrete spawners that wrap this same call (e.g.
    a job-object or container spawner) without touching callers.
    """

    @abstractmethod
    def spawn(
        self,
        venv_path: Path,
        *,
        cwd: Path,
        declared_paths: DeclaredPaths,
        request_gpu: bool = False,
    ) -> ChildHandle: ...


class LocalChildRuntimeSpawner(ChildRuntimeSpawner):
    """Spawns the child runtime as a local subprocess.

    The default and only implementation in Phase 3b. The interface is
    stable; future sandboxed variants implement the same ``spawn``
    signature.
    """

    def spawn(
        self,
        venv_path: Path,
        *,
        cwd: Path,
        declared_paths: DeclaredPaths,
        request_gpu: bool = False,
    ) -> ChildHandle:
        python = venv_python(venv_path)
        if not python.exists():
            raise SpawnError(
                f"Child venv has no python at {python}. Did compose_env complete?"
            )

        endpoint_dir = Path(tempfile.mkdtemp(prefix="cuvis_runtime_endpoint_"))
        endpoint_file = endpoint_dir / "endpoint.txt"
        env = self._build_child_env(
            venv_path=venv_path,
            declared_paths=declared_paths,
            request_gpu=request_gpu,
        )
        # Capture child stdout/stderr to files in the declared scratch
        # dir rather than ``subprocess.PIPE``. With PIPE the parent must
        # actively drain the OS pipe (~64 KB on Windows) or the child's
        # next write blocks forever — a deadlock loguru triggered
        # reliably while preinstalled plugins were being registered.
        # File sinks have no such limit, and we can still read them on
        # crash to surface the failure to the caller.
        log_dir = declared_paths.scratch_dir / "runtime"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "child.stdout.log"
        stderr_path = log_dir / "child.stderr.log"
        cmd = [
            str(python),
            "-m",
            "cuvis_ai_core.run_runtime",
            "--endpoint-file",
            str(endpoint_file),
        ]
        logger.info(
            f"Spawning child runtime: {' '.join(cmd)} "
            f"(stdout→{stdout_path}, stderr→{stderr_path})"
        )
        stdout_fh = stdout_path.open("w", encoding="utf-8")
        stderr_fh = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )

        try:
            endpoint = self._wait_for_endpoint(
                endpoint_file, process, stderr_log=stderr_path
            )
            handle = ChildHandle(
                endpoint=endpoint, process=process, stderr_log=stderr_path
            )
            self._wait_for_health(handle)
        except SpawnError:
            if process.poll() is None:
                process.terminate()
            raise
        finally:
            shutil.rmtree(endpoint_dir, ignore_errors=True)
            # Close the file handles the *parent* opened; the child
            # inherited duplicates so its writes continue uninterrupted.
            stdout_fh.close()
            stderr_fh.close()
        return handle

    def _build_child_env(
        self,
        *,
        venv_path: Path,
        declared_paths: DeclaredPaths,
        request_gpu: bool,
    ) -> dict[str, str]:
        """Build the curated env handed to the child process.

        Starts from the parent's full environment so platform-mandatory
        vars (``SystemRoot`` / ``APPDATA`` on Windows, the system ``PATH``,
        ``PATHEXT``, etc.) are inherited and the venv's ``pyvenv.cfg``
        loads correctly. Then strips a deny-list of credential / API-key
        env vars, overrides ``HOME`` / ``TEMP`` to run-scoped scratch
        dirs, and prepends the venv's bin/Scripts directory to ``PATH``.
        ``PYTHONPATH`` is always cleared — uv's ``.pth`` files handle
        site-packages, and inheriting ``PYTHONPATH`` would risk
        shadowing the composed venv's packages.
        """
        env = {k: v for k, v in os.environ.items() if not _is_denied(k)}

        # When GPU is not requested, also strip CUDA_* so the child can't
        # accidentally grab a GPU. Production callers must opt in.
        if not request_gpu:
            for var in _CUDA_VARS:
                env.pop(var, None)

        # Fake HOME / scratch redirects (libraries assume HOME exists).
        fake_home = declared_paths.output_dir / ".home"
        env["HOME"] = str(fake_home)
        env["USERPROFILE"] = str(fake_home)
        env["TEMP"] = str(declared_paths.scratch_dir)
        env["TMP"] = str(declared_paths.scratch_dir)
        env["TMPDIR"] = str(declared_paths.scratch_dir)

        # PATH: prepend the venv's bin/Scripts so child scripts resolve.
        bin_dir = venv_bin_dir(venv_path)
        env["PATH"] = _prepend_path(env.get("PATH", ""), str(bin_dir))
        return env

    def _wait_for_endpoint(
        self,
        endpoint_file: Path,
        process: subprocess.Popen,
        *,
        stderr_log: Path | None = None,
    ) -> str:
        deadline = time.monotonic() + _ENDPOINT_POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = _read_stderr_log(stderr_log)
                raise SpawnError(
                    f"Child runtime exited before reporting an endpoint "
                    f"(returncode={process.returncode}). stderr:\n{stderr}"
                )
            if endpoint_file.exists():
                text = endpoint_file.read_text(encoding="utf-8").strip()
                if text:
                    return text
            time.sleep(_ENDPOINT_POLL_INTERVAL_SECONDS)
        process.terminate()
        raise SpawnError(
            f"Child runtime did not write endpoint within "
            f"{_ENDPOINT_POLL_TIMEOUT_SECONDS}s."
        )

    def _wait_for_health(self, handle: ChildHandle) -> None:
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if handle.process.poll() is not None:
                stderr = _read_stderr_log(handle.stderr_log)
                raise SpawnError(
                    f"Child runtime died before HealthCheck succeeded "
                    f"(returncode={handle.process.returncode}). stderr:\n{stderr}"
                )
            try:
                response = handle.stub().HealthCheck(
                    cuvis_ai_pb2.HealthCheckRequest(),
                    timeout=1.0,
                )
                if response.status == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING:
                    return
            except grpc.RpcError as exc:
                last_error = exc
            time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
        raise SpawnError(
            f"Child runtime at {handle.endpoint} did not become SERVING within "
            f"{_HEALTH_POLL_TIMEOUT_SECONDS}s. Last error: {last_error}"
        )


def _prepend_path(existing: str, new_entry: str) -> str:
    if not existing:
        return new_entry
    return f"{new_entry}{os.pathsep}{existing}"
