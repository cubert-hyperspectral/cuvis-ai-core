"""Child runtime spawner — the single seam a future sandbox layer replaces.

The local implementation launches a ``python -m cuvis_ai_core.run_runtime``
subprocess inside the composed venv. The child binds an ephemeral
loopback TCP port and writes its endpoint to a file the parent reads;
that decouples the two so we don't race on port allocation. Once the
endpoint file appears, the parent ``HealthCheck``-polls the gRPC stub
until SERVING, then returns the handle.

The spawner is the single place that decides what env vars the child
inherits:

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

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import grpc
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from loguru import logger

from cuvis_ai_core.orchestrator.venv_paths import venv_bin_dir, venv_python

# Child-runtime startup deadlines (seconds). The two *_TIMEOUT defaults are
# overridable via env so a slow cold start — e.g. the first CUDA-torch import in
# a freshly composed venv on Windows — does not trip them before the child's gRPC
# server comes up. Env unset / non-positive / invalid falls back to the default.
_ENDPOINT_POLL_TIMEOUT_SECONDS = 30.0
_ENDPOINT_POLL_TIMEOUT_ENV = "CUVIS_RUNTIME_ENDPOINT_TIMEOUT_SECONDS"
_ENDPOINT_POLL_INTERVAL_SECONDS = 0.05
_HEALTH_POLL_TIMEOUT_SECONDS = 30.0
_HEALTH_POLL_TIMEOUT_ENV = "CUVIS_RUNTIME_HEALTH_TIMEOUT_SECONDS"
_HEALTH_POLL_INTERVAL_SECONDS = 0.2
_HEALTHCHECK_RPC_TIMEOUT_SECONDS = 1.0
_STOP_RUN_RPC_TIMEOUT_CAP_SECONDS = 5.0
_TERMINATE_KILL_WAIT_SECONDS = 5.0
_GRACEFUL_WAIT_FLOOR_SECONDS = 1.0

# Parent<->child runtime messages carry full tensor batches (cubes, masks),
# which dwarf gRPC's 4 MB default receive cap. Match the public server's limit
# (same GRPC_MAX_MSG_SIZE env, 300 MB default) so large inference inputs and
# outputs forward across the bridge instead of failing RESOURCE_EXHAUSTED.
_GRPC_MAX_MSG_SIZE = int(os.getenv("GRPC_MAX_MSG_SIZE", 300 * 1024 * 1024))
# CUDA device-selection vars dropped from the child env when GPU is not
# requested. LD_LIBRARY_PATH is deliberately NOT in this set: it is the
# dynamic-linker search path the child interpreter (and torch's own shared
# libraries) may need to start at all. Dropping it made the child exit 127
# before Python could run on runners whose interpreter resolves its libs via
# LD_LIBRARY_PATH.
_CUDA_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "NVIDIA_VISIBLE_DEVICES",
)


def _timeout_from_env(env_name: str, default: float) -> float:
    """Read a positive float timeout from ``env_name``; fall back to ``default``."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"{env_name}={raw!r} is not a number; using {default}s.")
        return default
    if value <= 0:
        logger.warning(f"{env_name}={raw!r} must be > 0; using {default}s.")
        return default
    return value


# Env vars that must never leak from the parent into the child, stripped
# after the initial os.environ copy. Exact names are removed outright;
# any var whose name starts with one of the prefixes is removed too.
# This is a best-effort deny-list, deliberately non-exhaustive: the child
# runs plugin code, so the principled boundary is an allow-list of the
# vars the runtime actually needs. Until that exists, keep this list
# covering the common credential carriers.
_DENY_EXACT = frozenset(
    {
        "PYTHONPATH",  # uv's .pth files handle site-packages; PYTHONPATH would shadow
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GH_TOKEN",
        "GITLAB_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "MONGODB_URI",
        "SLACK_TOKEN",
        "SLACK_BOT_TOKEN",
        "NPM_TOKEN",
        "PYPI_TOKEN",
        "TWILIO_AUTH_TOKEN",
        "SENTRY_DSN",
        "DOCKERHUB_TOKEN",
        "DOCKER_PASSWORD",
    }
)
_DENY_PREFIXES = (
    "AWS_",
    "GITHUB_",
    "AZURE_",
    "STRIPE_",
    "GCP_",
    "CLOUDFLARE_",
    "DIGITALOCEAN_",
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
    return name in _DENY_EXACT or name.startswith(_DENY_PREFIXES)


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
            self._channel = grpc.insecure_channel(
                self.endpoint,
                options=[
                    ("grpc.max_send_message_length", _GRPC_MAX_MSG_SIZE),
                    ("grpc.max_receive_message_length", _GRPC_MAX_MSG_SIZE),
                ],
            )
        return cuvis_ai_pb2_grpc.RunRuntimeStub(self._channel)

    def terminate(self, grace_s: float = 5.0) -> int | None:
        """Politely ask the child to exit; hard-kill after ``grace_s``.

        Returns the process exit code (``None`` if already dead).
        """
        if self.process.poll() is not None:
            self._close_channel()
            return self.process.returncode

        # Send a graceful StopRun if we can still talk to it.
        try:
            self.stub().StopRun(
                cuvis_ai_pb2.StopRunRequest(grace_seconds=int(grace_s)),
                timeout=min(grace_s, _STOP_RUN_RPC_TIMEOUT_CAP_SECONDS),
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

        self.process.terminate()
        try:
            return self.process.wait(timeout=max(grace_s, _GRACEFUL_WAIT_FLOOR_SECONDS))
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
                self.process.wait(timeout=_TERMINATE_KILL_WAIT_SECONDS)
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

    Concrete spawners (a local subprocess today; job-object or container
    variants later) implement ``spawn`` without touching callers.
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

    The default subprocess-based implementation. The interface is
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
        # Open the log sinks inside an ExitStack so they are always
        # closed — even if Popen (or the second open) raises before the
        # main try/finally below is entered. The child inherits duplicate
        # descriptors, so closing the parent's copies leaves its writes
        # uninterrupted.
        with contextlib.ExitStack() as log_files:
            stdout_fh = log_files.enter_context(stdout_path.open("w", encoding="utf-8"))
            stderr_fh = log_files.enter_context(stderr_path.open("w", encoding="utf-8"))
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

        # When GPU is not requested, drop CUDA_* so the child does not
        # inherit an explicit device selection. NOTE: popping these vars
        # does not actually hide GPUs — torch still sees all visible devices
        # by default. Truly hiding the GPU needs CUDA_VISIBLE_DEVICES set to
        # "" (empty); that device/egress policy is left to a future sandbox
        # layer.
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
        timeout_s = _timeout_from_env(
            _ENDPOINT_POLL_TIMEOUT_ENV, _ENDPOINT_POLL_TIMEOUT_SECONDS
        )
        deadline = time.monotonic() + timeout_s
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
        raise SpawnError(f"Child runtime did not write endpoint within {timeout_s}s.")

    def _wait_for_health(self, handle: ChildHandle) -> None:
        timeout_s = _timeout_from_env(
            _HEALTH_POLL_TIMEOUT_ENV, _HEALTH_POLL_TIMEOUT_SECONDS
        )
        deadline = time.monotonic() + timeout_s
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
                    timeout=_HEALTHCHECK_RPC_TIMEOUT_SECONDS,
                )
                if (
                    response.status
                    == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
                ):
                    return
            except grpc.RpcError as exc:
                last_error = exc
            time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
        raise SpawnError(
            f"Child runtime at {handle.endpoint} did not become SERVING within "
            f"{timeout_s}s. Last error: {last_error}"
        )


def _prepend_path(existing: str, new_entry: str) -> str:
    if not existing:
        return new_entry
    return f"{new_entry}{os.pathsep}{existing}"
