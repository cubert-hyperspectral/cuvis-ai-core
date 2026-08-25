"""Production-ready gRPC server with Docker support, health checks, and graceful shutdown."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any

import grpc
from dotenv import load_dotenv
from grpc_health.v1 import health_pb2_grpc

from cuvis_ai_core.grpc import cuvis_ai_pb2_grpc
from cuvis_ai_core.grpc.health import HealthService
from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.orchestrator.composer import evict_run_cache, resolve_cache_root
from cuvis_ai_core.orchestrator.leases import reap_orphans, take_process_snapshot

# Cadence of the background cache-maintenance pass (orphan reaping +
# eviction). Startup-only reaping would leave a permanent hole: a sibling
# server crashing right after our one pass would leak its children until
# some server restarts.
_MAINTENANCE_INTERVAL_SECONDS = 3600.0
# Kill switch: "0" disables the maintenance loop entirely (test suites and
# operators who want eviction to run only via the explicit CLI).
_MAINTENANCE_ENV = "CUVIS_RUN_CACHE_MAINTENANCE"


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Support caller-provided extras (e.g., request ids)
        extras = getattr(record, "extra", None)
        if isinstance(extras, dict):
            log_data.update(extras)

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure root logging with optional JSON formatting."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()

    # Avoid duplicate handlers when reconfiguring (tests, reloads)
    root_logger.handlers.clear()

    if log_format.lower() == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    logging.getLogger("grpc").setLevel(level)


def load_tls_credentials(cert_path: str, key_path: str) -> grpc.ServerCredentials:
    """Load TLS credentials from disk."""
    cert_file = Path(cert_path).expanduser()
    key_file = Path(key_path).expanduser()

    if not cert_file.exists():
        raise FileNotFoundError(f"Certificate file not found: {cert_file}")
    if not key_file.exists():
        raise FileNotFoundError(f"Key file not found: {key_file}")

    cert = cert_file.read_bytes()
    key = key_file.read_bytes()

    return grpc.ssl_server_credentials([(key, cert)])


class ProductionServer:
    """Production-ready gRPC server with graceful shutdown."""

    def __init__(
        self,
        port: int = 50051,
        max_workers: int = 10,
        max_msg_size: int | None = None,
        use_tls: bool = False,
        tls_cert_path: str | None = None,
        tls_key_path: str | None = None,
    ) -> None:
        self.port = port
        self.max_workers = max_workers
        self.max_msg_size = max_msg_size
        self.use_tls = use_tls
        self.tls_cert_path = tls_cert_path
        self.tls_key_path = tls_key_path

        self.server: grpc.Server | None = None
        self.health_service: HealthService | None = None
        # Kept so shutdown() can reach the SessionManager and reap any child
        # runtimes the orchestrator spawned for live sessions.
        self.cuvis_service: CuvisAIService | None = None
        self.logger = logging.getLogger(__name__)
        self._shutdown_requested = False
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(
            signum: int, frame: Any
        ) -> None:  # pragma: no cover - small wrapper
            self.logger.info("Received signal %s, initiating graceful shutdown", signum)
            self._shutdown_requested = True

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal_handler)

    def start(self) -> None:
        """Start the gRPC server."""
        if self.server is not None:
            self.logger.warning("Server already started")
            return

        self.logger.info("Starting gRPC server")

        grpc_srv_opts = []
        if self.max_msg_size is not None:
            grpc_srv_opts.append(("grpc.max_send_message_length", self.max_msg_size))
            grpc_srv_opts.append(("grpc.max_receive_message_length", self.max_msg_size))

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=grpc_srv_opts,
        )

        self.cuvis_service = CuvisAIService()
        cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(
            self.cuvis_service, self.server
        )

        self.health_service = HealthService()
        health_pb2_grpc.add_HealthServicer_to_server(self.health_service, self.server)

        if self.use_tls:
            if not self.tls_cert_path or not self.tls_key_path:
                raise ValueError("TLS enabled but cert/key paths not provided")

            self.logger.info("Configuring TLS/SSL")
            credentials = load_tls_credentials(self.tls_cert_path, self.tls_key_path)
            bound_port = self.server.add_secure_port(
                f"0.0.0.0:{self.port}", credentials
            )
            if bound_port == 0:
                raise RuntimeError("Failed to bind secure port")
            self.logger.info("Server listening on 0.0.0.0:%s (TLS enabled)", bound_port)
        else:
            bound_port = self.server.add_insecure_port(f"0.0.0.0:{self.port}")
            if bound_port == 0:
                raise RuntimeError("Failed to bind insecure port")
            self.logger.info("Server listening on 0.0.0.0:%s (insecure)", bound_port)

        self.server.start()
        if self.health_service:
            self.health_service.set_serving()

        self.logger.info("Server started successfully")
        self._setup_signal_handlers()
        self._start_maintenance_thread()

    def _start_maintenance_thread(self) -> None:
        """Kick off the background cache-maintenance loop (post-bind).

        Runs orphan reaping + cache eviction immediately, then hourly.
        Deliberately AFTER the server reports SERVING: reaping can spend
        seconds per orphan (terminate → wait → kill), and blocking the
        bind would trip the host app's startup health polling exactly on
        recovery from a crash — the scenario the reaper exists for.
        """
        if os.environ.get(_MAINTENANCE_ENV, "1").strip() == "0":
            self.logger.info(
                "Cache maintenance disabled (%s=0); clean-run-cache remains "
                "the manual path",
                _MAINTENANCE_ENV,
            )
            return
        if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
            return

        def _loop() -> None:
            """Reap orphans + evict the run cache; repeat until shutdown."""
            # psutil's first full process scan in a process is slow on
            # Windows (tens of seconds for a few hundred processes) and its
            # process objects are cached afterwards. Paying it here keeps it
            # off the first compose, whose eviction pass needs the same scan.
            try:
                started = time.monotonic()
                take_process_snapshot()
                self.logger.debug(
                    "Process scan warmed in %.1fs", time.monotonic() - started
                )
            except Exception:
                self.logger.exception("Process scan warm-up failed")
            while True:
                try:
                    root = resolve_cache_root(None)
                    reap_orphans(root)
                    evict_run_cache(root)
                except Exception:
                    self.logger.exception("Cache maintenance pass failed")
                if self._maintenance_stop.wait(_MAINTENANCE_INTERVAL_SECONDS):
                    return

        self._maintenance_thread = threading.Thread(
            target=_loop, daemon=True, name="cuvis-cache-maintenance"
        )
        self._maintenance_thread.start()

    def wait_for_termination(self) -> None:
        """Block until a shutdown signal is received."""
        try:
            while not self._shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        if self.server is None:
            return

        self.logger.info("Initiating graceful shutdown")
        self._shutdown_requested = True
        self._maintenance_stop.set()

        if self.health_service:
            self.health_service.set_not_serving()

        # Reap any child runtimes the orchestrator spawned for live sessions.
        # close_session terminates the session's child handle; without this
        # walk the child subprocesses (which have no kill-on-parent-death)
        # outlive the server and leak, holding GPU memory until reboot.
        self._close_all_sessions()

        self.logger.info("Waiting for in-flight requests to complete")
        stop_future = self.server.stop(grace=5)
        stop_future.wait(timeout=10)

        self.server = None
        self.logger.info("Server shutdown complete")

    def _close_all_sessions(self) -> None:
        """Close every open session so its child runtime is terminated.

        Best-effort and per-session isolated: a failure closing one session
        is logged and does not block the rest of shutdown.
        """
        if self.cuvis_service is None:
            return
        session_manager = self.cuvis_service.session_manager
        for session_id in session_manager.list_sessions():
            try:
                session_manager.close_session(session_id)
            except Exception as exc:
                self.logger.warning(
                    "Failed to close session %s during shutdown: %s",
                    session_id,
                    exc,
                )


def serve() -> None:  # pragma: no cover
    """Start the production gRPC server with configuration from environment."""
    load_dotenv(override=True)

    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "json")
    setup_logging(log_level, log_format)

    logger = logging.getLogger(__name__)
    logger.info("Starting cuvis.ai gRPC server")

    port = int(os.getenv("GRPC_PORT", "50051"))
    max_workers = int(os.getenv("GRPC_MAX_WORKERS", "10"))
    max_msg_size = int(os.getenv("GRPC_MAX_MSG_SIZE", 300 * 1024 * 1024))  # 300 MB
    use_tls = os.getenv("GRPC_USE_TLS", "false").lower() == "true"
    tls_cert_path = os.getenv("GRPC_TLS_CERT_PATH")
    tls_key_path = os.getenv("GRPC_TLS_KEY_PATH")

    logger.info(
        "Configuration: port=%s, workers=%s, tls=%s", port, max_workers, use_tls
    )

    server = ProductionServer(
        port=port,
        max_workers=max_workers,
        max_msg_size=max_msg_size,
        use_tls=use_tls,
        tls_cert_path=tls_cert_path,
        tls_key_path=tls_key_path,
    )

    try:
        server.start()
        server.wait_for_termination()
    except Exception:
        logger.error("Server error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    serve()
