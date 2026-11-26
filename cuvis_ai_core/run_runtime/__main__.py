"""``python -m cuvis_ai_core.run_runtime`` entry point.

Launched by :class:`LocalChildRuntimeSpawner` inside a per-pipeline
composed venv. Binds an ephemeral loopback port, writes the chosen
endpoint to a file the parent reads, and blocks on
``server.wait_for_termination`` until ``StopRun`` sets the shutdown
event. Every call has to carry the token the parent put in
``CUVIS_RUNTIME_TOKEN``; a child started by hand without one checks nothing.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from concurrent import futures
from pathlib import Path

import grpc
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2_grpc
from loguru import logger

from cuvis_ai_core.run_runtime.service import RunRuntimeServicer
from cuvis_ai_core.utils.grpc_auth import (
    RUNTIME_TOKEN_ENV,
    TokenInterceptor,
    max_message_size,
    pop_token,
)

_DEFAULT_BIND = "127.0.0.1:0"
_DEFAULT_MAX_WORKERS = 8
# The parent forwards a client's request, at most max_message_size() bytes, with a resolved
# path in place of the one the client sent; this margin covers that growth.
FORWARD_MARGIN = 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Read before any plugin code is imported, and removed so that code cannot read it.
    token = pop_token(RUNTIME_TOKEN_ENV)
    if token is None:
        logger.warning(
            f"No {RUNTIME_TOKEN_ENV} set: every process on this machine may call this runtime"
        )
    shutdown_event = threading.Event()
    servicer = RunRuntimeServicer(shutdown_event=shutdown_event)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            # Inference answers (full frames and score maps) can be large.
            ("grpc.max_send_message_length", -1),
            # No more than the parent accepts, so a local caller cannot make it allocate more.
            ("grpc.max_receive_message_length", _receive_limit()),
        ],
        interceptors=[TokenInterceptor(token)] if token else [],
    )
    cuvis_ai_pb2_grpc.add_RunRuntimeServicer_to_server(servicer, server)
    chosen_port = server.add_insecure_port(args.bind)
    if not chosen_port:
        logger.error(f"Failed to bind RunRuntime server to {args.bind}")
        return 1
    endpoint = _format_endpoint(args.bind, chosen_port)
    server.start()
    logger.info(f"RunRuntime listening on {endpoint}")

    if args.endpoint_file:
        _write_endpoint_file(Path(args.endpoint_file), endpoint)
    if args.print_endpoint:
        print(endpoint, flush=True)

    _install_signal_handlers(shutdown_event)

    try:
        shutdown_event.wait()
    finally:
        logger.info("Stopping RunRuntime server (grace 5s)")
        server.stop(grace=5.0).wait()
    return 0


def _receive_limit() -> int:
    """The parent's cap plus :data:`FORWARD_MARGIN`; ``-1`` (no limit) when the parent has none."""
    limit = max_message_size()
    return -1 if limit < 0 else limit + FORWARD_MARGIN


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cuvis_ai_core.run_runtime",
        description="Child runtime gRPC service for the cuvis-ai orchestrator.",
    )
    parser.add_argument(
        "--bind",
        default=_DEFAULT_BIND,
        help=(
            "Bind address, including '0' for an ephemeral port (default: 127.0.0.1:0)."
        ),
    )
    parser.add_argument(
        "--endpoint-file",
        default=None,
        help=(
            "Path to write the chosen 'host:port' endpoint to once the "
            "server is bound. The parent polls this file."
        ),
    )
    parser.add_argument(
        "--print-endpoint",
        action="store_true",
        help="Print the chosen endpoint to stdout (useful in manual runs).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=_DEFAULT_MAX_WORKERS,
        help=f"gRPC thread pool size (default: {_DEFAULT_MAX_WORKERS}).",
    )
    return parser.parse_args(argv)


def _format_endpoint(bind: str, chosen_port: int) -> str:
    # Child runtimes bind IPv4 loopback (see _DEFAULT_BIND), so a plain
    # host:port split is sufficient; IPv6 bracket forms are not handled.
    host = bind.rpartition(":")[0] or "127.0.0.1"
    return f"{host}:{chosen_port}"


def _write_endpoint_file(path: Path, endpoint: str) -> None:
    """Write the endpoint atomically so the parent's poll never sees a partial value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(endpoint, encoding="utf-8")
    os.replace(tmp, path)


def _install_signal_handlers(shutdown_event: threading.Event) -> None:
    def handle(signum: int, _frame) -> None:
        logger.info(f"Received signal {signum}; setting shutdown event.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    if hasattr(signal, "SIGBREAK"):  # Windows
        signal.signal(signal.SIGBREAK, handle)


if __name__ == "__main__":
    sys.exit(main())
