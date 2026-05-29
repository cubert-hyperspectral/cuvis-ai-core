"""``python -m cuvis_ai_core.run_runtime`` entry point.

Launched by :class:`LocalChildRuntimeSpawner` inside a per-pipeline
composed venv. Binds an ephemeral loopback port, writes the chosen
endpoint to a file the parent reads, and blocks on
``server.wait_for_termination`` until ``StopRun`` sets the shutdown
event.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import threading
from concurrent import futures
from pathlib import Path

import grpc
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2_grpc
from loguru import logger

from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

_DEFAULT_BIND = "127.0.0.1:0"
_DEFAULT_MAX_WORKERS = 8


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    shutdown_event = threading.Event()
    servicer = RunRuntimeServicer(shutdown_event=shutdown_event)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cuvis_ai_core.run_runtime",
        description="Child runtime gRPC service for the cuvis-ai orchestrator.",
    )
    parser.add_argument(
        "--bind",
        default=_DEFAULT_BIND,
        help=(
            "Bind address, including '0' for an ephemeral port "
            "(default: 127.0.0.1:0)."
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
    host, _, port = bind.rpartition(":")
    if not host:
        host = "127.0.0.1"
    # Preserve IPv6 bracket form if present.
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
