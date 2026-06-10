"""Unit tests for the run_runtime entry point (``python -m cuvis_ai_core.run_runtime``).

``main()`` blocks on ``shutdown_event.wait()``; every test that calls it swaps in
``_ImmediateEvent`` whose ``wait()`` returns at once, so the tests cannot hang.
The gRPC server, servicer, and signal registration are mocked, so no socket is
bound and no real signal handler is installed.
"""

from __future__ import annotations

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock

from cuvis_ai_core.run_runtime import __main__ as main_mod


class _ImmediateEvent:
    """Stand-in for ``threading.Event`` whose ``wait()`` returns immediately."""

    def __init__(self) -> None:
        self._set = False

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def _fake_server(chosen_port: int) -> MagicMock:
    server = MagicMock(name="grpc_server")
    server.add_insecure_port.return_value = chosen_port
    return server


def _patch_main(monkeypatch, *, chosen_port: int) -> MagicMock:
    """Mock every collaborator ``main()`` touches; return the fake server."""
    server = _fake_server(chosen_port)
    # Local rebind so ``threading.Event()`` returns a non-blocking stub.
    monkeypatch.setattr(main_mod, "threading", SimpleNamespace(Event=_ImmediateEvent))
    monkeypatch.setattr(main_mod.grpc, "server", lambda *a, **k: server)
    monkeypatch.setattr(main_mod, "RunRuntimeServicer", MagicMock(name="Servicer"))
    monkeypatch.setattr(
        main_mod.cuvis_ai_pb2_grpc, "add_RunRuntimeServicer_to_server", MagicMock()
    )
    # Patch signal registration so the test process keeps its own handlers.
    monkeypatch.setattr(main_mod.signal, "signal", MagicMock(name="signal.signal"))
    return server


def test_main_happy_path_writes_endpoint_and_stops(monkeypatch, tmp_path):
    server = _patch_main(monkeypatch, chosen_port=12345)
    endpoint_file = tmp_path / "ep.txt"

    rc = main_mod.main(["--bind", "127.0.0.1:0", "--endpoint-file", str(endpoint_file)])

    assert rc == 0
    assert endpoint_file.read_text(encoding="utf-8") == "127.0.0.1:12345"
    server.start.assert_called_once()
    server.stop.assert_called_once_with(grace=5.0)
    server.stop.return_value.wait.assert_called_once()


def test_main_bind_failure_returns_1(monkeypatch, tmp_path):
    server = _patch_main(monkeypatch, chosen_port=0)
    endpoint_file = tmp_path / "ep.txt"

    rc = main_mod.main(
        ["--bind", "127.0.0.1:9999", "--endpoint-file", str(endpoint_file)]
    )

    assert rc == 1
    server.start.assert_not_called()
    assert not endpoint_file.exists()


def test_main_prints_endpoint_when_flag_set(monkeypatch, capsys):
    _patch_main(monkeypatch, chosen_port=23456)

    rc = main_mod.main(["--print-endpoint"])

    assert rc == 0
    assert "127.0.0.1:23456" in capsys.readouterr().out


def test_parse_args_defaults():
    args = main_mod._parse_args([])
    assert args.bind == "127.0.0.1:0"
    assert args.endpoint_file is None
    assert args.print_endpoint is False
    assert args.max_workers == 8


def test_parse_args_custom():
    args = main_mod._parse_args(
        [
            "--bind",
            "127.0.0.1:5005",
            "--endpoint-file",
            "/tmp/ep",
            "--print-endpoint",
            "--max-workers",
            "3",
        ]
    )
    assert args.bind == "127.0.0.1:5005"
    assert args.endpoint_file == "/tmp/ep"
    assert args.print_endpoint is True
    assert args.max_workers == 3


def test_format_endpoint():
    assert main_mod._format_endpoint("127.0.0.1:0", 12345) == "127.0.0.1:12345"
    assert main_mod._format_endpoint("0.0.0.0:0", 8888) == "0.0.0.0:8888"
    # No colon in the bind string falls back to the loopback host.
    assert main_mod._format_endpoint("localhost", 5555) == "127.0.0.1:5555"


def test_write_endpoint_file_atomic(tmp_path):
    target = tmp_path / "nested" / "ep.txt"

    main_mod._write_endpoint_file(target, "127.0.0.1:7000")

    assert target.read_text(encoding="utf-8") == "127.0.0.1:7000"
    # The temp file is renamed away, not left behind.
    assert not (target.parent / "ep.txt.tmp").exists()


def test_install_signal_handlers_sets_event(monkeypatch):
    registered: dict[int, object] = {}

    monkeypatch.setattr(
        main_mod.signal,
        "signal",
        lambda signum, handler: registered.__setitem__(signum, handler),
    )
    event = _ImmediateEvent()

    main_mod._install_signal_handlers(event)

    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered
    # Invoking a registered handler sets the shutdown event.
    registered[signal.SIGINT](signal.SIGINT, None)
    assert event.is_set()
