"""Unit tests for the run_runtime entry point (``python -m cuvis_ai_core.run_runtime``).

``main()`` blocks on ``shutdown_event.wait()``; every test that calls it swaps in
``_ImmediateEvent`` whose ``wait()`` returns at once, so the tests cannot hang.
The gRPC server, servicer, and signal registration are mocked, so no socket is
bound and no real signal handler is installed.
"""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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


def test_main_started_again_for_another_gpu_serves_nothing(monkeypatch, tmp_path):
    # Lightning's DDP launcher runs `python -m cuvis_ai_core.run_runtime ...` again for each
    # extra GPU, with LOCAL_RANK set and without the token the first runtime already removed.
    server = _patch_main(monkeypatch, chosen_port=12345)
    monkeypatch.setenv("LOCAL_RANK", "1")
    endpoint_file = tmp_path / "ep.txt"

    rc = main_mod.main(["--endpoint-file", str(endpoint_file)])

    assert rc == 1
    server.add_insecure_port.assert_not_called()
    server.start.assert_not_called()
    assert not endpoint_file.exists()


def test_main_on_the_first_gpu_serves(monkeypatch, tmp_path):
    server = _patch_main(monkeypatch, chosen_port=12345)
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert main_mod.main(["--endpoint-file", str(tmp_path / "ep.txt")]) == 0
    server.start.assert_called_once()


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


def _record_server_kwargs(monkeypatch, server: MagicMock) -> dict:
    seen: dict = {}

    def factory(*args, **kwargs):
        seen.update(kwargs)
        return server

    monkeypatch.setattr(main_mod.grpc, "server", factory)
    return seen


@pytest.mark.parametrize("cap", [None, "1000"])
def test_main_receives_no_more_than_the_parent_accepts(monkeypatch, tmp_path, cap):
    from cuvis_ai_core.utils.grpc_auth import MAX_MSG_SIZE_ENV, max_message_size

    server = _patch_main(monkeypatch, chosen_port=12345)
    seen = _record_server_kwargs(monkeypatch, server)
    if cap is None:
        monkeypatch.delenv(MAX_MSG_SIZE_ENV, raising=False)
    else:
        monkeypatch.setenv(MAX_MSG_SIZE_ENV, cap)

    assert main_mod.main(["--endpoint-file", str(tmp_path / "ep.txt")]) == 0

    options = dict(seen["options"])
    # The parent forwards a client's message with a resolved path added, hence the margin.
    assert options["grpc.max_receive_message_length"] == (
        max_message_size() + main_mod.FORWARD_MARGIN
    )
    assert max_message_size() == (300 * 1024 * 1024 if cap is None else 1000)
    # Inference answers (full frames and score maps) can be large; sending stays open.
    assert options["grpc.max_send_message_length"] == -1


def test_main_keeps_an_unlimited_parent_setting_unlimited(monkeypatch, tmp_path):
    from cuvis_ai_core.utils.grpc_auth import MAX_MSG_SIZE_ENV

    server = _patch_main(monkeypatch, chosen_port=12345)
    seen = _record_server_kwargs(monkeypatch, server)
    monkeypatch.setenv(MAX_MSG_SIZE_ENV, "-1")  # gRPC's spelling of "no limit"

    assert main_mod.main(["--endpoint-file", str(tmp_path / "ep.txt")]) == 0

    assert dict(seen["options"])["grpc.max_receive_message_length"] == -1


def test_main_checks_the_token_the_parent_hands_over(monkeypatch, tmp_path):
    from cuvis_ai_core.utils.grpc_auth import RUNTIME_TOKEN_ENV, TokenInterceptor

    server = _patch_main(monkeypatch, chosen_port=12345)
    seen = _record_server_kwargs(monkeypatch, server)
    monkeypatch.setenv(RUNTIME_TOKEN_ENV, "child-token")

    rc = main_mod.main(["--endpoint-file", str(tmp_path / "ep.txt")])

    assert rc == 0
    (interceptor,) = seen["interceptors"]
    assert isinstance(interceptor, TokenInterceptor)
    # Plugin code runs in this process; it does not get to read the token.
    assert RUNTIME_TOKEN_ENV not in os.environ


def test_main_started_by_hand_without_a_token_checks_nothing(monkeypatch, tmp_path):
    from cuvis_ai_core.utils.grpc_auth import RUNTIME_TOKEN_ENV

    server = _patch_main(monkeypatch, chosen_port=12345)
    seen = _record_server_kwargs(monkeypatch, server)
    monkeypatch.delenv(RUNTIME_TOKEN_ENV, raising=False)

    rc = main_mod.main(["--endpoint-file", str(tmp_path / "ep.txt")])

    assert rc == 0
    assert not seen.get("interceptors")
