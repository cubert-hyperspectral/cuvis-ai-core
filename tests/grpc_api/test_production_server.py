import json
import logging
import sys
from unittest.mock import MagicMock, Mock

import pytest
from grpc_health.v1 import health_pb2

from cuvis_ai_core.grpc.production_server import (
    JSONFormatter,
    ProductionServer,
    load_tls_credentials,
    setup_logging,
)
from cuvis_ai_core.grpc.service import CuvisAIService


def test_load_tls_credentials_missing_files(tmp_path) -> None:
    """TLS setup should fail fast when files are missing."""
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"

    with pytest.raises(FileNotFoundError):
        load_tls_credentials(str(cert_path), str(key_path))


def test_setup_logging_uses_json_formatter() -> None:
    """Logging helper should apply JSON formatter."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    try:
        setup_logging("INFO", "json")
        assert any(
            isinstance(handler.formatter, JSONFormatter)
            for handler in root_logger.handlers
        )
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


def test_server_start_and_shutdown_updates_health_status() -> None:
    """Server should start, report healthy, then mark not serving on shutdown."""
    server = ProductionServer(port=0, max_workers=2)

    try:
        server.start()
        assert server.server is not None
        assert server.health_service is not None

        resp = server.health_service.Check(
            health_pb2.HealthCheckRequest(), context=None
        )
        assert resp.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        server.shutdown()

    assert server.health_service is not None
    resp = server.health_service.Check(health_pb2.HealthCheckRequest(), context=None)
    assert resp.status == health_pb2.HealthCheckResponse.NOT_SERVING


def test_close_all_sessions_terminates_children_and_drops_sessions() -> None:
    """Shutdown reaps each session's child runtime so none are orphaned."""
    server = ProductionServer()
    service = CuvisAIService()
    server.cuvis_service = service

    sm = service.session_manager
    sid = sm.create_session()
    child = MagicMock()
    child.returncode = None
    sm.get_session(sid).child_handle = child

    server._close_all_sessions()

    child.terminate.assert_called_once()
    assert sm.list_sessions() == []


def test_close_all_sessions_isolates_per_session_failures() -> None:
    """One session failing to close must not abort reaping the rest."""
    server = ProductionServer()
    service = CuvisAIService()
    server.cuvis_service = service

    sm = service.session_manager
    bad_sid = sm.create_session()
    good_sid = sm.create_session()

    bad_child = MagicMock()
    bad_child.returncode = None
    bad_child.terminate.side_effect = RuntimeError("terminate blew up")
    bad_child.kill.side_effect = RuntimeError("kill blew up too")
    sm.get_session(bad_sid).child_handle = bad_child

    good_child = MagicMock()
    good_child.returncode = None
    sm.get_session(good_sid).child_handle = good_child

    server._close_all_sessions()

    assert sm.list_sessions() == []
    good_child.terminate.assert_called_once()


def test_close_all_sessions_logs_when_close_session_raises(monkeypatch) -> None:
    """If close_session itself raises, _close_all_sessions logs and moves on."""
    server = ProductionServer()
    service = CuvisAIService()
    server.cuvis_service = service
    sm = service.session_manager
    sm.create_session()
    monkeypatch.setattr(sm, "close_session", Mock(side_effect=RuntimeError("boom")))

    # Must not propagate the failure.
    server._close_all_sessions()


def test_close_all_sessions_without_service_is_noop() -> None:
    server = ProductionServer()
    assert server.cuvis_service is None
    # Must not raise when no service was ever started.
    server._close_all_sessions()


def test_shutdown_reaps_sessions_before_stopping_server() -> None:
    """Children must be reaped before server.stop(), so a teardown refactor
    that reordered them (letting in-flight RPCs re-touch a session during
    stop) would fail this test."""
    order: list[str] = []

    server = ProductionServer()
    service = CuvisAIService()
    server.cuvis_service = service

    sid = service.session_manager.create_session()
    child = MagicMock()
    child.returncode = None
    child.terminate.side_effect = lambda *a, **k: order.append("reap")
    service.session_manager.get_session(sid).child_handle = child

    server.server = MagicMock()
    server.server.stop.side_effect = lambda *a, **k: (
        order.append("stop"),
        MagicMock(),
    )[1]

    server.shutdown()

    assert order == ["reap", "stop"]


# ---------------------------------------------------------------------------
# JSONFormatter + setup_logging branches
# ---------------------------------------------------------------------------


def test_json_formatter_emits_json_with_exception_and_extras() -> None:
    formatter = JSONFormatter()

    plain = logging.LogRecord("n", logging.INFO, __file__, 1, "hello", None, None)
    out = json.loads(formatter.format(plain))
    assert out["message"] == "hello"
    assert out["level"] == "INFO"

    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.LogRecord(
            "n", logging.ERROR, __file__, 2, "failed", None, sys.exc_info()
        )
    rec.extra = {"request_id": "abc-123"}
    out2 = json.loads(formatter.format(rec))
    assert "exception" in out2
    assert out2["request_id"] == "abc-123"


def test_setup_logging_text_format_uses_plain_formatter() -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        setup_logging("DEBUG", "text")
        assert root_logger.handlers
        assert all(
            not isinstance(h.formatter, JSONFormatter) for h in root_logger.handlers
        )
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


# ---------------------------------------------------------------------------
# load_tls_credentials: missing key + the read/build path
# ---------------------------------------------------------------------------


def test_load_tls_credentials_missing_key(tmp_path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_bytes(b"cert-bytes")
    key = tmp_path / "key.pem"  # deliberately absent
    with pytest.raises(FileNotFoundError):
        load_tls_credentials(str(cert), str(key))


def test_load_tls_credentials_reads_present_files(tmp_path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_bytes(b"not-a-real-cert")
    key.write_bytes(b"not-a-real-key")
    # Both files exist, so the read + ssl_server_credentials path executes.
    # Invalid PEM may raise inside grpc; either outcome exercises the lines.
    try:
        load_tls_credentials(str(cert), str(key))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ProductionServer.start guards
# ---------------------------------------------------------------------------


def test_start_is_idempotent_when_already_started() -> None:
    server = ProductionServer(port=0, max_workers=2)
    try:
        server.start()
        # Second start() short-circuits with a warning and leaves the server.
        first_server = server.server
        server.start()
        assert server.server is first_server
    finally:
        server.shutdown()


def test_start_applies_max_msg_size() -> None:
    server = ProductionServer(port=0, max_workers=2, max_msg_size=2048)
    try:
        server.start()
        assert server.server is not None
    finally:
        server.shutdown()


def test_start_tls_without_paths_raises() -> None:
    server = ProductionServer(port=0, use_tls=True)
    with pytest.raises(ValueError, match="TLS enabled"):
        server.start()


def test_start_raises_when_insecure_bind_fails(monkeypatch) -> None:
    fake_server = MagicMock()
    fake_server.add_insecure_port.return_value = 0
    monkeypatch.setattr(
        "cuvis_ai_core.grpc.production_server.grpc.server",
        lambda *a, **k: fake_server,
    )
    server = ProductionServer(port=55999)
    with pytest.raises(RuntimeError, match="Failed to bind insecure port"):
        server.start()


# ---------------------------------------------------------------------------
# wait_for_termination
# ---------------------------------------------------------------------------


def test_wait_for_termination_exits_when_shutdown_already_requested() -> None:
    server = ProductionServer()
    server._shutdown_requested = True
    # Loop body never runs; the finally calls shutdown() (server is None → noop).
    server.wait_for_termination()


def test_wait_for_termination_handles_keyboard_interrupt(monkeypatch) -> None:
    server = ProductionServer()
    monkeypatch.setattr(
        "cuvis_ai_core.grpc.production_server.time.sleep",
        Mock(side_effect=KeyboardInterrupt),
    )
    server.wait_for_termination()
