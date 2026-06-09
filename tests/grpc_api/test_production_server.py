import logging
from unittest.mock import MagicMock

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
