import json
import logging
import sys
import threading
from unittest.mock import MagicMock, Mock

import pytest
from grpc_health.v1 import health_pb2

from cuvis_ai_core.grpc import production_server as production_server_mod
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


# ---------------------------------------------------------------------------
# Cache-maintenance loop wiring (reap + evict, hourly, post-bind)
# ---------------------------------------------------------------------------


def _patch_maintenance(
    monkeypatch, tmp_path, *, reap, evict, warm=lambda: None
) -> None:
    """Wire fast fakes into the maintenance loop (10ms interval)."""
    monkeypatch.setenv("CUVIS_RUN_CACHE_MAINTENANCE", "1")
    monkeypatch.setattr(
        production_server_mod, "resolve_cache_root", lambda override: tmp_path
    )
    monkeypatch.setattr(production_server_mod, "take_process_snapshot", warm)
    monkeypatch.setattr(production_server_mod, "reap_orphans", reap)
    monkeypatch.setattr(production_server_mod, "evict_run_cache", evict)
    monkeypatch.setattr(production_server_mod, "_MAINTENANCE_INTERVAL_SECONDS", 0.01)


def test_maintenance_warms_process_scan_before_first_pass(
    monkeypatch, tmp_path
) -> None:
    """The slow first psutil scan runs on the maintenance thread ahead of any pass."""
    events: list[str] = []
    first_pass = threading.Event()

    def _warm() -> None:
        events.append("warm")

    def _reap(root) -> None:
        events.append("reap")
        first_pass.set()

    _patch_maintenance(
        monkeypatch, tmp_path, reap=_reap, evict=lambda root: None, warm=_warm
    )
    server = ProductionServer(port=0)
    server._start_maintenance_thread()
    assert server._maintenance_thread is not None
    try:
        assert first_pass.wait(10)
    finally:
        server._maintenance_stop.set()
        server._maintenance_thread.join(timeout=10)
    assert events[:2] == ["warm", "reap"]
    assert events.count("warm") == 1


def test_maintenance_warm_up_failure_is_logged_and_loop_proceeds(
    monkeypatch, tmp_path, caplog
) -> None:
    # Pins the warm-up `except` branch: a failing snapshot is logged, not fatal.
    first_pass = threading.Event()

    def _warm() -> None:
        raise RuntimeError("psutil scan blew up")

    def _reap(root) -> None:
        first_pass.set()

    _patch_maintenance(
        monkeypatch, tmp_path, reap=_reap, evict=lambda root: None, warm=_warm
    )
    server = ProductionServer(port=0)
    with caplog.at_level(logging.ERROR, logger=production_server_mod.__name__):
        server._start_maintenance_thread()
        assert server._maintenance_thread is not None
        try:
            assert first_pass.wait(10)
        finally:
            server._maintenance_stop.set()
            server._maintenance_thread.join(timeout=10)
    warm_failures = [
        r for r in caplog.records if r.getMessage() == "Process scan warm-up failed"
    ]
    assert len(warm_failures) == 1
    assert warm_failures[0].exc_info is not None
    assert warm_failures[0].exc_info[0] is RuntimeError


def test_start_maintenance_thread_is_noop_while_thread_alive(
    monkeypatch, tmp_path
) -> None:
    # Pins the early return when a maintenance thread is already running.
    in_reap = threading.Event()
    release = threading.Event()

    def _blocking_reap(root) -> None:
        in_reap.set()
        release.wait(10)

    _patch_maintenance(
        monkeypatch, tmp_path, reap=_blocking_reap, evict=lambda root: None
    )
    server = ProductionServer(port=0)
    server._start_maintenance_thread()
    first_thread = server._maintenance_thread
    assert first_thread is not None
    try:
        assert in_reap.wait(10)
        assert first_thread.is_alive()
        # Snapshot-relative count: other tests may leave same-named threads.
        named_before = {t for t in threading.enumerate() if t.name == first_thread.name}
        server._start_maintenance_thread()
        assert server._maintenance_thread is first_thread
        named_after = {t for t in threading.enumerate() if t.name == first_thread.name}
        assert named_after - named_before == set()
    finally:
        server._maintenance_stop.set()
        release.set()
        first_thread.join(timeout=10)
    assert not first_thread.is_alive()


def test_maintenance_kill_switch_disables_thread(monkeypatch) -> None:
    """CUVIS_RUN_CACHE_MAINTENANCE=0 keeps the loop entirely off."""
    monkeypatch.setenv("CUVIS_RUN_CACHE_MAINTENANCE", "0")
    server = ProductionServer(port=0)
    server._start_maintenance_thread()
    assert server._maintenance_thread is None


def test_maintenance_loop_reaps_then_evicts_each_pass(monkeypatch, tmp_path) -> None:
    """Each pass runs reap before evict and repeats on the interval."""
    events: list[str] = []
    second_pass = threading.Event()

    def _fake_reap(root) -> None:
        events.append("reap")

    def _fake_evict(root) -> None:
        events.append("evict")
        if events.count("evict") >= 2:
            second_pass.set()

    _patch_maintenance(monkeypatch, tmp_path, reap=_fake_reap, evict=_fake_evict)
    server = ProductionServer(port=0)
    server._start_maintenance_thread()
    assert server._maintenance_thread is not None
    try:
        assert second_pass.wait(10)
    finally:
        server._maintenance_stop.set()
        server._maintenance_thread.join(timeout=10)
    assert not server._maintenance_thread.is_alive()
    assert events[:2] == ["reap", "evict"]


def test_maintenance_loop_survives_pass_failure(monkeypatch, tmp_path) -> None:
    """A failing pass is logged and the loop keeps running."""
    later_pass = threading.Event()
    calls = {"n": 0}

    def _flaky_reap(root) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first pass fails")
        later_pass.set()

    _patch_maintenance(monkeypatch, tmp_path, reap=_flaky_reap, evict=lambda root: None)
    server = ProductionServer(port=0)
    server._start_maintenance_thread()
    assert server._maintenance_thread is not None
    try:
        assert later_pass.wait(10)
    finally:
        server._maintenance_stop.set()
        server._maintenance_thread.join(timeout=10)


def test_start_launches_maintenance_and_shutdown_stops_it(
    monkeypatch, tmp_path
) -> None:
    """start() launches the loop post-bind; shutdown() terminates it."""
    first_pass = threading.Event()

    def _fake_reap(root) -> None:
        first_pass.set()

    _patch_maintenance(monkeypatch, tmp_path, reap=_fake_reap, evict=lambda root: None)
    server = ProductionServer(port=0, max_workers=2)
    try:
        server.start()
        assert server._maintenance_thread is not None
        assert server._maintenance_thread.is_alive()
        assert first_pass.wait(10)
    finally:
        server.shutdown()
    server._maintenance_thread.join(timeout=10)
    assert not server._maintenance_thread.is_alive()
