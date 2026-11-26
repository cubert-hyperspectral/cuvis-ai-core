"""Shared-secret token on the gRPC servers: the interceptor, the client side and the server rules."""

from __future__ import annotations

import logging
import os
import threading
from concurrent import futures
from unittest.mock import MagicMock

import grpc
import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from google.protobuf.message_factory import GetMessageClass
from grpc_health.v1 import health_pb2, health_pb2_grpc

from cuvis_ai_core.utils import grpc_auth as auth
from cuvis_ai_core.grpc import production_server as production_server_mod
from cuvis_ai_core.utils.grpc_auth import (
    TokenInterceptor,
    authorized_channel,
    call_metadata,
)
from cuvis_ai_core.grpc.health import HealthService
from cuvis_ai_core.grpc.production_server import ProductionServer
from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

TOKEN = "s3cret-token-value"

_CUVIS = cuvis_ai_pb2.DESCRIPTOR.services_by_name["CuvisAIService"]
_RUNTIME = cuvis_ai_pb2.DESCRIPTOR.services_by_name["RunRuntime"]


def _start(add_servicers) -> tuple[grpc.Server, str]:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[TokenInterceptor(TOKEN)],
    )
    add_servicers(server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


@pytest.fixture(scope="module")
def cuvis_endpoint():
    def add(server):
        cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(CuvisAIService(), server)
        health = HealthService()
        health_pb2_grpc.add_HealthServicer_to_server(health, server)
        health.set_serving()

    server, endpoint = _start(add)
    yield endpoint
    server.stop(grace=None)


@pytest.fixture(scope="module")
def runtime_endpoint():
    def add(server):
        servicer = RunRuntimeServicer(shutdown_event=threading.Event())
        cuvis_ai_pb2_grpc.add_RunRuntimeServicer_to_server(servicer, server)

    server, endpoint = _start(add)
    yield endpoint
    server.stop(grace=None)


def _call(channel: grpc.Channel, service, method, metadata=None) -> grpc.StatusCode:
    """Call ``method`` with an empty request through a generic stub; return the status."""
    request_cls = GetMessageClass(method.input_type)
    path = f"/{service.full_name}/{method.name}"
    serialize = request_cls.SerializeToString
    kwargs = {"metadata": metadata, "timeout": 10}
    try:
        if method.client_streaming and method.server_streaming:
            list(
                channel.stream_stream(path, serialize, bytes)(
                    iter([request_cls()]), **kwargs
                )
            )
        elif method.client_streaming:
            channel.stream_unary(path, serialize, bytes)(
                iter([request_cls()]), **kwargs
            )
        elif method.server_streaming:
            list(channel.unary_stream(path, serialize, bytes)(request_cls(), **kwargs))
        else:
            channel.unary_unary(path, serialize, bytes)(request_cls(), **kwargs)
    except grpc.RpcError as exc:
        return exc.code()
    return grpc.StatusCode.OK


@pytest.mark.parametrize("method", list(_CUVIS.methods), ids=lambda m: m.name)
def test_every_cuvis_ai_method_refuses_a_call_without_the_token(cuvis_endpoint, method):
    with grpc.insecure_channel(cuvis_endpoint) as channel:
        assert _call(channel, _CUVIS, method) == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.parametrize("method", list(_RUNTIME.methods), ids=lambda m: m.name)
def test_every_runtime_method_refuses_a_call_without_the_token(
    runtime_endpoint, method
):
    with grpc.insecure_channel(runtime_endpoint) as channel:
        assert _call(channel, _RUNTIME, method) == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.parametrize(
    ("endpoint", "service", "method_name"),
    [
        ("cuvis_endpoint", _CUVIS, "CreateSession"),
        ("runtime_endpoint", _RUNTIME, "LoadPipelineWeights"),
    ],
)
def test_a_call_without_the_token_is_refused_before_its_request_is_parsed(
    request, endpoint, service, method_name
):
    # Wire type 7 does not exist, so parsing these bytes would fail with INTERNAL.
    path = f"/{service.full_name}/{method_name}"
    with grpc.insecure_channel(request.getfixturevalue(endpoint)) as channel:
        with pytest.raises(grpc.RpcError) as refused:
            channel.unary_unary(path, None, bytes)(b"\xff\xff\xff", timeout=10)

    assert refused.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.parametrize(
    "path",
    [f"/{_CUVIS.full_name}/CreateSession", "/grpc.health.v1.Health/Check"],
    ids=["CreateSession", "Health"],
)
def test_a_call_without_the_token_is_refused_before_its_body_is_read(path):
    # The server takes 1 KiB at most. Refused without being read, an oversized body gets
    # UNAUTHENTICATED; read first, it would get RESOURCE_EXHAUSTED, as the call with
    # the token shows.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[TokenInterceptor(TOKEN)],
        options=[("grpc.max_receive_message_length", 1024)],
    )
    cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(CuvisAIService(), server)
    health_pb2_grpc.add_HealthServicer_to_server(HealthService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    codes = []
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = channel.unary_unary(path, None, bytes)
            for metadata in (None, call_metadata(TOKEN)):
                with pytest.raises(grpc.RpcError) as refused:
                    call(b"\x00" * 65536, metadata=metadata, timeout=10)
                codes.append(refused.value.code())
    finally:
        server.stop(grace=None)

    assert codes == [
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        [("authorization", "Bearer wrong-token")],
        [("authorization", "wrong-token")],
        [("authorization", f"Bearer {TOKEN}x")],
        [("authorization", "")],
        [("x-other", TOKEN)],
    ],
    ids=["bearer-wrong", "raw-wrong", "longer", "empty", "other-key"],
)
def test_a_wrong_token_is_refused(cuvis_endpoint, metadata):
    method = _CUVIS.methods_by_name["CreateSession"]
    with grpc.insecure_channel(cuvis_endpoint) as channel:
        assert (
            _call(channel, _CUVIS, method, metadata) == grpc.StatusCode.UNAUTHENTICATED
        )


@pytest.mark.parametrize("value", [f"Bearer {TOKEN}", f"bearer {TOKEN}", TOKEN])
def test_the_right_token_reaches_the_servicer(cuvis_endpoint, value):
    with grpc.insecure_channel(cuvis_endpoint) as channel:
        stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
        metadata = [("authorization", value)]
        session = stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(), metadata=metadata
        )
        assert session.session_id
        stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=session.session_id),
            metadata=metadata,
        )


def test_health_on_the_main_port_needs_the_token(cuvis_endpoint):
    # Probes without the token use the health port (GRPC_HEALTH_PORT) instead.
    with grpc.insecure_channel(cuvis_endpoint) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        with pytest.raises(grpc.RpcError) as refused:
            stub.Check(health_pb2.HealthCheckRequest(), timeout=10)
        response = stub.Check(
            health_pb2.HealthCheckRequest(), metadata=call_metadata(TOKEN), timeout=10
        )

    assert refused.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert response.status == health_pb2.HealthCheckResponse.SERVING


def test_call_metadata_carries_a_bearer_token():
    assert call_metadata(TOKEN) == [("authorization", f"Bearer {TOKEN}")]


def test_authorized_channel_sends_the_token_on_unary_and_streaming_calls(
    runtime_endpoint,
):
    with grpc.insecure_channel(runtime_endpoint) as raw:
        channel = authorized_channel(raw, TOKEN)
        stub = cuvis_ai_pb2_grpc.RunRuntimeStub(channel)
        response = stub.HealthCheck(cuvis_ai_pb2.HealthCheckRequest(), timeout=10)
        assert (
            response.status == cuvis_ai_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
        )
        # A server-streaming call passes the check too; what the servicer then answers
        # for an empty request is its own business.
        status = _call(channel, _RUNTIME, _RUNTIME.methods_by_name["Train"])
        assert status != grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_empty_token_counts_as_unset(value):
    assert auth.normalize_token(value) is None


def test_token_interceptor_refuses_an_empty_token():
    with pytest.raises(ValueError):
        TokenInterceptor("")


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def test_pop_token_reads_and_removes_the_variable(monkeypatch):
    monkeypatch.setenv(auth.TOKEN_ENV, f"  {TOKEN}  ")

    assert auth.pop_token(auth.TOKEN_ENV) == TOKEN
    assert auth.TOKEN_ENV not in os.environ


def test_pop_token_of_an_empty_variable_is_none(monkeypatch):
    monkeypatch.setenv(auth.TOKEN_ENV, "")

    assert auth.pop_token(auth.TOKEN_ENV) is None


# ---------------------------------------------------------------------------
# ProductionServer
# ---------------------------------------------------------------------------


@pytest.fixture
def recorded_grpc_server(monkeypatch):
    """A grpc.server stand-in that records its keyword arguments and binds port 4321."""
    monkeypatch.setenv("CUVIS_RUN_CACHE_MAINTENANCE", "0")
    fake = MagicMock()
    fake.add_insecure_port.return_value = 4321
    fake.add_secure_port.return_value = 4321
    calls: list[dict] = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return fake

    monkeypatch.setattr("cuvis_ai_core.grpc.production_server.grpc.server", factory)
    monkeypatch.setattr(
        production_server_mod, "load_tls_credentials", lambda cert, key: "credentials"
    )
    fake.calls = calls
    return fake


def _started(server: ProductionServer) -> None:
    try:
        server.start()
    finally:
        server.shutdown()


def test_server_with_a_token_installs_the_interceptor(recorded_grpc_server):
    _started(ProductionServer(port=50551, token=TOKEN))

    (interceptor,) = recorded_grpc_server.calls[0]["interceptors"]
    assert isinstance(interceptor, TokenInterceptor)


def test_server_without_a_token_installs_no_interceptor(recorded_grpc_server):
    _started(ProductionServer(port=50551))

    assert not recorded_grpc_server.calls[0].get("interceptors")


def test_remote_bind_with_token_and_tls_starts(recorded_grpc_server, caplog):
    server = ProductionServer(
        port=50551,
        host="0.0.0.0",
        token=TOKEN,
        use_tls=True,
        tls_cert_path="cert.pem",
        tls_key_path="key.pem",
    )
    with caplog.at_level(logging.WARNING):
        _started(server)

    recorded_grpc_server.add_secure_port.assert_called_once_with(
        "0.0.0.0:50551", "credentials"
    )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_remote_bind_with_token_but_no_tls_is_refused(recorded_grpc_server):
    server = ProductionServer(port=50551, host="0.0.0.0", token=TOKEN)

    with pytest.raises(RuntimeError, match="GRPC_USE_TLS"):
        server.start()

    recorded_grpc_server.add_insecure_port.assert_not_called()


def test_remote_bind_with_tls_but_no_token_is_refused(recorded_grpc_server):
    server = ProductionServer(
        port=50551,
        host="0.0.0.0",
        use_tls=True,
        tls_cert_path="cert.pem",
        tls_key_path="key.pem",
    )

    with pytest.raises(RuntimeError, match="CUVIS_GRPC_TOKEN"):
        server.start()

    recorded_grpc_server.add_secure_port.assert_not_called()


def test_insecure_remote_override_still_starts_without_token_or_tls(
    recorded_grpc_server, caplog
):
    server = ProductionServer(port=50551, host="0.0.0.0", allow_insecure_remote=True)
    with caplog.at_level(logging.WARNING):
        _started(server)

    recorded_grpc_server.add_insecure_port.assert_called_once_with("0.0.0.0:50551")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_server_from_env_takes_the_token_out_of_the_environment(monkeypatch):
    monkeypatch.setenv("CUVIS_GRPC_TOKEN", TOKEN)

    server = production_server_mod.server_from_env()

    assert server.token == TOKEN
    assert "CUVIS_GRPC_TOKEN" not in os.environ


def test_server_from_env_without_a_token(monkeypatch):
    monkeypatch.delenv("CUVIS_GRPC_TOKEN", raising=False)

    assert production_server_mod.server_from_env().token is None


# Health port (GRPC_HEALTH_PORT)
# ---------------------------------------------------------------------------


def test_no_health_port_unless_one_is_set(recorded_grpc_server):
    _started(ProductionServer(port=50551, token=TOKEN))

    assert len(recorded_grpc_server.calls) == 1
    recorded_grpc_server.add_insecure_port.assert_called_once_with("localhost:50551")


def test_a_health_port_adds_a_listener_on_the_main_host(recorded_grpc_server):
    _started(ProductionServer(port=50551, host="127.0.0.1", health_port=50552))

    main, listener = recorded_grpc_server.calls
    assert listener["maximum_concurrent_rpcs"] == 8
    assert not listener.get("interceptors")
    assert [
        c.args[0] for c in recorded_grpc_server.add_insecure_port.call_args_list
    ] == [
        "127.0.0.1:50551",
        "127.0.0.1:50552",
    ]
    # Both servers stop at shutdown (one stand-in plays both).
    assert recorded_grpc_server.stop.call_count == 2


def test_the_health_port_may_not_be_the_main_port(recorded_grpc_server):
    with pytest.raises(ValueError, match="GRPC_HEALTH_PORT"):
        ProductionServer(port=50551, health_port=50551).start()

    assert recorded_grpc_server.calls == []


def test_a_health_port_that_cannot_bind_stops_the_main_server(recorded_grpc_server):
    recorded_grpc_server.add_insecure_port.side_effect = [4321, 0]
    server = ProductionServer(port=50551, health_port=50552)

    with pytest.raises(RuntimeError, match="GRPC_HEALTH_PORT"):
        server.start()

    recorded_grpc_server.start.assert_not_called()
    recorded_grpc_server.stop.assert_called()
    assert server.server is None


@pytest.mark.parametrize(
    ("value", "expected"), [("50552", 50552), ("", None), (None, None)]
)
def test_server_from_env_reads_the_health_port(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("GRPC_HEALTH_PORT", raising=False)
    else:
        monkeypatch.setenv("GRPC_HEALTH_PORT", value)

    assert production_server_mod.server_from_env().health_port == expected
