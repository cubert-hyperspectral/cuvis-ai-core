"""The health port: Check without a token, nothing else, and nothing a client can stall."""

from __future__ import annotations

import threading

import grpc
import pytest
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc

from cuvis_ai_core.grpc.health import HealthService, start_health_listener

CHECK = "/grpc.health.v1.Health/Check"
SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
_CREATE_SESSION = f"/{cuvis_ai_pb2.DESCRIPTOR.services_by_name['CuvisAIService'].full_name}/CreateSession"


@pytest.fixture
def listener():
    health = HealthService()
    server, port = start_health_listener(health, "127.0.0.1:0")
    yield health, f"127.0.0.1:{port}"
    server.stop(grace=None)


def test_the_health_port_answers_without_a_token(listener):
    health, endpoint = listener
    with grpc.insecure_channel(endpoint) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        serving = stub.Check(health_pb2.HealthCheckRequest(), timeout=10).status
        health.set_not_serving()
        stopping = stub.Check(health_pb2.HealthCheckRequest(), timeout=10).status

    assert (serving, stopping) == (SERVING, NOT_SERVING)


def test_a_check_whose_request_never_arrives_holds_up_nothing(listener):
    # A unary handler waits in its worker thread for the request; the health port has two
    # workers, so three withheld requests would starve a probe if it waited.
    _, endpoint = listener
    release = threading.Event()

    def withheld():
        release.wait(30)
        yield b""

    with grpc.insecure_channel(endpoint) as channel:
        stalled_call = channel.stream_unary(
            CHECK, response_deserializer=health_pb2.HealthCheckResponse.FromString
        )
        try:
            stalled = [stalled_call.future(withheld(), timeout=30) for _ in range(3)]
            probe = health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(), timeout=2
            )
            answers = [call.result(timeout=5).status for call in stalled]
        finally:
            release.set()

    assert probe.status == SERVING
    assert answers == [SERVING] * 3


def test_an_oversized_request_is_answered_without_being_read(listener):
    # Never read, so the 1 KiB cap never sees it: nothing past the headers is buffered.
    _, endpoint = listener
    with grpc.insecure_channel(endpoint) as channel:
        call = channel.unary_unary(
            CHECK, response_deserializer=health_pb2.HealthCheckResponse.FromString
        )
        response = call(b"\x00" * (1 << 20), timeout=10)

    assert response.status == SERVING


@pytest.mark.parametrize(
    "path",
    ["/grpc.health.v1.Health/Watch", _CREATE_SESSION],
    ids=["Watch", "CreateSession"],
)
def test_the_health_port_serves_check_only(listener, path):
    _, endpoint = listener
    with grpc.insecure_channel(endpoint) as channel:
        with pytest.raises(grpc.RpcError) as refused:
            channel.unary_unary(path)(b"", timeout=10)

    assert refused.value.code() == grpc.StatusCode.UNIMPLEMENTED


def test_the_health_port_bounds_each_connection(monkeypatch):
    # Answered calls whose response a client never reads outlive maximum_concurrent_rpcs;
    # the stream and age limits are what free them.
    recorded: dict = {}
    real_server = grpc.server

    def spy(*args, **kwargs):
        recorded.update(kwargs)
        return real_server(*args, **kwargs)

    monkeypatch.setattr(grpc, "server", spy)
    server, _ = start_health_listener(HealthService(), "127.0.0.1:0")
    server.stop(grace=None)

    assert dict(recorded["options"]) == {
        "grpc.max_receive_message_length": 1024,
        "grpc.max_concurrent_streams": 2,
        "grpc.max_connection_age_ms": 60_000,
        "grpc.max_connection_age_grace_ms": 5_000,
    }
    assert recorded["maximum_concurrent_rpcs"] == 8
    assert not recorded.get("interceptors")
