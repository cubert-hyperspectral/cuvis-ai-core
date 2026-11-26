"""Shared-secret token and message size cap for the gRPC servers.

The server checks a token when its launcher sets one (``CUVIS_GRPC_TOKEN``), and every child
runtime checks the per-spawn token its parent hands it (``CUVIS_RUNTIME_TOKEN``). The token
travels as ``authorization`` metadata, ``Bearer <token>`` or the bare value. gRPC call
credentials are not used: they are dropped on insecure channels, and a loopback channel is
insecure. A call without the token is refused before its request is parsed.
"""

from __future__ import annotations

import collections
import hmac
import os
from collections.abc import Callable
from typing import Any

import grpc

TOKEN_ENV = "CUVIS_GRPC_TOKEN"
RUNTIME_TOKEN_ENV = "CUVIS_RUNTIME_TOKEN"
MAX_MSG_SIZE_ENV = "GRPC_MAX_MSG_SIZE"
DEFAULT_MAX_MSG_SIZE = 300 * 1024 * 1024

_METADATA_KEY = "authorization"
_BEARER = "bearer "
# Health stays open: launchers poll it before they hold a channel with the token.
_OPEN_PREFIXES = ("/grpc.health.v1.Health/",)
_REFUSAL = "missing or wrong token"


def normalize_token(value: str | None) -> str | None:
    """The token with surrounding whitespace removed; ``None`` when nothing is left."""
    if value is None:
        return None
    return value.strip() or None


def pop_token(name: str) -> str | None:
    """Read the token from environment variable ``name`` and remove the variable.

    Removing it keeps the token away from every process this one starts later
    (uv, plugin builds, child runtimes), which copy the environment.
    """
    return normalize_token(os.environ.pop(name, None))


def max_message_size() -> int:
    """The largest message a client may send the server, in bytes (``GRPC_MAX_MSG_SIZE``)."""
    return int(os.environ.get(MAX_MSG_SIZE_ENV, DEFAULT_MAX_MSG_SIZE))


def call_metadata(token: str) -> list[tuple[str, str]]:
    """The metadata a client sends with every call."""
    return [(_METADATA_KEY, f"Bearer {token}")]


def _presented_token(metadata: Any) -> str | None:
    for key, value in metadata or ():
        if key.lower() != _METADATA_KEY or not isinstance(value, str):
            continue
        if value[: len(_BEARER)].lower() == _BEARER:
            value = value[len(_BEARER) :]
        return value.strip()
    return None


def _refusing(handler: grpc.RpcMethodHandler) -> grpc.RpcMethodHandler:
    """A handler that answers UNAUTHENTICATED without reading the request.

    It takes the request as a stream it never reads, whatever ``handler`` expects: gRPC reads
    a unary request in full before calling the handler, so a caller without the token would
    otherwise make the server receive its whole message first.
    """

    def refuse(_requests: Any, context: grpc.ServicerContext) -> Any:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, _REFUSAL)

    codecs = {
        "request_deserializer": None,
        "response_serializer": handler.response_serializer,
    }
    if handler.response_streaming:
        return grpc.stream_stream_rpc_method_handler(refuse, **codecs)
    return grpc.stream_unary_rpc_method_handler(refuse, **codecs)


class TokenInterceptor(grpc.ServerInterceptor):
    """Refuse every call that does not carry the token, Health excepted."""

    def __init__(self, token: str) -> None:
        token = normalize_token(token) or ""
        if not token:
            raise ValueError("TokenInterceptor needs a non-empty token")
        self._expected = token.encode("utf-8")

    def _accepts(self, metadata: Any) -> bool:
        presented = _presented_token(metadata)
        if presented is None:
            return False
        return hmac.compare_digest(presented.encode("utf-8"), self._expected)

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler | None],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        """Pass an authorized call (or a Health call) on; answer any other one UNAUTHENTICATED."""
        handler = continuation(handler_call_details)
        if handler is None or handler_call_details.method.startswith(_OPEN_PREFIXES):
            return handler
        if self._accepts(handler_call_details.invocation_metadata):
            return handler
        return _refusing(handler)


class _CallDetails(
    collections.namedtuple(
        "_CallDetails",
        (
            "method",
            "timeout",
            "metadata",
            "credentials",
            "wait_for_ready",
            "compression",
        ),
    ),
    grpc.ClientCallDetails,
):
    pass


class _TokenClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """Add the token to the metadata of every call on the channel."""

    def __init__(self, token: str) -> None:
        self._metadata = call_metadata(token)

    def _with_token(self, details: grpc.ClientCallDetails) -> _CallDetails:
        metadata = [*(details.metadata or ()), *self._metadata]
        return _CallDetails(
            details.method,
            details.timeout,
            metadata,
            details.credentials,
            getattr(details, "wait_for_ready", None),
            getattr(details, "compression", None),
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._with_token(client_call_details), request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._with_token(client_call_details), request)

    def intercept_stream_unary(
        self, continuation, client_call_details, request_iterator
    ):
        return continuation(self._with_token(client_call_details), request_iterator)

    def intercept_stream_stream(
        self, continuation, client_call_details, request_iterator
    ):
        return continuation(self._with_token(client_call_details), request_iterator)


def authorized_channel(channel: grpc.Channel, token: str) -> grpc.Channel:
    """``channel`` with the token added to every call made on it."""
    return grpc.intercept_channel(channel, _TokenClientInterceptor(token))
