"""Shared gRPC error handling helpers.

Centralises the session-lookup, pipeline-precondition, and exception-to-status
patterns that repeat across every service file.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any, get_type_hints

import grpc
from loguru import logger
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .session_manager import SessionManager, SessionState


def get_session_or_error(
    session_manager: SessionManager,
    session_id: str,
    context: grpc.ServicerContext,
) -> SessionState | None:
    """Look up a session, setting NOT_FOUND on the gRPC context if missing.

    Returns the session on success or ``None`` when the caller should
    return an empty / failure response immediately.
    """
    try:
        return session_manager.get_session(session_id)
    except ValueError as exc:
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(str(exc))
        return None


def require_pipeline(
    session: SessionState,
    context: grpc.ServicerContext,
) -> bool:
    """Assert that *session* has a pipeline, otherwise set FAILED_PRECONDITION.

    Returns ``True`` when a pipeline exists, ``False`` when the caller
    should return an empty / failure response.
    """
    if session.pipeline is not None:
        return True
    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
    context.set_details(
        "No pipeline is available for this session. Build pipeline first."
    )
    return False


# Ordered from most specific to least specific.  The first matching
# exception type wins, so ``Exception`` must come last.
_EXCEPTION_STATUS_MAP: list[tuple[type[Exception], grpc.StatusCode]] = [
    (FileNotFoundError, grpc.StatusCode.NOT_FOUND),
    (json.JSONDecodeError, grpc.StatusCode.INVALID_ARGUMENT),
    (ValidationError, grpc.StatusCode.INVALID_ARGUMENT),
    (ValueError, grpc.StatusCode.INVALID_ARGUMENT),
]


def grpc_handler(error_prefix: str = "") -> Callable:
    """Decorator that catches exceptions and maps them to gRPC status codes.

    Exception mapping (checked in order):

    * ``FileNotFoundError``    → ``NOT_FOUND``
    * ``json.JSONDecodeError`` → ``INVALID_ARGUMENT``
    * ``ValidationError``      → ``INVALID_ARGUMENT``
    * ``ValueError``           → ``INVALID_ARGUMENT``
    * ``Exception`` (fallback) → ``INTERNAL`` (with *error_prefix*)

    The response type is inferred from the method's return annotation
    and an empty instance is returned on error.

    Usage::

        @grpc_handler("Failed to load pipeline")
        def load_pipeline(self, request, context):
            ...
    """

    def decorator(method: Callable) -> Callable:
        # Resolve the return type once at decoration time so we can
        # instantiate an empty proto response on error.
        hints = get_type_hints(method)
        response_type: type | None = hints.get("return")

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return method(*args, **kwargs)
            except tuple(exc for exc, _ in _EXCEPTION_STATUS_MAP) as exc:
                # Find the matching status code.
                context = _extract_context(args, kwargs)
                for exc_type, status_code in _EXCEPTION_STATUS_MAP:
                    if isinstance(exc, exc_type):
                        context.set_code(status_code)
                        context.set_details(str(exc))
                        break
            except Exception as exc:
                context = _extract_context(args, kwargs)
                context.set_code(grpc.StatusCode.INTERNAL)
                prefix = f"{error_prefix}: " if error_prefix else ""
                context.set_details(f"{prefix}{exc}")
                logger.opt(exception=True).error(
                    f"gRPC {method.__name__}: {prefix}{exc}"
                )

            return response_type() if response_type is not None else None

        return wrapper

    return decorator


def _extract_context(args: tuple, kwargs: dict) -> grpc.ServicerContext:
    """Pull the gRPC *context* from positional or keyword arguments.

    Service methods have the signature ``(self, request, context)`` so
    *context* is always ``args[2]`` or ``kwargs["context"]``.
    """
    if "context" in kwargs:
        return kwargs["context"]
    # args = (self, request, context)
    return args[2]


__all__ = ["get_session_or_error", "grpc_handler", "require_pipeline"]
