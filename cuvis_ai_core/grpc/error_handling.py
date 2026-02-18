"""Shared gRPC error handling helpers.

Centralises the session-lookup and pipeline-precondition patterns that
repeat across every service file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

if TYPE_CHECKING:
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


__all__ = ["get_session_or_error", "require_pipeline"]
