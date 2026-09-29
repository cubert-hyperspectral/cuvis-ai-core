"""The schema handler's generic failure goes through the shared INTERNAL fallback."""

from __future__ import annotations

from unittest.mock import Mock

import grpc

from cuvis_ai_core.grpc import config_service
from cuvis_ai_core.grpc.config_service import ConfigService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def test_schema_generation_failure_is_internal_with_the_handler_prefix(monkeypatch):
    monkeypatch.setattr(
        config_service, "generate_json_schema", Mock(side_effect=RuntimeError("boom"))
    )
    context = Mock(spec=grpc.ServicerContext)

    response = ConfigService(SessionManager()).get_parameter_schema(
        cuvis_ai_pb2.GetParameterSchemaRequest(config_type="training"), context
    )

    assert response == cuvis_ai_pb2.GetParameterSchemaResponse()
    context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    context.set_details.assert_called_once_with("Failed to get parameter schema: boom")


def test_unknown_config_type_is_not_found(monkeypatch):
    monkeypatch.setattr(
        config_service,
        "generate_json_schema",
        Mock(side_effect=ValueError("no such type")),
    )
    context = Mock(spec=grpc.ServicerContext)

    response = ConfigService(SessionManager()).get_parameter_schema(
        cuvis_ai_pb2.GetParameterSchemaRequest(config_type="nope"), context
    )

    assert response == cuvis_ai_pb2.GetParameterSchemaResponse()
    context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
    context.set_details.assert_called_once_with("no such type")
