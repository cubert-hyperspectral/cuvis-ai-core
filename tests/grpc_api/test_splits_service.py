"""Tests for the ResolveSplits servicer (registry-mediated, plugin-agnostic)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest

from cuvis_ai_core.grpc.splits_service import SplitsService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


class _ResolvingModule:
    """Stand-in for a plugin DataModule exposing the resolve_splits hook."""

    last_config: dict | None = None

    @classmethod
    def resolve_splits(cls, config):
        cls.last_config = config
        payload = {"version": 1, "coco_hash_per_file": {}, "files": {"/a.cu3s": {}}}
        return payload, (
            f"{config['workspace_path']}/splits.json" if config["write"] else None
        )


class _PlainModule:
    """A DataModule without split-resolution support."""


def _service(data_modules):
    session = SimpleNamespace(node_registry=SimpleNamespace(data_modules=data_modules))
    manager = Mock()
    manager.get_session = Mock(return_value=session)
    return SplitsService(manager)


def _request(**overrides):
    cfg = {"workspace_path": "/ws", "data_module": "cu3s_workspace", "write": True}
    cfg.update(overrides)
    return cuvis_ai_pb2.ResolveSplitsRequest(
        session_id="s1", config_bytes=json.dumps(cfg).encode("utf-8")
    )


def test_resolve_splits_delegates_to_registry_hook():
    service = _service({"cu3s_workspace": _ResolvingModule})
    context = Mock()
    response = service.resolve_splits(_request(strategy="stratified", seed=7), context)
    assert response.splits_path == "/ws/splits.json"
    payload = json.loads(response.splits_bytes)
    assert payload["version"] == 1 and "/a.cu3s" in payload["files"]
    # the full SplitsResolveConfig (with defaults applied) reaches the plugin
    assert _ResolvingModule.last_config["strategy"] == "stratified"
    assert _ResolvingModule.last_config["seed"] == 7
    assert _ResolvingModule.last_config["train_ratio"] == pytest.approx(0.70)
    context.set_code.assert_not_called()


def test_write_false_returns_empty_path():
    service = _service({"cu3s_workspace": _ResolvingModule})
    response = service.resolve_splits(_request(write=False), Mock())
    assert response.splits_path == ""
    assert json.loads(response.splits_bytes)["files"]


def test_unknown_data_module_is_invalid_argument():
    service = _service({})
    context = Mock()
    service.resolve_splits(_request(), context)
    context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)


def test_module_without_hook_is_unimplemented():
    service = _service({"cu3s_workspace": _PlainModule})
    context = Mock()
    service.resolve_splits(_request(), context)
    context.set_code.assert_called_with(grpc.StatusCode.UNIMPLEMENTED)


def test_plugin_valueerror_maps_to_invalid_argument():
    class _Exploding:
        @classmethod
        def resolve_splits(cls, config):
            raise ValueError("workspace has no measurements")

    service = _service({"cu3s_workspace": _Exploding})
    context = Mock()
    response = service.resolve_splits(_request(), context)
    context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
    assert response.splits_bytes == b""


def test_missing_session_returns_empty():
    manager = Mock()
    manager.get_session = Mock(side_effect=ValueError("no such session"))
    service = SplitsService(manager)
    context = Mock()
    response = service.resolve_splits(_request(), context)
    context.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)
    assert response.splits_bytes == b""


def test_service_delegation_wires_resolve_splits():
    from cuvis_ai_core.grpc.service import CuvisAIService

    assert hasattr(CuvisAIService, "ResolveSplits")
