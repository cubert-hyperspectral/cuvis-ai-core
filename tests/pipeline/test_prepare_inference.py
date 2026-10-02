"""``prepare_inference``: the hook a node gets once its device and weights are final."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.grpc import pipeline_service as pipeline_service_module
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


def _record_prepare(pipeline: CuvisPipeline, calls: list[str]) -> None:
    for node in pipeline.nodes:
        node.prepare_inference = lambda name=node.name: calls.append(name)


class TestNodeHook:
    def test_default_is_a_no_op(self, simple_input_node):
        assert simple_input_node().prepare_inference() is None


class TestPipelinePrepareInference:
    def test_calls_every_node_once_in_graph_order(self, simple_three_node_pipeline):
        calls: list[str] = []
        _record_prepare(simple_three_node_pipeline, calls)

        simple_three_node_pipeline.prepare_inference()

        assert calls == [node.name for node in simple_three_node_pipeline._sorted_nodes]
        assert len(calls) == 3

    def test_failure_names_the_node_and_keeps_the_cause(
        self, simple_three_node_pipeline
    ):
        failing = simple_three_node_pipeline._sorted_nodes[1]
        cause = FileNotFoundError("engine missing")

        def _raise() -> None:
            raise cause

        failing.prepare_inference = _raise

        with pytest.raises(
            RuntimeError, match=f"'{failing.name}'.*engine missing"
        ) as info:
            simple_three_node_pipeline.prepare_inference()
        assert info.value.__cause__ is cause


class TestLoadPipelineClassmethod:
    """``CuvisPipeline.load_pipeline`` serves restore-pipeline and RestoreTrainRun."""

    @pytest.fixture
    def config_path(self, tmp_path, minimal_pipeline_dict) -> Path:
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump(minimal_pipeline_dict), encoding="utf-8")
        return path

    @pytest.fixture
    def order(self, monkeypatch) -> list[str]:
        order: list[str] = []
        monkeypatch.setattr(
            CuvisPipeline,
            "to",
            lambda self, device: order.append(f"to:{device}") or self,
        )
        monkeypatch.setattr(
            CuvisPipeline,
            "_restore_weights_from_checkpoint",
            lambda self, **kwargs: order.append("weights") or [],
        )
        monkeypatch.setattr(
            CuvisPipeline, "prepare_inference", lambda self: order.append("prepare")
        )
        return order

    def test_prepares_after_device_and_weights(self, config_path, order):
        CuvisPipeline.load_pipeline(config_path, weights_path="fitted.pt", device="cpu")
        assert order == ["to:cpu", "weights", "prepare"]

    def test_prepares_after_device_without_weights(self, config_path, order):
        CuvisPipeline.load_pipeline(config_path, device="cpu")
        assert order == ["to:cpu", "prepare"]


class TestPipelineServiceCalls:
    """LoadPipeline and LoadPipelineWeights prepare the pipeline as their last step."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = PipelineService(self.session_manager)
        self.ctx = Mock()
        self.order: list[str] = []
        self.pipeline = Mock()
        self.pipeline.to.side_effect = (
            lambda device: self.order.append(f"to:{device}") or self.pipeline
        )
        self.pipeline._restore_weights_from_checkpoint.side_effect = (
            lambda **kwargs: self.order.append("weights") or []
        )
        self.pipeline.prepare_inference.side_effect = lambda: self.order.append(
            "prepare"
        )

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def _load_request(self, session_id: str, minimal_pipeline_dict: dict):
        config = dict(minimal_pipeline_dict)
        config.pop("version", None)
        return cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=json.dumps(config).encode()
            ),
        )

    @pytest.fixture
    def built(self, monkeypatch):
        builder = Mock()
        builder.return_value.build_from_config.return_value = self.pipeline
        monkeypatch.setattr(pipeline_service_module, "PipelineBuilder", builder)
        monkeypatch.setattr(
            pipeline_service_module.torch.cuda, "is_available", lambda: True
        )

    def test_load_pipeline_prepares_after_the_device_move(
        self, built, minimal_pipeline_dict
    ):
        session_id = self.session_manager.create_session()

        response = self.service.load_pipeline(
            self._load_request(session_id, minimal_pipeline_dict), self.ctx
        )

        assert response.success is True
        assert self.order == ["to:cuda", "prepare"]
        assert self.session_manager.get_session(session_id).pipeline is self.pipeline

    def test_failed_prepare_fails_the_load_and_attaches_nothing(
        self, built, minimal_pipeline_dict
    ):
        session_id = self.session_manager.create_session()
        self.pipeline.prepare_inference.side_effect = RuntimeError(
            "Node 'segmenter' failed to prepare for inference: engine build failed"
        )

        response = self.service.load_pipeline(
            self._load_request(session_id, minimal_pipeline_dict), self.ctx
        )

        assert response.success is False
        assert "engine build failed" in self.ctx.set_details.call_args.args[0]
        assert self.session_manager.get_session(session_id).pipeline is None

    def test_load_pipeline_weights_prepares_after_the_weights(
        self, monkeypatch, tmp_path
    ):
        session_id = self.session_manager.create_session()
        self.session_manager.get_session(session_id).pipeline = self.pipeline
        weights = tmp_path / "fitted.pt"
        monkeypatch.setattr(
            pipeline_service_module.helpers,
            "find_weights_file",
            lambda path, dirs: weights,
        )

        response = self.service.load_pipeline_weights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id, weights_path=str(weights)
            ),
            self.ctx,
        )

        assert response.success is True
        assert self.order == ["weights", "prepare"]

    def test_load_pipeline_weights_from_bytes_prepares_after_the_weights(self):
        session_id = self.session_manager.create_session()
        self.session_manager.get_session(session_id).pipeline = self.pipeline

        response = self.service.load_pipeline_weights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id, weights_bytes=b"state"
            ),
            self.ctx,
        )

        assert response.success is True
        assert self.order == ["weights", "prepare"]
