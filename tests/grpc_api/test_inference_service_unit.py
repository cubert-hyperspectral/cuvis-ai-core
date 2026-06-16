"""Unit tests for InferenceService helper methods and happy-path inference.

Covers the branches not hit by test_inference_service_extra_inputs.py and
test_service_error_paths.py: _format_output_key, _should_return, _to_tensor,
_get_pipeline_device, _move_batch_to_pipeline_device, _parse_points, the
cube/wavelengths/mask/text_prompt input fields, and the inference() happy path.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cuvis_ai_core.grpc import helpers
from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service() -> InferenceService:
    return InferenceService(SessionManager())


# ---------------------------------------------------------------------------
# _format_output_key
# ---------------------------------------------------------------------------

class TestFormatOutputKey:
    def setup_method(self):
        self.service = _make_service()

    def test_tuple_key(self):
        assert self.service._format_output_key(("node", "port")) == "node.port"

    def test_string_key(self):
        assert self.service._format_output_key("output") == "output"

    def test_non_string_non_tuple(self):
        assert self.service._format_output_key(42) == "42"


# ---------------------------------------------------------------------------
# _should_return
# ---------------------------------------------------------------------------

class TestShouldReturn:
    def setup_method(self):
        self.service = _make_service()

    def test_empty_specs_always_true(self):
        assert self.service._should_return("node.output", set()) is True

    def test_full_name_match(self):
        assert self.service._should_return("node.output", {"node.output"}) is True

    def test_port_only_match(self):
        assert self.service._should_return("node.output", {"output"}) is True

    def test_no_match_returns_false(self):
        assert self.service._should_return("node.output", {"other"}) is False


# ---------------------------------------------------------------------------
# _to_tensor
# ---------------------------------------------------------------------------

class TestToTensor:
    def setup_method(self):
        self.service = _make_service()

    def test_torch_tensor_passthrough(self):
        t = torch.randn(3, 4)
        assert self.service._to_tensor(t) is t

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.service._to_tensor(arr)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor(arr))

    def test_list_iterable(self):
        result = self.service._to_tensor([1, 2, 3])
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [1, 2, 3]

    def test_scalar_int(self):
        result = self.service._to_tensor(42)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 42


# ---------------------------------------------------------------------------
# _get_pipeline_device
# ---------------------------------------------------------------------------

class TestGetPipelineDevice:
    def setup_method(self):
        self.service = _make_service()

    def test_no_layers_returns_cpu(self):
        pipeline = Mock()
        pipeline.torch_layers = []
        assert self.service._get_pipeline_device(pipeline) == torch.device("cpu")

    def test_layer_with_parameters(self):
        cpu_tensor = torch.zeros(1)
        layer = Mock()
        layer.parameters.return_value = iter([cpu_tensor])
        layer.buffers.return_value = iter([])
        pipeline = Mock()
        pipeline.torch_layers = [layer]
        assert self.service._get_pipeline_device(pipeline) == cpu_tensor.device

    def test_layer_with_buffers_no_params(self):
        cpu_tensor = torch.zeros(1)
        layer = Mock()
        layer.parameters.return_value = iter([])
        layer.buffers.return_value = iter([cpu_tensor])
        pipeline = Mock()
        pipeline.torch_layers = [layer]
        assert self.service._get_pipeline_device(pipeline) == cpu_tensor.device


# ---------------------------------------------------------------------------
# _move_batch_to_pipeline_device
# ---------------------------------------------------------------------------

class TestMoveBatchToPipelineDevice:
    def setup_method(self):
        self.service = _make_service()
        self.cpu_pipeline = Mock()
        self.cpu_pipeline.torch_layers = []  # → device cpu

    def test_none_pipeline_returns_batch_unchanged(self):
        batch = {"cube": torch.zeros(2, 3), "meta": [1, 2]}
        result = self.service._move_batch_to_pipeline_device(batch, None)
        assert result is batch

    def test_tensors_moved_to_device(self):
        tensor = torch.randn(2, 3)
        batch = {"cube": tensor}
        result = self.service._move_batch_to_pipeline_device(batch, self.cpu_pipeline)
        assert isinstance(result["cube"], torch.Tensor)
        torch.testing.assert_close(result["cube"], tensor)

    def test_non_tensor_values_pass_through(self):
        payload = [1, 2, 3]
        batch = {"bboxes": payload}
        result = self.service._move_batch_to_pipeline_device(batch, self.cpu_pipeline)
        assert result["bboxes"] is payload

    def test_empty_batch(self):
        result = self.service._move_batch_to_pipeline_device({}, self.cpu_pipeline)
        assert result == {}


# ---------------------------------------------------------------------------
# _parse_input_batch — additional fields not covered by existing tests
# ---------------------------------------------------------------------------

class TestParseInputBatchAdditional:
    def setup_method(self):
        self.service = _make_service()

    def test_parse_cube(self):
        cube = torch.rand(1, 4, 5, 3, dtype=torch.float32)
        inputs = cuvis_ai_pb2.InputBatch(cube=helpers.tensor_to_proto(cube))
        batch = self.service._parse_input_batch(inputs)
        assert "cube" in batch
        torch.testing.assert_close(batch["cube"], cube)

    def test_parse_wavelengths(self):
        wav = torch.rand(1, 10, dtype=torch.float32)
        inputs = cuvis_ai_pb2.InputBatch(wavelengths=helpers.tensor_to_proto(wav))
        batch = self.service._parse_input_batch(inputs)
        assert "wavelengths" in batch
        torch.testing.assert_close(batch["wavelengths"], wav)

    def test_parse_mask(self):
        mask = torch.zeros(1, 4, 5, dtype=torch.float32)
        inputs = cuvis_ai_pb2.InputBatch(mask=helpers.tensor_to_proto(mask))
        batch = self.service._parse_input_batch(inputs)
        assert "mask" in batch
        torch.testing.assert_close(batch["mask"], mask)

    def test_parse_text_prompt(self):
        inputs = cuvis_ai_pb2.InputBatch(text_prompt="detect anomaly")
        batch = self.service._parse_input_batch(inputs)
        assert batch["text_prompt"] == "detect anomaly"

    def test_parse_with_stack(self):
        cube = torch.rand(1, 4, 5, 3, dtype=torch.float32)
        inputs = cuvis_ai_pb2.InputBatch(cube=helpers.tensor_to_proto(cube))
        with ExitStack() as stack:
            batch = self.service._parse_input_batch(
                inputs, copy_tensors=False, stack=stack
            )
            assert "cube" in batch
            torch.testing.assert_close(batch["cube"], cube)


# ---------------------------------------------------------------------------
# _parse_points
# ---------------------------------------------------------------------------

class TestParsePoints:
    def setup_method(self):
        self.service = _make_service()

    def test_point_type_unspecified_becomes_neutral(self):
        points_proto = cuvis_ai_pb2.Points(
            points=[
                cuvis_ai_pb2.Point(
                    element_id=0,
                    x=0.1,
                    y=0.2,
                    type=cuvis_ai_pb2.POINT_TYPE_UNSPECIFIED,
                )
            ]
        )
        result = self.service._parse_points(points_proto)
        assert result[0]["type"] == "neutral"

    def test_point_type_positive(self):
        points_proto = cuvis_ai_pb2.Points(
            points=[
                cuvis_ai_pb2.Point(
                    element_id=1,
                    x=0.5,
                    y=0.6,
                    type=cuvis_ai_pb2.POINT_TYPE_POSITIVE,
                )
            ]
        )
        result = self.service._parse_points(points_proto)
        assert result[0]["type"] == "positive"

    def test_multiple_points_all_fields(self):
        points_proto = cuvis_ai_pb2.Points(
            points=[
                cuvis_ai_pb2.Point(
                    element_id=0, x=0.1, y=0.2, type=cuvis_ai_pb2.POINT_TYPE_POSITIVE
                ),
                cuvis_ai_pb2.Point(
                    element_id=1, x=0.3, y=0.4, type=cuvis_ai_pb2.POINT_TYPE_NEGATIVE
                ),
            ]
        )
        result = self.service._parse_points(points_proto)
        assert len(result) == 2
        assert result[0] == {"element_id": 0, "x": pytest.approx(0.1), "y": pytest.approx(0.2), "type": "positive"}
        assert result[1] == {"element_id": 1, "x": pytest.approx(0.3), "y": pytest.approx(0.4), "type": "negative"}


# ---------------------------------------------------------------------------
# inference() happy path
# ---------------------------------------------------------------------------

class TestInferenceHappyPath:
    """Test inference() with a mock pipeline injected into a real session."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = InferenceService(self.session_manager)
        self.ctx = Mock()
        self.session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(self.session_id)
        self.mock_pipeline = Mock()
        self.mock_pipeline.torch_layers = []
        session.pipeline = self.mock_pipeline

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def _infer(self, output_specs=None):
        kwargs = {"session_id": self.session_id}
        if output_specs:
            kwargs["output_specs"] = output_specs
        return self.service.inference(cuvis_ai_pb2.InferenceRequest(**kwargs), self.ctx)

    def test_inference_returns_tensor_output(self):
        tensor = torch.randn(2, 3)
        self.mock_pipeline.forward.return_value = {"out": tensor}
        response = self._infer()
        assert "out" in response.outputs
        with helpers.proto_to_tensor(response.outputs["out"]) as t:
            torch.testing.assert_close(t, tensor)

    def test_inference_returns_float_metric(self):
        self.mock_pipeline.forward.return_value = {"loss": 0.42}
        response = self._infer()
        assert "loss" in response.metrics
        assert response.metrics["loss"] == pytest.approx(0.42)

    def test_inference_returns_int_metric(self):
        self.mock_pipeline.forward.return_value = {"count": 7}
        response = self._infer()
        assert "count" in response.metrics
        assert response.metrics["count"] == 7.0

    def test_inference_bool_not_in_metrics(self):
        self.mock_pipeline.forward.return_value = {"flag": True}
        response = self._infer()
        assert "flag" not in response.metrics

    def test_inference_output_filtering_keeps_wanted(self):
        self.mock_pipeline.forward.return_value = {
            "wanted": torch.zeros(2),
            "unwanted": torch.ones(3),
        }
        response = self._infer(output_specs=["wanted"])
        assert "wanted" in response.outputs
        assert "unwanted" not in response.outputs

    def test_inference_port_only_filter(self):
        """output_specs using just port name (no node prefix) still matches."""
        self.mock_pipeline.forward.return_value = {"node.out": torch.zeros(2)}
        response = self._infer(output_specs=["out"])
        assert "node.out" in response.outputs

    def test_inference_non_tensorizable_skipped(self):
        class _Untensorizable:
            pass

        self.mock_pipeline.forward.return_value = {"bad": _Untensorizable()}
        response = self._infer()
        assert "bad" not in response.outputs
        assert "bad" not in response.metrics

    def test_inference_fallback_when_requested_spec_unserializable(self):
        """Requesting only a non-serializable output falls back to all serializable ones."""
        self.mock_pipeline.forward.return_value = {
            ("rgb_selector", "band_info"): {"strategy": "x", "bands": [1, 2, 3]},
            ("rgb_selector", "rgb_image"): torch.zeros(2, 3),
            ("detector", "scores"): torch.ones(4),
        }
        # band_info is a dict and cannot serialize; it is the only requested spec.
        response = self._infer(output_specs=["rgb_selector.band_info"])
        assert "rgb_selector.band_info" not in response.outputs
        # Fallback returns the rest of the serializable outputs.
        assert "rgb_selector.rgb_image" in response.outputs
        assert "detector.scores" in response.outputs

    def test_inference_tuple_key_formatted(self):
        tensor = torch.randn(2)
        self.mock_pipeline.forward.return_value = {("mynode", "out"): tensor}
        response = self._infer()
        assert "mynode.out" in response.outputs

    def test_inference_numpy_output_coerced_to_tensor(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.mock_pipeline.forward.return_value = {"numpy_out": arr}
        response = self._infer()
        assert "numpy_out" in response.outputs
        with helpers.proto_to_tensor(response.outputs["numpy_out"]) as t:
            torch.testing.assert_close(t, torch.tensor(arr))

    def test_inference_empty_outputs(self):
        self.mock_pipeline.forward.return_value = {}
        response = self._infer()
        assert len(response.outputs) == 0
        assert len(response.metrics) == 0
