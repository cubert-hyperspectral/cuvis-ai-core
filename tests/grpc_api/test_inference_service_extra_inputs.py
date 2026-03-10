"""Tests for dynamic tensor parsing in InferenceService._parse_input_batch."""

from __future__ import annotations

import torch

from cuvis_ai_core.grpc import helpers
from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def test_parse_extra_inputs() -> None:
    service = InferenceService(SessionManager())

    start_frame = torch.tensor([10], dtype=torch.int32)
    max_frames = torch.tensor([50], dtype=torch.int32)
    points = torch.tensor([[0.45, 0.55], [0.5, 0.6]], dtype=torch.float32)

    inputs = cuvis_ai_pb2.InputBatch(
        extra_inputs={
            "start_frame": helpers.tensor_to_proto(start_frame),
            "max_frames": helpers.tensor_to_proto(max_frames),
            "points": helpers.tensor_to_proto(points),
        }
    )

    batch = service._parse_input_batch(inputs)

    assert "start_frame" in batch
    assert "max_frames" in batch
    assert "points" in batch

    assert torch.equal(batch["start_frame"], start_frame)
    assert torch.equal(batch["max_frames"], max_frames)
    assert torch.equal(batch["points"], points)
