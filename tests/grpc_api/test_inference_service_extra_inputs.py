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


def test_parse_rgb_image_present() -> None:
    service = InferenceService(SessionManager())
    rgb_image = torch.rand(1, 8, 9, 3, dtype=torch.float32)

    inputs = cuvis_ai_pb2.InputBatch(
        rgb_image=helpers.tensor_to_proto(rgb_image),
    )

    batch = service._parse_input_batch(inputs)

    assert "rgb_image" in batch
    assert torch.equal(batch["rgb_image"], rgb_image)


def test_parse_frame_id_present() -> None:
    service = InferenceService(SessionManager())
    frame_id = torch.tensor([42], dtype=torch.int64)

    inputs = cuvis_ai_pb2.InputBatch(
        frame_id=helpers.tensor_to_proto(frame_id),
    )

    batch = service._parse_input_batch(inputs)

    assert "frame_id" in batch
    assert torch.equal(batch["frame_id"], frame_id)


def test_parse_frame_id_absent_or_empty() -> None:
    service = InferenceService(SessionManager())

    batch_without = service._parse_input_batch(cuvis_ai_pb2.InputBatch())
    assert "frame_id" not in batch_without

    empty_frame_id = torch.tensor([], dtype=torch.int64)
    batch_empty = service._parse_input_batch(
        cuvis_ai_pb2.InputBatch(
            frame_id=helpers.tensor_to_proto(empty_frame_id),
        )
    )
    assert "frame_id" not in batch_empty


def test_parse_mesu_index_present() -> None:
    service = InferenceService(SessionManager())
    mesu_index = torch.tensor([42], dtype=torch.int64)

    inputs = cuvis_ai_pb2.InputBatch(
        mesu_index=helpers.tensor_to_proto(mesu_index),
    )

    batch = service._parse_input_batch(inputs)

    assert "mesu_index" in batch
    assert torch.equal(batch["mesu_index"], mesu_index)


def test_parse_mesu_index_absent_or_empty() -> None:
    service = InferenceService(SessionManager())

    batch_without = service._parse_input_batch(cuvis_ai_pb2.InputBatch())
    assert "mesu_index" not in batch_without

    empty_mesu_index = torch.tensor([], dtype=torch.int64)
    batch_empty = service._parse_input_batch(
        cuvis_ai_pb2.InputBatch(
            mesu_index=helpers.tensor_to_proto(empty_mesu_index),
        )
    )
    assert "mesu_index" not in batch_empty
