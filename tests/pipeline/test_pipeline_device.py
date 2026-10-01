"""Characterization tests for ``CuvisPipeline.device`` and ``move_batch_to_device``."""

from __future__ import annotations

import torch
from torch import nn

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec


class _Source(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1, 2))}

    def forward(self, **inputs):
        return {"out": torch.zeros(1, 2)}


class _ParamSource(_Source):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))


class _BufferSource(_Source):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(2))


class _Sink(Node):
    INPUT_SPECS = {"inp": PortSpec(dtype=torch.float32, shape=(-1, 2))}
    OUTPUT_SPECS = {}

    def forward(self, **inputs):
        return {}


def _pipeline(source: Node) -> CuvisPipeline:
    pipeline = CuvisPipeline("device-test")
    pipeline.connect(source.outputs.out, _Sink().inp)
    return pipeline


def test_device_is_cpu_for_a_pipeline_without_tensors() -> None:
    assert CuvisPipeline("empty").device == torch.device("cpu")


def test_device_follows_the_first_parameter() -> None:
    source = _ParamSource()
    assert _pipeline(source).device == source.scale.device


def test_device_follows_the_first_buffer_when_there_is_no_parameter() -> None:
    source = _BufferSource()
    assert _pipeline(source).device == source.mean.device


def test_move_batch_to_device_moves_tensors_and_keeps_the_rest() -> None:
    pipeline = _pipeline(_ParamSource())
    tensor = torch.randn(2, 2)
    payload = [1, 2, 3]

    moved = pipeline.move_batch_to_device({"cube": tensor, "meta": payload})

    assert moved["cube"].device == pipeline.device
    torch.testing.assert_close(moved["cube"], tensor)
    assert moved["meta"] is payload


def test_move_batch_to_device_on_an_empty_batch() -> None:
    assert CuvisPipeline("empty").move_batch_to_device({}) == {}
