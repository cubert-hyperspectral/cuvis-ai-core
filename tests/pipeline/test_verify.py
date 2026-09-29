"""Characterization tests for ``CuvisPipeline.verify``: the cycle check and the happy path."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec


class _Relay(Node):
    INPUT_SPECS = {"inp": PortSpec(dtype=torch.float32, shape=(-1, 2), optional=True)}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1, 2))}

    def forward(self, inp=None, **inputs):
        return {"out": torch.zeros(1, 2) if inp is None else inp}


def test_verify_rejects_a_cycle() -> None:
    first, second = _Relay(name="first"), _Relay(name="second")
    pipeline = CuvisPipeline("cycle")
    pipeline.connect(first.outputs.out, second.inp)
    pipeline.connect(second.outputs.out, first.inp)

    with pytest.raises(ValueError, match="Graph contains cycles!"):
        pipeline.verify()


def test_verify_accepts_a_chain() -> None:
    first, second = _Relay(name="first"), _Relay(name="second")
    pipeline = CuvisPipeline("chain")
    pipeline.connect(first.outputs.out, second.inp)

    pipeline.verify()
