"""Characterization test: the warnings the weight restore logs for key mismatches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec


class _Source(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1, 2))}

    def forward(self, **inputs):
        return {"out": torch.zeros(1, 2)}


class _Mismatching(Node):
    """Reports twelve missing and two unexpected keys whatever it is given."""

    INPUT_SPECS = {"inp": PortSpec(dtype=torch.float32, shape=(-1, 2))}
    OUTPUT_SPECS = {}

    def forward(self, **inputs):
        return {}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return SimpleNamespace(
            missing_keys=[f"w{i}" for i in range(12)],
            unexpected_keys=["extra_a", "extra_b"],
        )


def test_restore_logs_long_and_short_key_mismatches(tmp_path: Path) -> None:
    source, sink = _Source(name="source"), _Mismatching(name="sink")
    pipeline = CuvisPipeline("restore-logging")
    pipeline.connect(source.outputs.out, sink.inp)
    weights = tmp_path / "weights.pt"
    torch.save({"state_dict": {"sink": {}}}, weights)
    messages: list[str] = []
    handle = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        absent = pipeline._restore_weights_from_checkpoint(
            weights, strict_weight_loading=False
        )
    finally:
        logger.remove(handle)

    assert absent == ["source"]
    assert messages == [
        "Node 'sink' missing 12 keys (showing first 5): ['w0', 'w1', 'w2', 'w3', 'w4']...",
        "Node 'sink' unexpected keys: ['extra_a', 'extra_b']",
    ]
