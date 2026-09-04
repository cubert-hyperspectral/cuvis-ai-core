"""Port-retention profile: per-node DEBUG lines with the bytes the forward holds.

Switched on by ``CUVIS_PROFILE_PORT_RETENTION=1`` (process-wide) or
``set_profiling(port_retention=True)`` (per pipeline); capped at two forwards
per stage.
"""

from __future__ import annotations

import re

import pytest
import torch
from loguru import logger

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline import pipeline as pipeline_mod
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


class _Source(Node):
    INPUT_SPECS = {"x": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, x, **kwargs):
        return {"out": x * 1.0}


class _PlusOne(Node):
    INPUT_SPECS = {"inp": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, inp, **kwargs):
        return {"out": inp + 1}


@pytest.fixture
def debug_lines():
    """Collect loguru DEBUG records emitted while the test runs."""
    records: list[str] = []
    sink_id = logger.add(
        lambda message: records.append(message.record["message"]), level="DEBUG"
    )
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _retention_lines(records: list[str]) -> list[str]:
    return [line for line in records if line.startswith("port-retention ")]


def _retained_bytes(line: str) -> int:
    match = re.search(r"retained_bytes=(\d+)", line)
    assert match is not None, line
    return int(match.group(1))


def _chain() -> CuvisPipeline:
    pipeline = CuvisPipeline("profile")
    a, b, c = _Source(), _PlusOne(), _PlusOne()
    pipeline.connect(a.outputs.out, b.inp)
    pipeline.connect(b.outputs.out, c.inp)
    return pipeline


_BATCH = {"x": torch.zeros(1000, dtype=torch.float32)}  # 4000 bytes per port


def test_off_by_default(debug_lines, monkeypatch) -> None:
    monkeypatch.delenv(pipeline_mod.PORT_RETENTION_ENV, raising=False)
    pipeline = _chain()
    pipeline.forward(batch=_BATCH)
    assert _retention_lines(debug_lines) == []


def test_env_switch_logs_one_line_per_node_for_two_forwards(
    debug_lines, monkeypatch
) -> None:
    monkeypatch.setenv(pipeline_mod.PORT_RETENTION_ENV, "1")
    pipeline = _chain()  # the env switch is read at construction

    for _ in range(3):
        pipeline.forward(batch=_BATCH)

    lines = _retention_lines(debug_lines)
    # Three nodes, but only the first two forwards of the stage are profiled.
    assert len(lines) == 6
    assert all("stage=inference" in line for line in lines)
    assert [line for line in lines if "forward=1" in line][:1]
    assert all("cuda 0.00 MiB" in line for line in lines)

    # Another stage gets its own two forwards.
    pipeline.forward(batch=_BATCH, stage=ExecutionStage.VAL)
    assert len(_retention_lines(debug_lines)) == 9


def test_set_profiling_switch_is_independent_of_timing_profiler(
    debug_lines, monkeypatch
) -> None:
    monkeypatch.delenv(pipeline_mod.PORT_RETENTION_ENV, raising=False)
    pipeline = _chain()
    pipeline.set_profiling(enabled=False, port_retention=True)
    assert pipeline.profiling_enabled is False

    pipeline.forward(batch=_BATCH)
    assert len(_retention_lines(debug_lines)) == 3

    # Full-replace semantics: a call without the flag switches it off again.
    pipeline.set_profiling(enabled=True)
    pipeline.forward(batch=_BATCH)
    assert len(_retention_lines(debug_lines)) == 3

    # Requesting it again re-arms the two-forward cap.
    pipeline.set_profiling(enabled=False, port_retention=True)
    pipeline.forward(batch=_BATCH)
    pipeline.forward(batch=_BATCH)
    pipeline.forward(batch=_BATCH)
    assert len(_retention_lines(debug_lines)) == 9


def test_profile_reports_the_smaller_footprint_with_port_freeing(
    debug_lines, monkeypatch
) -> None:
    monkeypatch.delenv(pipeline_mod.PORT_RETENTION_ENV, raising=False)
    pipeline = _chain()

    pipeline.set_profiling(enabled=False, port_retention=True)
    pipeline.forward(batch=_BATCH)
    kept_all = _retention_lines(debug_lines)[-1]

    pipeline.set_profiling(enabled=False, port_retention=True)
    pipeline.forward(batch=_BATCH, free_consumed_ports=True)
    freed = _retention_lines(debug_lines)[-1]

    # After the last node: three 4000-byte ports without freeing, one with.
    assert "freeing=False" in kept_all and _retained_bytes(kept_all) == 12000
    assert "freeing=True" in freed and _retained_bytes(freed) == 4000
    assert "ports=1" in freed


def test_tensor_bytes_walks_containers() -> None:
    tensor = torch.zeros(10, dtype=torch.float64)  # 80 bytes
    total, cuda = pipeline_mod._tensor_bytes(
        {"a": tensor, "b": [tensor, (tensor, "not a tensor")], "c": 3}
    )
    assert total == 240
    assert cuda == 0
