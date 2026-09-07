"""``CuvisPipeline.forward(free_consumed_ports=...)``: opt-in release of read ports.

Default ``False`` keeps the historical contract (every produced port comes back).
With ``True`` a port is dropped right after the last executing node that reads it
has run; pipeline outputs, the inputs of ``upto_node`` and ``keep_ports`` survive.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


class _Source(Node):
    INPUT_SPECS = {"x": _VEC}
    OUTPUT_SPECS = {"out": _VEC, "meta": _VEC}

    def forward(self, x, **kwargs):
        return {"out": x * 1.0, "meta": x * 0.0}


class _PlusOne(Node):
    INPUT_SPECS = {"inp": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, inp, **kwargs):
        return {"out": inp + 1}


class _Sum(Node):
    INPUT_SPECS = {"a": _VEC, "b": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, a, b, **kwargs):
        return {"out": a + b}


class _ValOnlyPlusOne(_PlusOne):
    EXECUTION_STAGES = {ExecutionStage.VAL}


def _chain():
    """a -> b -> c with ``a.meta`` left unconsumed."""
    pipeline = CuvisPipeline("chain")
    a, b, c = _Source(), _PlusOne(), _PlusOne()
    pipeline.connect(a.outputs.out, b.inp)
    pipeline.connect(b.outputs.out, c.inp)
    return pipeline, a, b, c


_X = torch.tensor([1.0, 2.0])


def test_default_returns_every_port() -> None:
    pipeline, a, b, c = _chain()
    outputs = pipeline.forward(batch={"x": _X})
    assert set(outputs) == {
        (a.name, "out"),
        (a.name, "meta"),
        (b.name, "out"),
        (c.name, "out"),
    }


def test_chain_frees_read_ports_and_keeps_outputs() -> None:
    pipeline, a, b, c = _chain()
    outputs = pipeline.forward(batch={"x": _X}, free_consumed_ports=True)
    # a.out was read by b, b.out by c: both gone. a.meta has no reader, c.out is
    # the pipeline output: both stay.
    assert set(outputs) == {(a.name, "meta"), (c.name, "out")}
    torch.testing.assert_close(outputs[(c.name, "out")], _X + 2)
    # A caller reading a freed port gets a KeyError naming it.
    with pytest.raises(KeyError, match=a.name):
        outputs[(a.name, "out")]


def test_fan_out_port_is_freed_only_after_its_last_reader() -> None:
    # a.out feeds both b and c; c also needs b.out. a.out must still be there
    # when c runs, and gone afterwards.
    pipeline = CuvisPipeline("fanout")
    a, b, c = _Source(), _PlusOne(), _Sum()
    pipeline.connect(a.outputs.out, b.inp)
    pipeline.connect(a.outputs.out, c.a)
    pipeline.connect(b.outputs.out, c.b)

    outputs = pipeline.forward(batch={"x": _X}, free_consumed_ports=True)

    assert set(outputs) == {(a.name, "meta"), (c.name, "out")}
    torch.testing.assert_close(outputs[(c.name, "out")], _X + (_X + 1))


def test_keep_ports_exact_key_and_bare_name() -> None:
    pipeline, a, b, c = _chain()

    exact = pipeline.forward(
        batch={"x": _X}, free_consumed_ports=True, keep_ports={(a.name, "out")}
    )
    assert set(exact) == {(a.name, "out"), (a.name, "meta"), (c.name, "out")}

    # A bare port name keeps that port on every node.
    by_name = pipeline.forward(
        batch={"x": _X}, free_consumed_ports=True, keep_ports={"out"}
    )
    assert set(by_name) == {
        (a.name, "out"),
        (a.name, "meta"),
        (b.name, "out"),
        (c.name, "out"),
    }


def test_keep_ports_rejects_malformed_entries() -> None:
    pipeline, *_ = _chain()
    with pytest.raises(TypeError, match="keep_ports"):
        pipeline.forward(
            batch={"x": _X}, free_consumed_ports=True, keep_ports=[("only-one",)]
        )


def test_upto_node_keeps_the_target_inputs() -> None:
    pipeline, a, b, c = _chain()
    outputs = pipeline.forward(batch={"x": _X}, upto_node=c, free_consumed_ports=True)
    # c does not run; its input b.out is what the caller reads. a.out was
    # consumed by b and is released.
    assert set(outputs) == {(a.name, "meta"), (b.name, "out")}
    torch.testing.assert_close(outputs[(b.name, "out")], _X + 1)


def test_stage_skipped_node_is_not_a_reader() -> None:
    pipeline = CuvisPipeline("stages")
    a, b, v = _Source(), _PlusOne(), _ValOnlyPlusOne()
    pipeline.connect(a.outputs.out, b.inp)
    pipeline.connect(a.outputs.out, v.inp)

    inference = pipeline.forward(
        batch={"x": _X}, stage=ExecutionStage.INFERENCE, free_consumed_ports=True
    )
    # v is skipped at inference, so b is the last reader of a.out.
    assert set(inference) == {(a.name, "meta"), (b.name, "out")}

    val = pipeline.forward(
        batch={"x": _X}, stage=ExecutionStage.VAL, free_consumed_ports=True
    )
    assert set(val) == {(a.name, "meta"), (b.name, "out"), (v.name, "out")}


def test_release_schedule_is_cached_per_stage_and_invalidated_on_connect() -> None:
    pipeline = CuvisPipeline("cache")
    a, b = _Source(), _PlusOne()
    pipeline.connect(a.outputs.out, b.inp)

    first = pipeline.forward(batch={"x": _X}, free_consumed_ports=True)
    assert set(first) == {(a.name, "meta"), (b.name, "out")}
    assert (ExecutionStage.INFERENCE, None) in pipeline._last_consumer_cache

    # Adding a reader for b.out must recompute the schedule.
    c = _PlusOne()
    pipeline.connect(b.outputs.out, c.inp)
    assert not pipeline._last_consumer_cache
    second = pipeline.forward(batch={"x": _X}, free_consumed_ports=True)
    assert set(second) == {(a.name, "meta"), (c.name, "out")}


def test_bare_string_stage_is_accepted() -> None:
    # ``forward(stage=...)`` still takes the plain strings ExecutionStage spells.
    pipeline, a, b, c = _chain()
    outputs = pipeline.forward(
        batch={"x": _X}, stage="inference", free_consumed_ports=True
    )
    assert set(outputs) == {(a.name, "meta"), (c.name, "out")}
    assert (ExecutionStage.INFERENCE, None) in pipeline._last_consumer_cache


def test_freed_forward_matches_default_forward_values() -> None:
    # Freeing changes what is returned, never what the nodes compute.
    pipeline, a, b, c = _chain()
    full = pipeline.forward(batch={"x": _X})
    freed = pipeline.forward(batch={"x": _X}, free_consumed_ports=True)
    for key, value in freed.items():
        torch.testing.assert_close(value, full[key])
