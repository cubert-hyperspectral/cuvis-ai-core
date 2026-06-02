"""A variadic (fan-in) input port collects a value from every inbound
connection into a list at runtime; non-variadic ports take a single value."""

from __future__ import annotations

import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

_CAPTURED: dict = {}


class _Source(Node):
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

    def __init__(self, value: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._value = value

    def forward(self, **_):
        return {"out": torch.tensor([self._value])}

    def load(self, params, serial_dir) -> None:
        return None


class _FanInSink(Node):
    INPUT_SPECS = {
        "items": PortSpec(dtype=torch.float32, shape=(-1,), variadic=True),
    }
    OUTPUT_SPECS = {"merged": PortSpec(dtype=torch.float32, shape=(-1,))}

    def forward(self, items, **_):
        _CAPTURED["items"] = items
        return {"merged": torch.cat(items)}

    def load(self, params, serial_dir) -> None:
        return None


def test_variadic_port_collects_list_from_multiple_sources() -> None:
    _CAPTURED.clear()
    pipeline = CuvisPipeline("fan_in")
    a = _Source(1.0, name="a")
    b = _Source(2.0, name="b")
    sink = _FanInSink(name="sink")

    pipeline.connect(a.outputs.out, sink.inputs.items)
    pipeline.connect(b.outputs.out, sink.inputs.items)

    outputs = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    # The variadic port delivered one value per inbound connection, as a list.
    assert isinstance(_CAPTURED["items"], list)
    assert len(_CAPTURED["items"]) == 2
    merged = torch.sort(outputs[(sink.name, "merged")]).values
    torch.testing.assert_close(merged, torch.tensor([1.0, 2.0]))


def test_variadic_flag_rejected_on_output_spec() -> None:
    """variadic is input-only; setting it on an OUTPUT_SPECS entry is an error."""

    class _BadOutput(Node):
        OUTPUT_SPECS = {
            "out": PortSpec(dtype=torch.float32, shape=(-1,), variadic=True),
        }

        def forward(self, **_):
            return {"out": torch.zeros(1)}

    try:
        _BadOutput(name="bad")
    except TypeError as exc:
        assert "variadic" in str(exc)
    else:
        raise AssertionError("expected TypeError for variadic output spec")
