"""StatisticalTrainer gathers a fit target's inputs exactly like a forward would."""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.trainers import StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec

_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


class Doubler(Node):
    INPUT_SPECS = {"cube": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, cube, **inputs):
        return {"out": cube * 2}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class TrainOnlyDoubler(Doubler):
    EXECUTION_STAGES = {ExecutionStage.TRAIN}


class MixedFit(Node):
    """Fits from a predecessor output and a batch key at the same time."""

    INPUT_SPECS = {
        "data": _VEC,
        "mask": PortSpec(dtype=torch.float32, shape=(-1,), optional=True),
    }
    OUTPUT_SPECS = {"out": _VEC}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen: list[dict] = []

    def forward(self, data, mask=None, **inputs):
        return {"out": data}

    def statistical_initialization(self, input_stream: InputStream) -> None:
        self.seen = [dict(batch) for batch in input_stream]

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class FanIn(Node):
    INPUT_SPECS = {"parts": PortSpec(dtype=torch.float32, shape=(-1,), variadic=True)}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, parts, **inputs):
        return {"out": torch.stack(parts).sum(0)}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class _ListDataModule(pl.LightningDataModule):
    def __init__(self, batches):
        super().__init__()
        self._batches = batches

    def train_dataloader(self):
        return self._batches


def _batch() -> dict:
    return {"cube": torch.tensor([1.0, 2.0]), "mask": torch.tensor([0.0, 1.0])}


def _stream(pipeline: CuvisPipeline, target: Node, batches: list[dict]) -> list[dict]:
    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=_ListDataModule(batches))
    return list(trainer._create_input_stream(target, batches))


def test_entry_node_reads_its_ports_from_the_batch():
    pipeline = CuvisPipeline("entry")
    entry, sink = Doubler(name="entry"), MixedFit(name="sink")
    pipeline.connect(entry.outputs.out, sink.inputs.data)

    (inputs,) = _stream(pipeline, entry, [_batch()])

    assert set(inputs) == {"cube"}
    assert torch.equal(inputs["cube"], torch.tensor([1.0, 2.0]))


def test_node_with_a_predecessor_still_receives_its_batch_key():
    pipeline = CuvisPipeline("mixed")
    entry, mixed = Doubler(name="entry"), MixedFit(name="mixed")
    pipeline.connect(entry.outputs.out, mixed.inputs.data)

    (inputs,) = _stream(pipeline, mixed, [_batch()])

    assert set(inputs) == {"data", "mask"}
    assert torch.equal(inputs["data"], torch.tensor([2.0, 4.0]))
    assert torch.equal(inputs["mask"], torch.tensor([0.0, 1.0]))


def test_fit_delivers_the_batch_key_to_a_node_with_predecessors():
    pipeline = CuvisPipeline("fit")
    entry, mixed = Doubler(name="entry"), MixedFit(name="mixed")
    pipeline.connect(entry.outputs.out, mixed.inputs.data)

    StatisticalTrainer(pipeline=pipeline, datamodule=_ListDataModule([_batch()])).fit()

    assert [set(seen) for seen in mixed.seen] == [{"data", "mask"}]


def test_variadic_port_collects_every_edge():
    pipeline = CuvisPipeline("fan-in")
    left, right, fan_in = (
        Doubler(name="left"),
        Doubler(name="right"),
        FanIn(name="fan_in"),
    )
    pipeline.connect(left.outputs.out, fan_in.inputs.parts)
    pipeline.connect(right.outputs.out, fan_in.inputs.parts)

    (inputs,) = _stream(pipeline, fan_in, [_batch()])

    assert isinstance(inputs["parts"], list) and len(inputs["parts"]) == 2


def test_variadic_port_never_appends_onto_a_batch_value_of_the_same_name():
    pipeline = CuvisPipeline("fan-in-batch")
    left, right, fan_in = (
        Doubler(name="left"),
        Doubler(name="right"),
        FanIn(name="fan_in"),
    )
    pipeline.connect(left.outputs.out, fan_in.inputs.parts)
    pipeline.connect(right.outputs.out, fan_in.inputs.parts)
    batch = {**_batch(), "parts": [torch.zeros(2)]}

    (inputs,) = _stream(pipeline, fan_in, [batch])

    assert isinstance(inputs["parts"], list) and len(inputs["parts"]) == 2
    assert len(batch["parts"]) == 1


def test_missing_required_predecessor_output_raises():
    pipeline = CuvisPipeline("missing")
    entry, mixed = TrainOnlyDoubler(name="entry"), MixedFit(name="mixed")
    pipeline.connect(entry.outputs.out, mixed.inputs.data)

    with pytest.raises(RuntimeError, match="entry.out"):
        _stream(pipeline, mixed, [_batch()])
