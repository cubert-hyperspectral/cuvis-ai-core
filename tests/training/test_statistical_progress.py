"""StatisticalTrainer reports each node's fit as typed events with frame progress."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.post_fit import (
    EpochProgress,
    FrameProgress,
    TrainEvent,
    TrainHook,
    describe_event,
)
from cuvis_ai_core.training.trainers import StatisticalTrainer
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec

_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


class Pass(Node):
    INPUT_SPECS = {"cube": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, cube, **inputs):
        return {"out": cube}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class Fit(Pass):
    """Drains its stream, then computes; the shared log records when the compute ran."""

    log: list[str] | None = None

    def statistical_initialization(self, input_stream: InputStream) -> None:
        batches = list(input_stream)
        if self.log is not None:
            self.log.append(f"compute {self.name} over {len(batches)} batches")


class _SizedLoader:
    """A loader with a sized dataset, the way a DataLoader exposes it."""

    def __init__(self, batches: list[dict]) -> None:
        self.batches = batches
        self.dataset = range(sum(int(batch["cube"].shape[0]) for batch in batches))

    def __iter__(self):
        return iter(self.batches)


class _DataModule(pl.LightningDataModule):
    def __init__(self, loader) -> None:
        super().__init__()
        self._loader = loader

    def train_dataloader(self):
        return self._loader


def _batch(frames: int) -> dict:
    return {"cube": torch.arange(frames, dtype=torch.float32)}


def _pipeline(log: list[str]) -> CuvisPipeline:
    pipeline = CuvisPipeline("stat")
    entry, first, second = Pass(name="entry"), Fit(name="first"), Fit(name="second")
    first.log = log
    second.log = log
    pipeline.connect(entry.outputs.out, first.inputs.cube)
    pipeline.connect(first.outputs.out, second.inputs.cube)
    return pipeline


def _run(batches: list[dict], loader=None) -> tuple[list[TrainEvent], list[str]]:
    log: list[str] = []
    events: list[TrainEvent] = []

    def emit(event: TrainEvent) -> None:
        events.append(event)
        log.append(f"{event.hook.name} {event.node}".rstrip())

    loader = loader if loader is not None else _SizedLoader(batches)
    StatisticalTrainer(
        pipeline=_pipeline(log), datamodule=_DataModule(loader), emit=emit
    ).fit()
    return events, log


def test_each_node_reports_its_start_batches_and_end_in_topological_order():
    events, _ = _run([_batch(2), _batch(2), _batch(2)])

    start, progress, end = (
        TrainHook.STATISTICAL_NODE_START,
        TrainHook.STATISTICAL_PROGRESS,
        TrainHook.STATISTICAL_NODE_END,
    )
    assert [(e.hook, e.node) for e in events] == [
        (start, "first"),
        *([(progress, "first")] * 3),
        (end, "first"),
        (start, "second"),
        *([(progress, "second")] * 3),
        (end, "second"),
    ]
    assert events[0].progress == FrameProgress(0, 6)
    assert [e.progress for e in events[1:4]] == [
        FrameProgress(2, 6),
        FrameProgress(4, 6),
        FrameProgress(6, 6),
    ]
    assert events[4].progress is None
    # The count restarts for the next node.
    assert events[5].progress == FrameProgress(0, 6)
    assert events[6].progress == FrameProgress(2, 6)
    assert all(e.epochs is None and e.split is None for e in events)


def test_a_partial_last_batch_counts_its_frames():
    events, _ = _run([_batch(2), _batch(2), _batch(1)])

    first = [e.progress for e in events if e.node == "first" and e.progress is not None]
    assert first == [
        FrameProgress(0, 5),
        FrameProgress(2, 5),
        FrameProgress(4, 5),
        FrameProgress(5, 5),
    ]


def test_a_loader_without_a_sized_dataset_reports_an_unknown_total():
    batches = [_batch(2), _batch(2)]
    events, _ = _run(batches, loader=batches)  # a plain list: no .dataset

    assert events[0].progress == FrameProgress(0, 0)
    assert [e.progress for e in events[1:3]] == [
        FrameProgress(2, 0),
        FrameProgress(4, 0),
    ]


def test_a_node_computing_after_its_last_batch_emits_nothing_before_its_end():
    _, log = _run([_batch(2), _batch(2)])

    assert log == [
        "STATISTICAL_NODE_START first",
        "STATISTICAL_PROGRESS first",
        "STATISTICAL_PROGRESS first",
        "compute first over 2 batches",
        "STATISTICAL_NODE_END first",
        "STATISTICAL_NODE_START second",
        "STATISTICAL_PROGRESS second",
        "STATISTICAL_PROGRESS second",
        "compute second over 2 batches",
        "STATISTICAL_NODE_END second",
    ]


def test_a_pipeline_without_statistical_nodes_emits_nothing():
    events: list[TrainEvent] = []
    pipeline = CuvisPipeline("plain")
    entry, sink = Pass(name="entry"), Pass(name="sink")
    pipeline.connect(entry.outputs.out, sink.inputs.cube)

    StatisticalTrainer(
        pipeline=pipeline,
        datamodule=_DataModule(_SizedLoader([_batch(2)])),
        emit=events.append,
    ).fit()

    assert events == []


def test_the_log_line_names_a_node_start_and_end_only():
    """The CLI's one line per node comes from describe_event; the batches stay silent."""
    assert (
        describe_event(
            TrainEvent(
                TrainHook.STATISTICAL_NODE_START,
                node="norm",
                progress=FrameProgress(0, 4),
            )
        )
        == "Fitting norm"
    )
    assert (
        describe_event(TrainEvent(TrainHook.STATISTICAL_NODE_END, node="norm"))
        == "Fitted norm"
    )
    for hook in (
        TrainHook.STATISTICAL_PROGRESS,
        TrainHook.EPOCH_START,
        TrainHook.TRAIN_PROGRESS,
        TrainHook.VALIDATION_START,
        TrainHook.VALIDATION_PROGRESS,
    ):
        event = TrainEvent(
            hook, node="norm", progress=FrameProgress(1, 4), epochs=EpochProgress(1, 2)
        )
        assert describe_event(event) is None
