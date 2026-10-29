"""Run a pipeline over one split without Lightning.

``forward_split`` is the one loop the post-fit steps share: eval mode, no gradients, the
batch moved to the pipeline's device, a cooperative stop check before each batch and a
frame count per batch. The calibration phase feeds it the inference stage and collects the
decider inputs; ``evaluate_split`` feeds it the validation or test stage and reduces the
metric nodes the way :class:`~cuvis_ai_core.training.trainers.GradientTrainer` logs them:
names in a node's ``POOLED_METRIC_NAMES`` by one ``compute()`` of the node's live
torchmetrics object at the end of the pass, every other name by the mean over batches.

The pass leaves the modules in eval mode: its callers run it after a fit, on a pipeline
that is about to be saved.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

if TYPE_CHECKING:
    from cuvis_ai_core.node.node import Node
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

OnBatch = Callable[[int, int], None]
"""Progress sink: called with ``(frames_done, frames_total)`` after each batch (total 0 when
the loader has no sized dataset)."""


class TrainingCancelled(Exception):
    """A training run was stopped cooperatively via the session's stop event.

    Raised by :class:`~cuvis_ai_core.training.trainers.StatisticalTrainer` at batch and
    node boundaries and by :func:`forward_split` before a batch. A node interrupted
    mid-initialization is left partially fitted; the cancelled run must not be treated as
    trained.

    ``swallowed_error`` carries the exception the run raised when the stop had already
    been requested, so the cancelled answer can still name the failure instead of hiding
    it.
    """

    def __init__(
        self, *args: object, swallowed_error: BaseException | None = None
    ) -> None:
        """Store the optional error this cancellation took precedence over."""
        super().__init__(*args)
        self.swallowed_error = swallowed_error


# The CUDA runtime and cuBLAS report a failed device allocation as a plain RuntimeError.
_CUDA_OUT_OF_MEMORY_TEXT = ("CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED")


def is_cuda_out_of_memory(exc: BaseException | None) -> bool:
    """Whether ``exc``, or an error it was raised from, is the GPU running out of memory.

    torch raises ``torch.cuda.OutOfMemoryError`` when its caching allocator cannot serve a
    request; the CUDA runtime and cuBLAS name the same condition in a ``RuntimeError``.
    Loaders re-raise with ``raise ... from exc``, so the causes count too.
    """
    while exc is not None:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(exc, RuntimeError) and any(
            text in str(exc) for text in _CUDA_OUT_OF_MEMORY_TEXT
        ):
            return True
        exc = exc.__cause__
    return False


@dataclass(frozen=True)
class SplitBatch:
    """One batch of a split pass: its index, the frames seen so far and the outputs."""

    batch_idx: int
    frames_done: int
    frames_total: int
    outputs: dict[tuple[str, str], Any]


@dataclass(frozen=True)
class EvaluationSummary:
    """The metrics of one split pass keyed ``<node>/<metric>`` and the frames it saw."""

    metrics: dict[str, float] = field(default_factory=dict)
    frames: int = 0


def frame_total(loader: Iterable[Any]) -> int:
    """``len(loader.dataset)`` when the loader carries a sized dataset, else 0 (unknown)."""
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return 0
    try:
        return len(dataset)
    except TypeError:
        return 0


def batch_frames(batch: Any) -> int:
    """The frames in ``batch``: the leading dimension of its first tensor, else 1."""
    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
        return 1
    if isinstance(batch, torch.Tensor) and batch.dim() > 0:
        return int(batch.shape[0])
    return 1


def forward_split(
    pipeline: CuvisPipeline,
    loader: Iterable[Any],
    *,
    stage: ExecutionStage,
    epoch: int = 0,
    global_step: int = 0,
    stop_event: threading.Event | None = None,
    on_batch: OnBatch | None = None,
    free_consumed_ports: bool = False,
    keep_ports: Collection[tuple[str, str] | str] | None = None,
) -> Iterator[SplitBatch]:
    """Forward every batch of ``loader`` through ``pipeline`` at ``stage``, without gradients.

    The modules are put in eval mode first. Before each batch the stop event is read; a set
    event raises :class:`TrainingCancelled`, so a stop lands at a batch boundary. After each
    batch ``on_batch`` receives the frames seen so far and the split's frame total (0 when
    the loader has no sized dataset), and the batch's outputs are yielded with the same
    counts.

    Parameters
    ----------
    pipeline : CuvisPipeline
        The pipeline to run; its ``move_batch_to_device`` places each batch.
    loader : iterable
        The split's batches, a torch DataLoader or any iterable of batches.
    stage : ExecutionStage
        The context stage every forward runs at.
    epoch, global_step : int
        The context's epoch and step for every batch; a caller after a fit passes an
        epoch no fit pass used and a step past the last logged one.
    stop_event : threading.Event, optional
        Cooperative cancel flag.
    on_batch : callable, optional
        Progress sink ``(frames_done, frames_total)``.
    free_consumed_ports, keep_ports
        Passed through to ``pipeline.forward``.
    """
    total = frame_total(loader)
    for module in pipeline.torch_layers:
        module.eval()
    done = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if stop_event is not None and stop_event.is_set():
                raise TrainingCancelled(f"{stage.value} pass stopped via StopTrain")
            moved = pipeline.move_batch_to_device(batch)
            outputs = pipeline.forward(
                batch=moved,
                context=Context(
                    stage=stage,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                ),
                free_consumed_ports=free_consumed_ports,
                keep_ports=keep_ports,
            )
            done += batch_frames(batch)
            if on_batch is not None:
                on_batch(done, total)
            yield SplitBatch(batch_idx, done, total, outputs)


def default_metric_nodes(pipeline: CuvisPipeline) -> list[Node]:
    """Every node with a ``metrics`` output port: what an evaluation reduces by default."""
    return [node for node in pipeline.nodes() if "metrics" in node.OUTPUT_SPECS]


def _pooled_objects(node: Node) -> dict[str, Any]:
    """The live torchmetrics objects of ``node`` whose names it declares as pooled."""
    getter = getattr(node, "pooled_metrics", None)
    if getter is None:
        return {}
    names = getattr(node, "POOLED_METRIC_NAMES", frozenset())
    return {name: obj for name, obj in getter().items() if name in names}


def evaluate_split(
    pipeline: CuvisPipeline,
    loader: Iterable[Any],
    *,
    stage: ExecutionStage,
    metric_nodes: Collection[Node],
    epoch: int = 0,
    global_step: int = 0,
    stop_event: threading.Event | None = None,
    on_batch: OnBatch | None = None,
) -> EvaluationSummary:
    """Run one evaluation pass and reduce the metric nodes over it.

    Every pooled metric object is reset before the pass, so an earlier pass never leaks
    into this one. Per batch, each metric node's ``metrics`` port (a list of
    :class:`~cuvis_ai_schemas.execution.Metric`) contributes its non-pooled values to a
    running mean; at the end each pooled object is reduced once by ``compute()``. Keys are
    ``<node>/<metric>``, the names the gradient trainer logs; two nodes that report the
    same name are told apart by the node name, two ports of one node are not. An empty
    loader returns an empty summary with zero frames.
    """
    for node in metric_nodes:
        for obj in _pooled_objects(node).values():
            obj.reset()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    frames = 0
    for item in forward_split(
        pipeline,
        loader,
        stage=stage,
        epoch=epoch,
        global_step=global_step,
        stop_event=stop_event,
        on_batch=on_batch,
        free_consumed_ports=True,
        keep_ports={"metrics"},
    ):
        frames = item.frames_done
        for node in metric_nodes:
            pooled_names = getattr(node, "POOLED_METRIC_NAMES", frozenset())
            for metric in item.outputs.get((node.name, "metrics"), None) or []:
                if metric.name in pooled_names:
                    continue
                key = f"{node.name}/{metric.name}"
                sums[key] = sums.get(key, 0.0) + float(metric.value)
                counts[key] = counts.get(key, 0) + 1

    if frames == 0:
        return EvaluationSummary()

    metrics = {key: sums[key] / counts[key] for key in sums}
    for node in metric_nodes:
        for name, obj in _pooled_objects(node).items():
            if getattr(obj, "_update_count", 1) == 0:
                continue
            metrics[f"{node.name}/{name}"] = float(obj.compute())
    return EvaluationSummary(metrics=metrics, frames=frames)


__all__ = [
    "EvaluationSummary",
    "OnBatch",
    "SplitBatch",
    "TrainingCancelled",
    "batch_frames",
    "evaluate_split",
    "forward_split",
    "frame_total",
]
