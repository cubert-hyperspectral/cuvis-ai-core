"""The shared split forward (``forward_split``) and the post-fit evaluation summary.

Both run a pipeline over one split without Lightning: the calibration phase and the
post-fit evaluation share the loop (eval mode, no gradients, the device move, the stop
check), and ``evaluate_split`` reduces the metric nodes the way the gradient trainer logs
them (pooled names by one ``compute()`` at the end, the rest by the mean over batches).
"""

from __future__ import annotations

import threading

import pytest
import torch
from torchmetrics.classification import BinaryAUROC

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.evaluation import (
    TrainingCancelled,
    evaluate_split,
    forward_split,
    is_cuda_out_of_memory,
)
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Metric
from cuvis_ai_schemas.pipeline import PortSpec

pytestmark = pytest.mark.unit

_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


class _Source(Node):
    """Passes the batch's scores and targets through."""

    INPUT_SPECS = {"scores": _VEC, "targets": _VEC}
    OUTPUT_SPECS = {"scores": _VEC, "targets": _VEC}

    def forward(self, scores, targets, **_):
        """Echo both tensors."""
        return {"scores": scores, "targets": targets}


class _Metrics(Node):
    """A metric node with one pooled metric (AUROC) and one per-batch value."""

    INPUT_SPECS = {"scores": _VEC, "targets": _VEC}
    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=())}
    POOLED_METRIC_NAMES = frozenset({"auroc"})
    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auroc = BinaryAUROC()
        self.seen_stages: list[ExecutionStage] = []

    def forward(self, scores, targets, context, **_):
        """Update the pooled AUROC and report the batch mean score."""
        self.seen_stages.append(context.stage)
        self.auroc.update(scores.flatten(), targets.flatten().int())
        return {
            "metrics": [
                Metric(
                    name="auroc",
                    value=float(self.auroc.compute()),
                    stage=context.stage,
                    epoch=context.epoch,
                ),
                Metric(
                    name="mean_score",
                    value=float(scores.mean()),
                    stage=context.stage,
                    epoch=context.epoch,
                ),
            ]
        }

    def pooled_metrics(self):
        """Expose the pooled object the trainer reduces at the end."""
        return {"auroc": self.auroc}


def _pipeline() -> tuple[CuvisPipeline, _Metrics]:
    pipe = CuvisPipeline("eval_test")
    src = _Source(name="src")
    met = _Metrics(name="met")
    pipe.connect(src.outputs.scores, met.inputs.scores)
    pipe.connect(src.outputs.targets, met.inputs.targets)
    return pipe, met


def _batches() -> list[dict[str, torch.Tensor]]:
    return [
        {"scores": torch.tensor([0.1, 0.2]), "targets": torch.tensor([0.0, 0.0])},
        {"scores": torch.tensor([0.8, 0.9]), "targets": torch.tensor([1.0, 1.0])},
        {"scores": torch.tensor([0.3]), "targets": torch.tensor([0.0])},
    ]


class _SizedLoader:
    """A loader with a sized dataset, like a torch DataLoader."""

    def __init__(self, batches, frames: int):
        self._batches = batches
        self.dataset = list(range(frames))

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def test_forward_split_counts_frames_and_yields_outputs():
    pipe, met = _pipeline()

    seen = list(forward_split(pipe, _batches(), stage=ExecutionStage.VAL, epoch=7))

    assert [(b.batch_idx, b.frames_done, b.frames_total) for b in seen] == [
        (0, 2, 0),
        (1, 4, 0),
        (2, 5, 0),
    ]
    assert ("met", "metrics") in seen[0].outputs
    assert met.seen_stages == [ExecutionStage.VAL] * 3
    assert not met.training  # the pass runs in eval mode


def test_forward_split_reads_the_frame_total_from_a_sized_dataset():
    pipe, _ = _pipeline()
    loader = _SizedLoader(_batches(), frames=5)

    seen = list(forward_split(pipe, loader, stage=ExecutionStage.TEST))

    assert [b.frames_total for b in seen] == [5, 5, 5]


def test_forward_split_raises_when_the_stop_event_is_set():
    pipe, _ = _pipeline()
    stop = threading.Event()
    gen = forward_split(pipe, _batches(), stage=ExecutionStage.VAL, stop_event=stop)
    next(gen)
    stop.set()

    with pytest.raises(TrainingCancelled):
        next(gen)


def test_evaluate_split_pools_auroc_and_averages_per_batch_values():
    pipe, met = _pipeline()

    summary = evaluate_split(
        pipe, _batches(), stage=ExecutionStage.VAL, metric_nodes=[met], epoch=3
    )

    assert summary.frames == 5
    assert summary.metrics["met/auroc"] == pytest.approx(1.0)  # pooled over the split
    assert summary.metrics["met/mean_score"] == pytest.approx((0.15 + 0.85 + 0.3) / 3)


def test_evaluate_split_resets_pooled_metrics_before_the_pass():
    pipe, met = _pipeline()
    # Inverted junk from an earlier pass: without a reset the pooled AUROC would drop.
    met.auroc.update(torch.tensor([0.9, 0.1]), torch.tensor([0, 1]))

    summary = evaluate_split(
        pipe, _batches(), stage=ExecutionStage.TEST, metric_nodes=[met]
    )

    assert summary.metrics["met/auroc"] == pytest.approx(1.0)


def test_evaluate_split_reports_progress_and_handles_an_empty_loader():
    pipe, met = _pipeline()
    seen: list[tuple[int, int]] = []

    evaluate_split(
        pipe,
        _batches(),
        stage=ExecutionStage.VAL,
        metric_nodes=[met],
        on_batch=lambda done, total: seen.append((done, total)),
    )
    empty = evaluate_split(pipe, [], stage=ExecutionStage.VAL, metric_nodes=[met])

    assert seen == [(2, 0), (4, 0), (5, 0)]
    assert empty.metrics == {} and empty.frames == 0


def test_statistical_trainer_validate_and_test_reduce_through_the_shared_pass():
    import pytorch_lightning as pl

    from cuvis_ai_core.training.trainers import StatisticalTrainer

    class _SplitsModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.val_ds = _batches()
            self.test_ds = _batches()

        def val_dataloader(self):
            return self.val_ds

        def test_dataloader(self):
            return self.test_ds

    pipe, met = _pipeline()
    trainer = StatisticalTrainer(pipeline=pipe, datamodule=_SplitsModule())

    validation = trainer.validate()
    test = trainer.test()

    assert validation.metrics["met/auroc"] == pytest.approx(1.0)
    assert (validation.frames, test.frames) == (5, 5)
    assert met.seen_stages == [ExecutionStage.VAL] * 3 + [ExecutionStage.TEST] * 3


def test_cuda_out_of_memory_is_recognised_through_its_causes():
    allocator = torch.cuda.OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 2.00 GiB"
    )
    runtime = RuntimeError("CUDA error: out of memory")
    cublas = RuntimeError(
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"
    )
    loader = RuntimeError("Node 'det' failed to prepare for inference")
    loader.__cause__ = allocator
    assert all(
        is_cuda_out_of_memory(exc) for exc in (allocator, runtime, cublas, loader)
    )
    assert not any(
        is_cuda_out_of_memory(exc)
        for exc in (
            None,
            ValueError("out of memory"),
            RuntimeError("boom at inference"),
        )
    )
