"""Characterization tests: what each ``GradientTrainer`` step collects and logs."""

from __future__ import annotations

import warnings
from unittest.mock import Mock

import pytorch_lightning as pl
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import TrainingConfig
from cuvis_ai_core.training.trainers import GradientTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec


class _Source(Node):
    INPUT_SPECS = {}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

    def forward(self, **inputs):
        return {"out": torch.tensor(1.0)}


class _Loss(Node):
    INPUT_SPECS = {"value": PortSpec(dtype=torch.float32, shape=())}
    OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}
    EXECUTION_STAGES = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}

    def forward(self, value, **kwargs):
        return {"loss": value}


class _DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return [{"dummy": torch.tensor(1.0)}]


def _trainer() -> tuple[GradientTrainer, Mock]:
    pipeline = CuvisPipeline("steps")
    loss = _Loss(name="loss")
    pipeline.connect(_Source().outputs.out, loss.value)
    monitor = Mock()
    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=_DataModule(),
        training_config=TrainingConfig(max_epochs=1),
        loss_nodes=[loss],
        monitors=[monitor],
    )
    trainer.setup("fit")
    trainer._collect_metrics = Mock(wraps=trainer._collect_metrics)
    trainer._collect_losses = Mock(wraps=trainer._collect_losses)
    return trainer, monitor


def _run(step, batch_idx: int = 0) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*self\.log\(\).*", category=UserWarning
        )
        return step({"dummy": torch.tensor(1.0)}, batch_idx)


def test_training_step_collects_losses_only_and_logs_train_loss() -> None:
    trainer, monitor = _trainer()

    loss = _run(trainer.training_step)

    assert trainer._collect_losses.call_args.args[1] == "train"
    trainer._collect_metrics.assert_not_called()
    monitor.log.assert_called_once_with("train/loss", loss, step=trainer.global_step)


def test_validation_step_collects_metrics_and_logs_val_loss() -> None:
    trainer, monitor = _trainer()

    loss = _run(trainer.validation_step)

    assert trainer._collect_losses.call_args.args[1] == "val"
    trainer._collect_metrics.assert_called_once()
    monitor.log.assert_called_once_with("val/loss", loss, step=trainer.global_step)


def test_test_step_collects_metrics_and_does_not_log() -> None:
    trainer, monitor = _trainer()

    _run(trainer.test_step)

    assert trainer._collect_losses.call_args.args[1] == "test"
    trainer._collect_metrics.assert_called_once()
    monitor.log.assert_not_called()
