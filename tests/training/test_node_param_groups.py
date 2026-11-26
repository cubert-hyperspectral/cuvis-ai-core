"""Nodes contribute optimizer learning-rate groups through ``Node.optimizer_param_groups``.

``GradientTrainer.configure_optimizers`` asks every node for its groups: a group keeps
its own options (``lr``, ``weight_decay``), only trainable parameters count, a
parameter in two groups stays in the first, and every trainable parameter no group
lists is optimized at the configured rate. Without any group the optimizer is built
exactly as before.
"""

from __future__ import annotations

import pytorch_lightning as pl
import pytest
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from cuvis_ai_core.training.trainers import GradientTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

pytestmark = pytest.mark.unit


class _TwoParamNode(Node):
    """A trainable node with a backbone and a head parameter."""

    INPUT_SPECS = {}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

    def __init__(self, groups=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = torch.nn.Parameter(torch.tensor(1.0))
        self.head = torch.nn.Parameter(torch.tensor(2.0))
        self._groups = groups

    def optimizer_param_groups(self, base_lr: float):
        return self._groups(self, base_lr) if self._groups else None

    def forward(self, **inputs):
        return {"out": self.backbone + self.head}


class _LossNode(Node):
    INPUT_SPECS = {"value": PortSpec(dtype=torch.float32, shape=())}
    OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}
    EXECUTION_STAGES = {ExecutionStage.TRAIN}

    def forward(self, value, **kwargs):
        return {"loss": value}


def _optimizer(groups=None, freeze_backbone=False, scheduler=None):
    pipeline = CuvisPipeline("param_groups")
    node = _TwoParamNode(groups=groups, name="model")
    loss = _LossNode(name="loss")
    pipeline.connect(node.outputs.out, loss.value)
    if freeze_backbone:
        node.backbone.requires_grad_(False)
    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=pl.LightningDataModule(),
        loss_nodes=[],
        training_config=TrainingConfig(
            max_epochs=4,
            optimizer=OptimizerConfig(name="adamw", lr=0.01, weight_decay=0.05),
            scheduler=scheduler,
        ),
    )
    result = trainer.configure_optimizers()
    optimizer = result["optimizer"] if isinstance(result, dict) else result
    return optimizer, node


def _names(group, node):
    by_id = {id(node.backbone): "backbone", id(node.head): "head"}
    return [by_id[id(p)] for p in group["params"]]


def test_without_groups_the_optimizer_is_unchanged() -> None:
    optimizer, node = _optimizer()
    assert len(optimizer.param_groups) == 1
    assert sorted(_names(optimizer.param_groups[0], node)) == ["backbone", "head"]
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_node_group_keeps_its_options_and_the_rest_uses_the_config() -> None:
    def groups(node, base_lr):
        return [{"params": [node.backbone], "lr": base_lr * 0.1, "weight_decay": 0.0}]

    optimizer, node = _optimizer(groups)
    first, rest = optimizer.param_groups
    assert _names(first, node) == ["backbone"]
    assert first["lr"] == pytest.approx(0.001)
    assert first["weight_decay"] == 0.0
    assert _names(rest, node) == ["head"]
    assert rest["lr"] == 0.01
    assert rest["weight_decay"] == 0.05


def test_a_group_without_options_takes_the_config_values() -> None:
    optimizer, node = _optimizer(lambda n, lr: [{"params": [n.head]}])
    first = optimizer.param_groups[0]
    assert _names(first, node) == ["head"]
    assert first["lr"] == 0.01 and first["weight_decay"] == 0.05


def test_frozen_parameters_and_empty_groups_are_dropped() -> None:
    def groups(node, base_lr):
        return [
            {"params": [node.backbone], "lr": 0.5},  # frozen below: dropped
            {"params": [node.head], "lr": 0.2},
        ]

    optimizer, node = _optimizer(groups, freeze_backbone=True)
    assert len(optimizer.param_groups) == 1
    assert _names(optimizer.param_groups[0], node) == ["head"]
    assert optimizer.param_groups[0]["lr"] == 0.2


def test_a_parameter_in_two_groups_stays_in_the_first() -> None:
    def groups(node, base_lr):
        return [
            {"params": [node.head], "lr": 0.3},
            {"params": [node.head, node.backbone], "lr": 0.4},
        ]

    optimizer, node = _optimizer(groups)
    assert [_names(g, node) for g in optimizer.param_groups] == [["head"], ["backbone"]]
    assert [g["lr"] for g in optimizer.param_groups] == [0.3, 0.4]


def test_a_single_tensor_as_params_is_accepted() -> None:
    optimizer, node = _optimizer(lambda n, lr: [{"params": n.head, "lr": 0.3}])
    assert _names(optimizer.param_groups[0], node) == ["head"]


def test_scheduler_scales_every_group() -> None:
    def groups(node, base_lr):
        return [{"params": [node.backbone], "lr": base_lr * 0.1}]

    optimizer, _node = _optimizer(
        groups, scheduler=SchedulerConfig(name="step", step_size=1, gamma=0.5)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([0.0005, 0.005])
