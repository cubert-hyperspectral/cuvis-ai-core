"""The pipeline's module list follows the graph, and a trainer freezes the graph."""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import TrainingConfig
from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec

_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


class Source(Node):
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"out": _VEC}

    def forward(self, **inputs):
        return {"out": torch.zeros(2)}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class Scale(Node):
    """One learnable parameter, so the module list and ``parameters()`` are observable."""

    INPUT_SPECS = {"x": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x, **inputs):
        return {"out": x * self.weight}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class Loss(Node):
    INPUT_SPECS = {"value": _VEC}
    OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}
    EXECUTION_STAGES = {ExecutionStage.TRAIN, ExecutionStage.VAL}

    def forward(self, value, **inputs):
        return {"loss": value.sum()}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class FitProbe(Node):
    """Statistical node whose fit runs a callable the test plants."""

    INPUT_SPECS = {"x": _VEC}
    OUTPUT_SPECS = {"out": _VEC}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.during_fit = None

    def forward(self, x, **inputs):
        return {"out": x}

    def statistical_initialization(self, input_stream: InputStream) -> None:
        for _ in input_stream:
            pass
        if self.during_fit is not None:
            self.during_fit()

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class _ListDataModule(pl.LightningDataModule):
    def __init__(self, batches):
        super().__init__()
        self._batches = batches

    def train_dataloader(self):
        return self._batches


def _two_stage_pipeline() -> tuple[CuvisPipeline, Scale]:
    pipeline = CuvisPipeline("cache")
    source, first = Source(name="source"), Scale(name="first")
    pipeline.connect(source.outputs.out, first.inputs.x)
    return pipeline, first


def _gradient_trainer(pipeline: CuvisPipeline, loss: Loss) -> GradientTrainer:
    return GradientTrainer(
        pipeline=pipeline,
        datamodule=_ListDataModule([]),
        loss_nodes=[loss],
        training_config=TrainingConfig(max_epochs=1),
    )


def test_connect_after_parameters_yields_the_new_nodes_parameters():
    pipeline, first = _two_stage_pipeline()
    assert [id(p) for p in pipeline.parameters()] == [id(first.weight)]

    second = Scale(name="second")
    pipeline.connect(first.outputs.out, second.inputs.x)

    assert [id(p) for p in pipeline.parameters()] == [
        id(first.weight),
        id(second.weight),
    ]
    assert second in list(pipeline.torch_layers)


def test_to_reaches_a_node_connected_after_the_first_move():
    pipeline, first = _two_stage_pipeline()
    pipeline.to(torch.float64)

    second = Scale(name="second")
    pipeline.connect(first.outputs.out, second.inputs.x)
    pipeline.to(torch.float64)

    assert second.weight.dtype == torch.float64


def test_cleanup_empties_the_module_list_and_lifts_the_freeze():
    pipeline, _ = _two_stage_pipeline()
    assert len(pipeline.torch_layers) == 2
    pipeline.freeze_structure("a test")
    assert pipeline.structure_frozen_by == "a test"

    pipeline.cleanup()

    assert len(pipeline.torch_layers) == 0
    assert pipeline.structure_frozen_by is None
    pipeline.connect(Source(name="again").outputs.out, Scale(name="third").inputs.x)
    assert len(pipeline.torch_layers) == 2


def test_gradient_trainer_freezes_the_structure():
    pipeline, first = _two_stage_pipeline()
    loss = Loss(name="loss")
    pipeline.connect(first.outputs.out, loss.inputs.value)
    trainer = _gradient_trainer(pipeline, loss)
    late = Scale(name="late")

    with pytest.raises(RuntimeError, match="GradientTrainer"):
        pipeline.connect(first.outputs.out, late.inputs.x)
    trainer.configure_optimizers()
    with pytest.raises(RuntimeError, match="GradientTrainer"):
        pipeline.connect(first.outputs.out, late.inputs.x)

    assert late not in pipeline.nodes()
    assert len(pipeline.torch_layers) == 3


def test_statistical_trainer_freezes_the_structure_while_fitting():
    pipeline = CuvisPipeline("stat")
    source, probe = Source(name="source"), FitProbe(name="probe")
    pipeline.connect(source.outputs.out, probe.inputs.x)
    late = Scale(name="late")
    probe.during_fit = lambda: pipeline.connect(probe.outputs.out, late.inputs.x)
    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=_ListDataModule([{}]))

    with pytest.raises(RuntimeError, match="StatisticalTrainer"):
        trainer.fit()

    assert pipeline.structure_frozen_by is None
    pipeline.connect(probe.outputs.out, late.inputs.x)
    assert late in pipeline.nodes()
