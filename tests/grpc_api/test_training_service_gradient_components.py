"""The validation errors of ``TrainingService._configure_gradient_components``."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.training_service import TrainingService
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import DataConfig, TrainingConfig, TrainRunConfig
from cuvis_ai_schemas.pipeline import PortSpec


class _Source(Node):
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"data": PortSpec(torch.Tensor, (-1, 3))}

    def forward(self, **inputs):
        return {"data": torch.zeros(1, 3)}


class _Scale(Node):
    INPUT_SPECS = {"data": PortSpec(torch.Tensor, (-1, 3))}
    OUTPUT_SPECS = {"scaled": PortSpec(torch.Tensor, (-1, 3))}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gain = torch.nn.Parameter(torch.ones(1))

    def forward(self, data, **inputs):
        return {"scaled": data * self.gain}


def _trainrun(**overrides) -> TrainRunConfig:
    fields = dict(
        name="t",
        pipeline=None,
        data=DataConfig(),
        training=TrainingConfig(),
        loss_nodes=["scale"],
        metric_nodes=[],
        freeze_nodes=[],
        unfreeze_nodes=[],
        output_dir=".",
        tags={},
    )
    fields.update(overrides)
    return TrainRunConfig(**fields)


@pytest.fixture
def session():
    manager = SessionManager()
    state = manager.get_session(manager.create_session())
    pipeline = CuvisPipeline("p")
    source, scale = _Source(name="source"), _Scale(name="scale")
    pipeline.connect(source.outputs.data, scale.inputs.data)
    state.pipeline = pipeline
    state.trainrun_config = _trainrun()
    return state


@pytest.fixture
def service():
    return TrainingService(SessionManager())


def test_missing_trainrun_config_is_rejected(service, session):
    session.trainrun_config = None
    with pytest.raises(ValueError, match="requires explicit TrainRunConfig"):
        service._configure_gradient_components(session)


def test_empty_loss_nodes_are_rejected(service, session):
    session.trainrun_config = _trainrun(loss_nodes=[])
    with pytest.raises(ValueError, match="at least one loss node"):
        service._configure_gradient_components(session)


def test_unknown_loss_node_names_the_pipeline_nodes(service, session):
    session.trainrun_config = _trainrun(loss_nodes=["ghost"])
    with pytest.raises(
        ValueError,
        match="Loss node 'ghost' not found in pipeline. Available nodes: scale, source",
    ):
        service._configure_gradient_components(session)


def test_unknown_metric_node_is_rejected(service, session):
    session.trainrun_config = _trainrun(metric_nodes=["ghost"])
    with pytest.raises(ValueError, match="Metric node 'ghost' not found in pipeline"):
        service._configure_gradient_components(session)


def test_frozen_pipeline_without_unfreeze_is_rejected(service, session):
    for param in session.pipeline.parameters():
        param.requires_grad_(False)
    with pytest.raises(ValueError, match="No trainable parameters found"):
        service._configure_gradient_components(session)


def test_resolves_nodes_by_name_and_unfreezes_the_requested_ones(service, session):
    for param in session.pipeline.parameters():
        param.requires_grad_(False)
    session.trainrun_config = _trainrun(
        loss_nodes=["scale"], metric_nodes=["source"], unfreeze_nodes=["scale"]
    )
    loss_nodes, metric_nodes = service._configure_gradient_components(session)
    assert [node.name for node in loss_nodes] == ["scale"]
    assert [node.name for node in metric_nodes] == ["source"]
    assert all(param.requires_grad for param in session.pipeline.parameters())
