"""Tests for the Predictor inference orchestrator."""

from __future__ import annotations

import pytorch_lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.predictor import Predictor
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


class PredictSourceNode(Node):
    INPUT_SPECS = {
        "value": PortSpec(
            dtype=torch.float32, shape=(-1, -1), description="Input value"
        ),
    }
    OUTPUT_SPECS = {
        "doubled": PortSpec(
            dtype=torch.float32, shape=(-1, -1), description="Doubled value"
        ),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.contexts: list[Context] = []
        self.input_devices: list[torch.device] = []

    def forward(self, value: torch.Tensor, context: Context | None = None, **_) -> dict:
        if context is not None:
            self.contexts.append(context)
        self.input_devices.append(value.device)
        return {"doubled": value * 2.0}


class PredictSinkNode(Node):
    INPUT_SPECS = {
        "doubled": PortSpec(
            dtype=torch.float32, shape=(-1, -1), description="Input stream"
        ),
    }
    OUTPUT_SPECS = {}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.forward_calls = 0
        self.reset_calls = 0
        self.close_calls = 0

    def forward(
        self, doubled: torch.Tensor, context: Context | None = None, **_
    ) -> dict:
        del doubled, context
        self.forward_calls += 1
        return {}

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class DictDataset(Dataset):
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"value": self.values[idx]}


class PredictDataModule(pl.LightningDataModule):
    def __init__(self, values: torch.Tensor, batch_size: int = 1) -> None:
        super().__init__()
        self.values = values
        self.batch_size = batch_size
        self.predict_ds: DictDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "predict" or stage is None:
            self.predict_ds = DictDataset(self.values)

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise RuntimeError("predict dataset not initialized")
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False)


class BadPredictDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.predict_ds: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "predict" or stage is None:
            self.predict_ds = TensorDataset(torch.ones(2, 1))

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise RuntimeError("predict dataset not initialized")
        return DataLoader(self.predict_ds, batch_size=1, shuffle=False)


def _build_pipeline() -> tuple[CuvisPipeline, PredictSourceNode, PredictSinkNode]:
    pipeline = CuvisPipeline("predict_pipeline")
    source = PredictSourceNode(name="source")
    sink = PredictSinkNode(name="sink")
    pipeline.connect(source.outputs.doubled, sink.inputs.doubled)
    return pipeline, source, sink


def test_predictor_runs_inference_with_context_and_hooks() -> None:
    pipeline, source, sink = _build_pipeline()
    datamodule = PredictDataModule(
        values=torch.tensor([[1.0], [2.0], [3.0]]), batch_size=1
    )

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    outputs = predictor.predict(collect_outputs=True)

    assert outputs is not None
    assert len(outputs) == 3
    assert all(("source", "doubled") in out for out in outputs)

    assert sink.reset_calls == 1
    assert sink.close_calls == 1
    assert sink.forward_calls == 3

    assert [ctx.stage for ctx in source.contexts] == [
        ExecutionStage.INFERENCE,
        ExecutionStage.INFERENCE,
        ExecutionStage.INFERENCE,
    ]
    assert [ctx.batch_idx for ctx in source.contexts] == [0, 1, 2]
    assert all(device.type == "cpu" for device in source.input_devices)


def test_predictor_max_batches_limits_iteration() -> None:
    pipeline, source, sink = _build_pipeline()
    datamodule = PredictDataModule(
        values=torch.tensor([[1.0], [2.0], [3.0]]), batch_size=1
    )

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    result = predictor.predict(max_batches=2, collect_outputs=False)

    assert result is None
    assert sink.forward_calls == 2
    assert sink.close_calls == 1
    assert [ctx.batch_idx for ctx in source.contexts] == [0, 1]


def test_predictor_rejects_non_positive_max_batches() -> None:
    pipeline, _, _ = _build_pipeline()
    datamodule = PredictDataModule(values=torch.tensor([[1.0]]), batch_size=1)

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    with pytest.raises(ValueError, match="max_batches"):
        predictor.predict(max_batches=0)


def test_predictor_rejects_non_dict_batch() -> None:
    pipeline, _, _ = _build_pipeline()
    datamodule = BadPredictDataModule()

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    with pytest.raises(TypeError, match="Expected batch to be dict"):
        predictor.predict()
