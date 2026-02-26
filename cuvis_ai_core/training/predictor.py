"""Inference orchestrator for running pipelines with datamodules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context


class Predictor:
    """Run inference for a pipeline/datamodule pair using a trainer-like API."""

    def __init__(
        self, pipeline: CuvisPipeline, datamodule: pl.LightningDataModule
    ) -> None:
        self.pipeline = pipeline
        self.datamodule = datamodule

    def predict(
        self,
        max_batches: int | None = None,
        collect_outputs: bool = False,
    ) -> list[dict[tuple[str, str], Any]] | None:
        """Run `ExecutionStage.INFERENCE` over the datamodule's predict dataloader."""
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be None or a positive integer.")

        self.datamodule.setup(stage="predict")
        dataloaders = self.datamodule.predict_dataloader()

        for module in self.pipeline.torch_layers:
            module.eval()

        collected: list[dict[tuple[str, str], Any]] = []
        batch_idx = 0

        self._reset_nodes()
        try:
            with torch.no_grad():
                for batch in self._iter_batches(dataloaders):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    moved_batch = self._move_batch_to_device(batch)
                    context = Context(
                        stage=ExecutionStage.INFERENCE,
                        batch_idx=batch_idx,
                        global_step=batch_idx,
                    )
                    outputs = self.pipeline.forward(batch=moved_batch, context=context)
                    if collect_outputs:
                        collected.append(outputs)
                    batch_idx += 1
        finally:
            self._close_nodes()

        return collected if collect_outputs else None

    @staticmethod
    def _iter_batches(dataloaders: Any) -> Iterable[dict[str, Any]]:
        """Yield batch dictionaries from one or more dataloaders."""
        if isinstance(dataloaders, DataLoader):
            for batch in dataloaders:
                yield batch
            return

        if isinstance(dataloaders, Mapping):
            for loader in dataloaders.values():
                if loader is None:
                    continue
                for batch in loader:
                    yield batch
            return

        if isinstance(dataloaders, Iterable):
            for loader in dataloaders:
                if loader is None:
                    continue
                for batch in loader:
                    yield batch
            return

        raise TypeError(
            "predict_dataloader() must return a DataLoader, mapping, or iterable of dataloaders."
        )

    def _get_pipeline_device(self) -> torch.device:
        """Resolve the active pipeline device from parameters or buffers."""
        for layer in self.pipeline.torch_layers:
            for param in layer.parameters():
                return param.device
            for buf in layer.buffers():
                return buf.device
        return torch.device("cpu")

    def _move_batch_to_device(self, batch: Any) -> dict[str, Any]:
        """Move tensor batch fields to the pipeline device."""
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be dict, got {type(batch)!r}.")

        device = self._get_pipeline_device()
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _reset_nodes(self) -> None:
        """Reset stateful nodes before a new prediction run."""
        for node in self.pipeline.nodes():
            reset_fn = getattr(node, "reset", None)
            if callable(reset_fn):
                reset_fn()

    def _close_nodes(self) -> None:
        """Close sink/writer nodes after prediction."""
        for node in self.pipeline.nodes():
            close_fn = getattr(node, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["Predictor"]
