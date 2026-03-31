"""Tests for restore helpers and CLI parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import cuvis_ai_core.utils.restore as restore_mod
from cuvis_ai_schemas.enums import ExecutionStage


class _FakeDataset:
    instances: list["_FakeDataset"] = []

    def __init__(
        self,
        *,
        cu3s_file_path: str,
        processing_mode: str,
        annotation_json_path: str | None,
        measurement_indices: list[int] | None,
    ) -> None:
        self.cu3s_file_path = cu3s_file_path
        self.processing_mode = processing_mode
        self.annotation_json_path = annotation_json_path
        self.measurement_indices = measurement_indices
        self.samples = [
            {"cube": torch.tensor([1.0], dtype=torch.float32)},
            {"cube": torch.tensor([2.0], dtype=torch.float32)},
        ]
        self.__class__.instances.append(self)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class _FakeTorchLayer:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


class _FakeVideoNode:
    def __init__(self) -> None:
        self.output_video_path = "out.mp4"
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakePipeline:
    def __init__(self) -> None:
        self.torch_layers = [_FakeTorchLayer()]
        self.video_node = _FakeVideoNode()
        self.nodes = [self.video_node, object()]
        self.profiling_enabled: list[bool] = []
        self.forward_calls: list[tuple[dict[str, torch.Tensor], object]] = []
        self.summary_calls: list[tuple[ExecutionStage, int]] = []

    def set_profiling(self, *, enabled: bool) -> None:
        self.profiling_enabled.append(enabled)

    def forward(self, *, batch: dict[str, torch.Tensor], context: object) -> dict:
        self.forward_calls.append((batch, context))
        return {}

    def format_profiling_summary(
        self,
        *,
        stage: ExecutionStage,
        total_frames: int,
    ) -> str:
        self.summary_calls.append((stage, total_frames))
        return "profiling-summary"

    def get_input_specs(self) -> dict[str, str]:
        return {"cube": "spec"}

    def get_output_specs(self) -> dict[str, str]:
        return {"node.output": "spec"}


def test_restore_module_imports() -> None:
    """Verify restore module imports resolve correctly after schema migration."""
    from cuvis_ai_core.utils.restore import Context, ExecutionStage

    assert ExecutionStage is not None
    assert Context is not None


def test_restore_pipeline_runs_inference_with_profiling_and_video_finalize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_pipeline = _FakePipeline()
    tqdm_calls: list[dict[str, object]] = []

    def _fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    _FakeDataset.instances.clear()
    monkeypatch.setattr(restore_mod, "SingleCu3sDataset", _FakeDataset)
    monkeypatch.setattr(restore_mod, "tqdm", _fake_tqdm)
    monkeypatch.setattr(
        restore_mod.CuvisPipeline,
        "load_pipeline",
        staticmethod(lambda *args, **kwargs: fake_pipeline),
    )

    pipeline = restore_mod.restore_pipeline(
        pipeline_path=tmp_path / "pipeline.yaml",
        device="cpu",
        cu3s_file_path=tmp_path / "sample.cu3s",
        processing_mode="SpectralRadiance",
        annotation_json_path=tmp_path / "sample.json",
        measurement_indices=[0, 3],
    )

    assert pipeline is fake_pipeline
    assert len(_FakeDataset.instances) == 1
    dataset = _FakeDataset.instances[0]
    assert dataset.processing_mode == "SpectralRadiance"
    assert dataset.annotation_json_path == str(tmp_path / "sample.json")
    assert dataset.measurement_indices == [0, 3]

    assert fake_pipeline.torch_layers[0].eval_calls == 1
    assert fake_pipeline.profiling_enabled == [True]
    assert len(fake_pipeline.forward_calls) == 2
    assert [ctx.batch_idx for _, ctx in fake_pipeline.forward_calls] == [0, 1]
    assert [ctx.global_step for _, ctx in fake_pipeline.forward_calls] == [0, 1]
    assert all(batch["cube"].device.type == "cpu" for batch, _ in fake_pipeline.forward_calls)
    assert fake_pipeline.summary_calls == [(ExecutionStage.INFERENCE, 2)]
    assert fake_pipeline.video_node.close_calls == 1
    assert tqdm_calls == [{"total": 2, "desc": "Inference", "unit": "frame"}]


def test_restore_pipeline_cli_parses_measurement_indices_and_annotation_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_restore_pipeline(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(restore_mod, "restore_pipeline", _fake_restore_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore-pipeline",
            "--pipeline-path",
            "configs/pipeline.yaml",
            "--cu3s-file-path",
            "data/sample.cu3s",
            "--annotation-json-path",
            "data/sample.json",
            "--measurement-indices",
            "0, 2,5",
            "--processing-mode",
            "SpectralRadiance",
        ],
    )

    restore_mod.restore_pipeline_cli()

    assert captured["pipeline_path"] == "configs/pipeline.yaml"
    assert captured["cu3s_file_path"] == "data/sample.cu3s"
    assert captured["annotation_json_path"] == "data/sample.json"
    assert captured["measurement_indices"] == [0, 2, 5]
    assert captured["processing_mode"] == "SpectralRadiance"
