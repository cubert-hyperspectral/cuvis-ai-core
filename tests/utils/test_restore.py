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
    assert all(
        batch["cube"].device.type == "cpu" for batch, _ in fake_pipeline.forward_calls
    )
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


def test_restore_pipeline_rejects_removed_plugins_path_kwarg(
    tmp_path: Path,
) -> None:
    """The legacy ``plugins_path`` aggregator-manifest argument is removed.

    Plugin loading goes through ``plugins_dirs`` / ``--plugins-dir`` (the
    ``configs/plugins/`` catalog) plus the pipeline's ``plugins:`` field.
    Passing the removed keyword must fail loudly, not silently no-op.
    """
    with pytest.raises(TypeError, match="plugins_path"):
        restore_mod.restore_pipeline(
            pipeline_path=tmp_path / "pipeline.yaml",
            device="cpu",
            plugins_path=tmp_path / "plugins.yaml",
        )


def test_restore_pipeline_cli_has_no_plugins_path_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``restore-pipeline`` no longer accepts ``--plugins-path``."""
    monkeypatch.setattr(restore_mod, "restore_pipeline", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore-pipeline",
            "--pipeline-path",
            "configs/pipeline.yaml",
            "--plugins-path",
            "configs/plugins/adaclip.yaml",
        ],
    )

    with pytest.raises(SystemExit):
        restore_mod.restore_pipeline_cli()


# ---------------------------------------------------------------------------
# _discover_plugins_dirs
# ---------------------------------------------------------------------------


def test_discover_plugins_dirs_walks_up_and_appends_explicit(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    plugins = project / "configs" / "plugins"
    plugins.mkdir(parents=True)
    pipeline_path = project / "nested" / "pipe.yaml"
    pipeline_path.parent.mkdir(parents=True)
    pipeline_path.write_text("nodes: []\n", encoding="utf-8")

    extra = tmp_path / "extra_plugins"
    extra.mkdir()

    found = restore_mod._discover_plugins_dirs(pipeline_path, [str(extra)])
    assert plugins.resolve() in found
    assert extra in found


# ---------------------------------------------------------------------------
# restore_pipeline: info mode (no cu3s) + visualization export
# ---------------------------------------------------------------------------


class _SpecPipeline:
    """Fake pipeline exposing only what info-mode + visualize need."""

    def __init__(self) -> None:
        self.visualize_calls: list[dict] = []

    def get_input_specs(self):
        return {"cube": "spec"}

    def get_output_specs(self):
        return {"out": "spec"}

    def visualize(self, *, format, output_path):
        self.visualize_calls.append({"format": format, "output_path": output_path})


def _write_no_plugin_pipeline(path: Path) -> None:
    path.write_text(
        "metadata:\n"
        "  name: T\n"
        "  description: d\n"
        "  created: '2024-01-01'\n"
        "  tags: []\n"
        "plugins: []\n"
        "nodes:\n"
        "  - name: data\n"
        "    class_name: tests.fixtures.mock_nodes.LentilsAnomalyDataNode\n"
        "    hparams:\n"
        "      normal_class_ids: [0, 1]\n"
        "connections: []\n",
        encoding="utf-8",
    )


def test_restore_pipeline_info_mode_no_cu3s(monkeypatch, tmp_path: Path) -> None:
    pipeline_path = tmp_path / "pipe.yaml"
    _write_no_plugin_pipeline(pipeline_path)
    fake = _SpecPipeline()
    monkeypatch.setattr(
        restore_mod.CuvisPipeline,
        "load_pipeline",
        staticmethod(lambda *a, **k: fake),
    )
    result = restore_mod.restore_pipeline(
        pipeline_path=pipeline_path,
        weights_path=tmp_path / "explicit.pt",
        device="cpu",
    )
    assert result is fake


@pytest.mark.parametrize(
    "ext, suffix, expected_format",
    [
        (restore_mod.PipelineVisFormat.PNG, ".png", "render"),
        (restore_mod.PipelineVisFormat.MD, ".md", "render_mermaid"),
    ],
)
def test_restore_pipeline_exports_visualization(
    monkeypatch, tmp_path: Path, ext, suffix, expected_format
) -> None:
    pipeline_path = tmp_path / "pipe.yaml"
    _write_no_plugin_pipeline(pipeline_path)
    fake = _SpecPipeline()
    monkeypatch.setattr(
        restore_mod.CuvisPipeline,
        "load_pipeline",
        staticmethod(lambda *a, **k: fake),
    )
    restore_mod.restore_pipeline(
        pipeline_path=pipeline_path, device="cpu", pipeline_vis_ext=ext
    )
    assert fake.visualize_calls[0]["format"] == expected_format
    assert fake.visualize_calls[0]["output_path"] == pipeline_path.with_suffix(suffix)


# ---------------------------------------------------------------------------
# restore_trainrun
# ---------------------------------------------------------------------------


def _write_trainrun(tmp_path: Path, mock_experiment_dict: dict) -> Path:
    import yaml

    cfg = dict(mock_experiment_dict)
    cfg["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_restore_trainrun_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        restore_mod.restore_trainrun(tmp_path / "absent.yaml")


def test_restore_trainrun_info_mode_builds_and_returns(
    tmp_path: Path, mock_experiment_dict
) -> None:
    path = _write_trainrun(tmp_path, mock_experiment_dict)
    # info mode builds the pipeline (device != auto exercises the .to() move)
    # then prints specs and returns without touching data or trainers.
    assert restore_mod.restore_trainrun(path, mode="info", device="cpu") is None


class _RecordingTrainer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []

    def fit(self) -> None:
        self.calls.append("fit")

    def validate(self) -> None:
        self.calls.append("validate")

    def test(self) -> None:
        self.calls.append("test")


class _FakeDataModule:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def setup(self, stage=None) -> None:
        pass


@pytest.mark.parametrize("mode", ["train", "validate", "test"])
def test_restore_trainrun_execution_modes_with_mocked_trainers(
    monkeypatch, tmp_path: Path, mock_experiment_dict, mode
) -> None:
    monkeypatch.setattr(restore_mod, "SingleCu3sDataModule", _FakeDataModule)
    monkeypatch.setattr(restore_mod, "StatisticalTrainer", _RecordingTrainer)
    monkeypatch.setattr(restore_mod, "GradientTrainer", _RecordingTrainer)

    path = _write_trainrun(tmp_path, mock_experiment_dict)
    # Should run end-to-end against the fake datamodule/trainers (no data).
    assert restore_mod.restore_trainrun(path, mode=mode, device="cpu") is None


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def test_restore_pipeline_cli_passes_vis_ext(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        restore_mod, "restore_pipeline", lambda **kw: captured.update(kw)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore-pipeline",
            "--pipeline-path",
            "configs/pipeline.yaml",
            "--pipeline-vis-ext",
            "md",
        ],
    )
    restore_mod.restore_pipeline_cli()
    assert captured["pipeline_vis_ext"] == restore_mod.PipelineVisFormat.MD


def test_restore_trainrun_cli_forwards_args(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        restore_mod, "restore_trainrun", lambda **kw: captured.update(kw)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore-trainrun",
            "--trainrun-path",
            "outputs/run.yaml",
            "--mode",
            "train",
            "--override",
            "data.batch_size=8",
        ],
    )
    restore_mod.restore_trainrun_cli()
    assert captured["trainrun_path"] == "outputs/run.yaml"
    assert captured["mode"] == "train"
    assert captured["overrides"] == ["data.batch_size=8"]
