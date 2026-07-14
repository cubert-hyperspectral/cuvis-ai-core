"""Tests for restore helpers and CLI parsing."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

import cuvis_ai_core.utils.restore as restore_mod
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from tests.fixtures.restore_doubles import (
    FakeRestoreDataModule,
    FakeRestorePipeline,
    RecordingTrainer,
)


def test_restore_module_imports() -> None:
    """Verify restore module imports resolve correctly after schema migration."""
    from cuvis_ai_core.utils.restore import Context, ExecutionStage

    assert ExecutionStage is not None
    assert Context is not None


def test_restore_pipeline_runs_inference_with_profiling_and_video_finalize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_pipeline = FakeRestorePipeline()
    tqdm_calls: list[dict[str, object]] = []

    def _fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    FakeRestoreDataModule.instances.clear()

    def _fake_build_data_module(registry, data_config, candidate_dirs):
        return FakeRestoreDataModule(data_config)

    monkeypatch.setattr(restore_mod, "_build_data_module", _fake_build_data_module)
    monkeypatch.setattr(restore_mod, "tqdm", _fake_tqdm)
    monkeypatch.setattr(
        restore_mod.CuvisPipeline,
        "load_pipeline",
        staticmethod(lambda *args, **kwargs: fake_pipeline),
    )

    pipeline = restore_mod.restore_pipeline(
        pipeline_path=tmp_path / "pipeline.yaml",
        device="cpu",
        data_module="cu3s",
        data_args={
            "cu3s_file_path": str(tmp_path / "sample.cu3s"),
            "processing_mode": "SpectralRadiance",
            "annotation_json_path": str(tmp_path / "sample.json"),
        },
    )

    assert pipeline is fake_pipeline
    assert len(FakeRestoreDataModule.instances) == 1
    datamodule = FakeRestoreDataModule.instances[0]
    assert datamodule.data_config.data_module == "cu3s"
    assert datamodule.data_config.params["processing_mode"] == "SpectralRadiance"
    assert datamodule.data_config.params["annotation_json_path"] == str(
        tmp_path / "sample.json"
    )
    assert datamodule.setup_stages == ["predict"]

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
    assert tqdm_calls == [{"desc": "Inference", "unit": "batch"}]


def test_restore_pipeline_cli_parses_data_module_and_data_args(
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
            "--data-module",
            "cu3s",
            "--data-arg",
            "cu3s_file_path=data/sample.cu3s",
            "--data-arg",
            "annotation_json_path=data/sample.json",
            "--data-arg",
            "processing_mode=SpectralRadiance",
        ],
    )

    restore_mod.restore_pipeline_cli()

    assert captured["pipeline_path"] == "configs/pipeline.yaml"
    assert captured["data_module"] == "cu3s"
    assert captured["data_args"] == {
        "cu3s_file_path": "data/sample.cu3s",
        "annotation_json_path": "data/sample.json",
        "processing_mode": "SpectralRadiance",
    }


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
# Plugin materialisation: _load_data_module_plugin + declared pipeline plugins
# ---------------------------------------------------------------------------


def _write_plugins_dir(tmp_path: Path, name: str, body: str) -> Path:
    pdir = tmp_path / "plugins"
    pdir.mkdir(exist_ok=True)
    (pdir / f"{name}.yaml").write_text(dedent(body), encoding="utf-8")
    return pdir


def test_load_data_module_plugin_registers_from_catalog(tmp_path: Path) -> None:
    """A data_module capability in the plugins-dir catalog is materialised by name."""
    plugins_dir = _write_plugins_dir(
        tmp_path,
        "fake_dataloader",
        """
        name: fake_dataloader
        path: "."
        package_name: fake_pkg
        capabilities:
          - class_name: tests.fixtures.fake_data_modules.FakeDataModule
            kind: data_module
            data_module_name: fake
        """,
    )
    registry = NodeRegistry()
    restore_mod._load_data_module_plugin(registry, "fake", [plugins_dir])
    assert "fake" in registry.data_modules
    assert registry.data_modules["fake"].DATA_MODULE_NAME == "fake"


def test_load_data_module_plugin_unknown_name_raises(tmp_path: Path) -> None:
    """A data module not provided by any catalog plugin fails loudly."""
    plugins_dir = _write_plugins_dir(
        tmp_path,
        "fake_dataloader",
        """
        name: fake_dataloader
        path: "."
        package_name: fake_pkg
        capabilities:
          - class_name: tests.fixtures.fake_data_modules.FakeDataModule
            kind: data_module
            data_module_name: fake
        """,
    )
    registry = NodeRegistry()
    with pytest.raises(ValueError, match="provides data module 'absent'"):
        restore_mod._load_data_module_plugin(registry, "absent", [plugins_dir])


def test_restore_pipeline_materialises_declared_plugins(
    monkeypatch, tmp_path: Path
) -> None:
    """A pipeline that declares a plugin gets a NodeRegistry with it installed."""
    plugins_dir = _write_plugins_dir(
        tmp_path,
        "fake_nodes",
        """
        name: fake_nodes
        path: "."
        package_name: fake_pkg
        capabilities:
          - class_name: tests.fixtures.mock_nodes.LentilsAnomalyDataNode
        """,
    )
    pipeline_path = tmp_path / "pipe.yaml"
    pipeline_path.write_text(
        "metadata:\n"
        "  name: T\n"
        "  description: d\n"
        "  created: '2024-01-01'\n"
        "  tags: []\n"
        "plugins: [fake_nodes]\n"
        "nodes:\n"
        "  - name: data\n"
        "    class_name: tests.fixtures.mock_nodes.LentilsAnomalyDataNode\n"
        "    hparams:\n"
        "      normal_class_ids: [0, 1]\n"
        "connections: []\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    fake = FakeRestorePipeline()

    def _fake_load(_path, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(
        restore_mod.CuvisPipeline, "load_pipeline", staticmethod(_fake_load)
    )
    result = restore_mod.restore_pipeline(
        pipeline_path=pipeline_path, device="cpu", plugins_dirs=[str(plugins_dir)]
    )
    assert result is fake
    registry = captured["node_registry"]
    assert registry is not None
    assert "fake_nodes" in registry.list_plugins()


# ---------------------------------------------------------------------------
# restore_pipeline: info mode (no cu3s) + visualization export
# ---------------------------------------------------------------------------


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
    fake = FakeRestorePipeline()
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
    fake = FakeRestorePipeline()
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


def _write_trainrun(
    tmp_path: Path, mock_experiment_dict: dict, mock_pipeline_dict: dict
) -> Path:
    import yaml

    cfg = dict(mock_experiment_dict)
    cfg["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # Materialise the pipeline the trainrun references (pipeline: pipeline.yaml).
    (tmp_path / cfg["pipeline"]).write_text(
        yaml.safe_dump(mock_pipeline_dict), encoding="utf-8"
    )
    return path


def test_restore_trainrun_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        restore_mod.restore_trainrun(tmp_path / "absent.yaml")


def test_restore_trainrun_info_mode_builds_and_returns(
    tmp_path: Path, mock_experiment_dict, mock_pipeline_dict
) -> None:
    path = _write_trainrun(tmp_path, mock_experiment_dict, mock_pipeline_dict)
    # info mode builds the pipeline (device != auto exercises the .to() move)
    # then prints specs and returns without touching data or trainers.
    assert restore_mod.restore_trainrun(path, mode="info", device="cpu") is None


_DS = object()  # sentinel: a non-None dataset is available


@pytest.mark.parametrize(
    "mode,use_gradient,requires_fit,val_ds,test_ds",
    [
        # stat-only (no GradientTrainer), requires_initial_fit=True, val+test present
        ("train", False, True, _DS, _DS),
        # stat-only, no val or test dataset configured
        ("train", False, False, None, None),
        # gradient training with val+test; stat.fit() also runs (requires_fit=True)
        ("train", True, True, _DS, _DS),
        # validate mode - stat only
        ("validate", False, True, None, None),
        # validate mode - gradient
        ("validate", True, False, None, None),
        # test mode - stat only
        ("test", False, True, None, None),
        # test mode - gradient
        ("test", True, False, None, None),
    ],
)
def test_restore_trainrun_execution_modes_with_mocked_trainers(
    monkeypatch,
    tmp_path: Path,
    mock_experiment_dict,
    statistical_experiment_dict,
    mock_pipeline_dict,
    mode,
    use_gradient,
    requires_fit,
    val_ds,
    test_ds,
) -> None:
    """Dispatch test: proves mode -> which trainer methods fire, and that save_to_file
    is only called in train mode.  No real training; all heavy components are patched.
    """
    exp_dict = mock_experiment_dict if use_gradient else statistical_experiment_dict
    path = _write_trainrun(tmp_path, exp_dict, mock_pipeline_dict)

    fake_pipeline = FakeRestorePipeline(node_fits=(requires_fit,))
    monkeypatch.setattr(
        restore_mod,
        "_build_pipeline_from_config",
        lambda *a, **k: fake_pipeline,
    )
    monkeypatch.setattr(
        restore_mod,
        "_create_datamodule_from_config",
        lambda *a, **k: FakeRestoreDataModule(val_ds=val_ds, test_ds=test_ds),
    )
    RecordingTrainer.all_instances.clear()
    monkeypatch.setattr(restore_mod, "StatisticalTrainer", RecordingTrainer)
    monkeypatch.setattr(restore_mod, "GradientTrainer", RecordingTrainer)

    restore_mod.restore_trainrun(path, mode=mode, device="cpu")

    # stat_trainer is always the first RecordingTrainer created
    stat_trainer = RecordingTrainer.all_instances[0]
    grad_trainer = RecordingTrainer.all_instances[1] if use_gradient else None

    if mode == "train":
        assert fake_pipeline.save_to_file_calls, "save_to_file must fire in train mode"
        if requires_fit:
            assert "fit" in stat_trainer.calls
        if use_gradient:
            assert grad_trainer is not None
            assert "fit" in grad_trainer.calls
            if val_ds is not None:
                assert "validate" in grad_trainer.calls
            if test_ds is not None:
                assert "test" in grad_trainer.calls
        else:
            if val_ds is not None:
                assert "validate" in stat_trainer.calls
            if test_ds is not None:
                assert "test" in stat_trainer.calls
    elif mode == "validate":
        assert not fake_pipeline.save_to_file_calls, (
            "save_to_file must NOT fire in validate mode"
        )
        if requires_fit:
            assert "fit" in stat_trainer.calls
        active = grad_trainer if use_gradient else stat_trainer
        assert "validate" in active.calls
    elif mode == "test":
        assert not fake_pipeline.save_to_file_calls, (
            "save_to_file must NOT fire in test mode"
        )
        if requires_fit:
            assert "fit" in stat_trainer.calls
        active = grad_trainer if use_gradient else stat_trainer
        assert "test" in active.calls


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


def test_resolve_splits_path_rewrites_relative_to_base_dir(tmp_path: Path) -> None:
    """A relative splits_path resolves against the trainrun dir, not the CWD Hydra sets."""
    from cuvis_ai_schemas.training.data import DataConfig, DataSplitConfig

    cfg = DataConfig(
        data_module="npz_multi",
        splits=DataSplitConfig(splits_path="splits.json"),
        params={"index_csv": "x.csv"},
    )
    out = restore_mod._resolve_splits_path_in_config(cfg, base_dir=tmp_path)
    assert out.splits.splits_path == str(tmp_path / "splits.json")
    assert Path(out.splits.splits_path).is_absolute()


def test_resolve_splits_path_absolute_and_none_pass_through(tmp_path: Path) -> None:
    from cuvis_ai_schemas.training.data import DataConfig, DataSplitConfig

    abs_path = str(tmp_path / "splits.json")
    cfg_abs = DataConfig(
        data_module="npz_multi", splits=DataSplitConfig(splits_path=abs_path)
    )
    assert (
        restore_mod._resolve_splits_path_in_config(
            cfg_abs, base_dir=tmp_path
        ).splits.splits_path
        == abs_path
    )  # absolute untouched

    cfg_inline = DataConfig(data_module="fake", splits=DataSplitConfig(train=[]))
    assert (
        restore_mod._resolve_splits_path_in_config(cfg_inline, base_dir=tmp_path)
        is cfg_inline
    )  # no splits_path -> identity

    cfg_rel = DataConfig(
        data_module="npz_multi", splits=DataSplitConfig(splits_path="splits.json")
    )
    assert (
        restore_mod._resolve_splits_path_in_config(cfg_rel, base_dir=None) is cfg_rel
    )  # no base_dir -> identity


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


def test_create_datamodule_from_config_sets_up_fit(monkeypatch) -> None:
    """The trainrun datamodule builder resolves the split path, builds the module via the
    registry, and returns it already ``setup(stage="fit")``."""
    from unittest.mock import MagicMock

    fake_dm = MagicMock()
    monkeypatch.setattr(restore_mod, "_build_data_module", lambda *a, **k: fake_dm)
    monkeypatch.setattr(
        restore_mod, "_resolve_splits_path_in_config", lambda data, base: data
    )

    result = restore_mod._create_datamodule_from_config(MagicMock())

    fake_dm.setup.assert_called_once_with(stage="fit")
    assert result is fake_dm


def test_restore_trainrun_redirects_checkpoint_dirpath(
    monkeypatch, tmp_path: Path, mock_experiment_dict, mock_pipeline_dict
) -> None:
    """A gradient trainrun that configures a checkpoint callback has its ``dirpath``
    redirected under the run's ``output_dir`` before the GradientTrainer is built."""
    import copy

    exp = copy.deepcopy(mock_experiment_dict)
    exp["training"]["callbacks"] = {"checkpoint": {"monitor": "val_loss"}}
    path = _write_trainrun(tmp_path, exp, mock_pipeline_dict)

    captured: dict = {}

    class _CapturingGrad(RecordingTrainer):
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        restore_mod,
        "_build_pipeline_from_config",
        lambda *a, **k: FakeRestorePipeline(node_fits=(True,)),
    )
    monkeypatch.setattr(
        restore_mod,
        "_create_datamodule_from_config",
        lambda *a, **k: FakeRestoreDataModule(val_ds=None, test_ds=None),
    )
    RecordingTrainer.all_instances.clear()
    monkeypatch.setattr(restore_mod, "StatisticalTrainer", RecordingTrainer)
    monkeypatch.setattr(restore_mod, "GradientTrainer", _CapturingGrad)

    restore_mod.restore_trainrun(path, mode="train", device="cpu")

    dirpath = captured["training_config"].callbacks.checkpoint.dirpath
    assert dirpath.endswith("checkpoints")
