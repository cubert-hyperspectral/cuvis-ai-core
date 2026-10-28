"""The export-pipeline CLI: a deploy folder from a pipeline yaml, its weights and
examples (files or a data module)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec
from tests.pipeline.test_export import _anomaly_pipeline

pytest.importorskip("onnxruntime")


@pytest.fixture
def saved_anomaly(tmp_path: Path) -> Path:
    pipeline, _ = _anomaly_pipeline()
    path = tmp_path / "canonical" / "anomaly.yaml"
    pipeline.save_to_file(path)
    return path


def _write_examples(folder: Path, count: int = 2) -> list[Path]:
    torch.manual_seed(1)
    wavelengths = torch.linspace(450.0, 950.0, 4).numpy()
    paths = []
    for index in range(count):
        path = folder / f"example{index}.npz"
        np.savez(
            path,
            **{
                "scale.data": torch.rand(1, 6, 6, 4).numpy(),
                "select.wavelengths": wavelengths,
            },
        )
        paths.append(path)
    return paths


_CUT = [
    "--input", "scale.data",
    "--input", "select.wavelengths",
    "--constant", "select.wavelengths",
    "--output", "detect.scores",
    "--output", "detect.anomaly_score",
]  # fmt: skip


def test_cli_exports_a_deploy_folder_from_example_files(
    saved_anomaly: Path, tmp_path: Path
):
    from scripts.export_pipeline import main

    examples = _write_examples(tmp_path)
    out_dir = tmp_path / "deploy"

    result = CliRunner().invoke(
        main,
        ["--pipeline-path", str(saved_anomaly), "--out-dir", str(out_dir), *_CUT]
        + [arg for path in examples for arg in ("--examples", str(path))],
    )

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "anomaly.onnx",
        "anomaly.onnx.json",
        "anomaly_deploy.pt",
        "anomaly_deploy.yaml",
    ]
    assert "anomaly_deploy.yaml" in result.output


class _Affine(Node):
    INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1, 2))}
    OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1, 2))}

    def __init__(self, offset: float = 1.0, **kwargs) -> None:
        super().__init__(offset=offset, **kwargs)
        self.register_buffer("offset_buf", torch.tensor(float(offset)))

    def forward(self, x, **_):
        return {"y": x + self.offset_buf}


def test_cli_takes_the_examples_from_a_data_module(tmp_path: Path):
    """The pipeline reads ``x`` from the batch; the fake data module provides it."""
    from scripts.export_pipeline import main

    first, second = _Affine(name="first"), _Affine(name="second", offset=2.0)
    pipeline = CuvisPipeline("affine")
    pipeline.connect(first.outputs.y, second.inputs.x)
    # With a plugins catalog in play, every node class must come from a declared
    # plugin, so the test node gets a manifest of its own next to the data module's.
    pipeline._plugins = ["fake_nodes"]
    pipeline_path = tmp_path / "affine.yaml"
    pipeline.save_to_file(pipeline_path)
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "fake_dataloader.yaml").write_text(
        dedent(
            """
            name: fake_dataloader
            path: "."
            package_name: fake_pkg
            capabilities:
              - class_name: tests.fixtures.fake_data_modules.FakeDataModule
                kind: data_module
                data_module_name: fake
            """
        ),
        encoding="utf-8",
    )
    (plugins_dir / "fake_nodes.yaml").write_text(
        dedent(
            f"""
            name: fake_nodes
            path: "."
            package_name: fake_pkg
            capabilities:
              - class_name: {_Affine.__module__}.{_Affine.__qualname__}
            """
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "deploy"

    result = CliRunner().invoke(
        main,
        [
            "--pipeline-path",
            str(pipeline_path),
            "--out-dir",
            str(out_dir),
            "--input",
            "first.x",
            "--output",
            "second.y",
            "--data-module",
            "fake",
            "--plugins-dir",
            str(plugins_dir),
            "--num-examples",
            "2",
        ],  # fmt: skip
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "affine.onnx").is_file()
    sidecar = (out_dir / "affine.onnx.json").read_text(encoding="utf-8")
    assert '"examples": 2' in sidecar
    assert '"shape": [\n          1,\n          2\n        ]' in sidecar


def test_cli_needs_example_files_or_a_data_module(saved_anomaly: Path, tmp_path: Path):
    from scripts.export_pipeline import main

    result = CliRunner().invoke(
        main, ["--pipeline-path", str(saved_anomaly), "--out-dir", str(tmp_path), *_CUT]
    )

    assert result.exit_code == 2
    assert "--examples" in result.output and "--data-module" in result.output


def test_cli_reports_a_bad_cut_as_an_error_message(saved_anomaly: Path, tmp_path: Path):
    from scripts.export_pipeline import main

    examples = _write_examples(tmp_path, count=1)

    result = CliRunner().invoke(
        main,
        [
            "--pipeline-path",
            str(saved_anomaly),
            "--out-dir",
            str(tmp_path / "deploy"),
            "--input",
            "scale.data",
            "--output",
            "detect.heatmap",
            "--examples",
            str(examples[0]),
        ],  # fmt: skip
    )

    assert result.exit_code == 1
    assert "detect.heatmap" in result.output
    assert "Traceback" not in result.output
