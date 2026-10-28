"""A node whose hparams hold a path relative to its pipeline file learns the file's folder.

``pipeline_source_dir()`` is set by the loaders that build from a file for the duration
of the build (an exported span's ONNX file sits next to its deploy yaml); a build from a
dict leaves it unset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import (
    PipelineBuilder,
    pipeline_source,
    pipeline_source_dir,
)
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec


class _Recorder(Node):
    """Remembers what ``pipeline_source_dir()`` said while it was built."""

    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seen = pipeline_source_dir()

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {"out": torch.zeros(1)}


def _write(folder: Path) -> Path:
    config = {
        "metadata": {"name": "recorder"},
        "nodes": [
            {"name": "rec", "class_name": f"{__name__}._Recorder", "hparams": {}}
        ],
        "connections": [],
    }
    path = folder / "pipe.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _recorder(pipeline: CuvisPipeline) -> _Recorder:
    (node,) = pipeline.nodes
    assert isinstance(node, _Recorder)
    return node


def test_outside_a_load_there_is_no_source_dir():
    assert pipeline_source_dir() is None


def test_load_pipeline_tells_nodes_the_folder_of_the_yaml(tmp_path):
    pipeline = CuvisPipeline.load_pipeline(_write(tmp_path))

    assert _recorder(pipeline).seen == tmp_path.resolve()
    assert pipeline_source_dir() is None


def test_the_builder_tells_nodes_the_folder_when_given_a_path(tmp_path):
    pipeline = PipelineBuilder().build_from_config(_write(tmp_path))

    assert _recorder(pipeline).seen == tmp_path.resolve()


def test_the_builder_leaves_it_unset_for_a_dict(tmp_path):
    config = yaml.safe_load(_write(tmp_path).read_text(encoding="utf-8"))

    pipeline = PipelineBuilder().build_from_config(config)

    assert _recorder(pipeline).seen is None


def test_pipeline_source_takes_a_file_or_a_folder_and_resets_after_an_error(tmp_path):
    with pipeline_source(_write(tmp_path)):
        assert pipeline_source_dir() == tmp_path.resolve()
    with pipeline_source(tmp_path):
        assert pipeline_source_dir() == tmp_path.resolve()
    with pytest.raises(RuntimeError), pipeline_source(tmp_path):
        raise RuntimeError("boom")
    assert pipeline_source_dir() is None
