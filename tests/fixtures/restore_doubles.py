"""Reusable test doubles for cuvis_ai_core.utils.restore tests.

Supersedes the inline ``_FakePipeline``, ``_SpecPipeline``, and
``_FakeInferenceDataModule`` classes that used to live in test_restore.py.
"""

from __future__ import annotations

import torch
from cuvis_ai_schemas.enums import ExecutionStage


class _FakeTorchLayer:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


class _FakeVideoNode:
    requires_initial_fit = False
    name = "video_node"

    def __init__(self) -> None:
        self.output_video_path = "out.mp4"
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _NodeList(list):
    """A list that is also callable.

    Lets FakeRestorePipeline.nodes serve both:
    - restore_pipeline: ``for node in pipeline.nodes`` (attribute iteration)
    - restore_trainrun: ``for node in pipeline.nodes()`` (called as method)
    """

    def __call__(self):
        return list(self)


class _FakeNode:
    """Minimal node stub with the two attributes restore_trainrun inspects."""

    def __init__(
        self, *, name: str = "node", requires_initial_fit: bool = False
    ) -> None:
        self.name = name
        self.requires_initial_fit = requires_initial_fit


class FakeRestorePipeline:
    """Unified fake pipeline for restore_pipeline (inference) and restore_trainrun tests.

    Pass ``node_fits`` to control which nodes report ``requires_initial_fit=True``;
    the first entry is used for the first node, etc. Defaults to a single node with
    ``requires_initial_fit=False``.

    ``self.nodes`` is a ``_NodeList``: it supports both attribute iteration (used by
    ``restore_pipeline`` when closing video nodes) and ``()`` call syntax (used by
    ``restore_trainrun`` at ``pipeline.nodes()``).
    """

    def __init__(
        self,
        *,
        name: str = "test_pipeline",
        node_fits: tuple[bool, ...] = (False,),
    ) -> None:
        self.name = name
        self.torch_layers = [_FakeTorchLayer()]
        self.video_node = _FakeVideoNode()
        fake_nodes = [
            _FakeNode(name=f"node{i}", requires_initial_fit=fit)
            for i, fit in enumerate(node_fits)
        ]
        self.nodes = _NodeList([self.video_node, *fake_nodes])
        self.profiling_enabled: list[bool] = []
        self.forward_calls: list[tuple[dict[str, torch.Tensor], object]] = []
        self.summary_calls: list[tuple[ExecutionStage, int]] = []
        self.save_to_file_calls: list[str] = []
        self.unfreeze_calls: list[list[str]] = []
        self.visualize_calls: list[dict] = []

    def set_profiling(self, *, enabled: bool) -> None:
        self.profiling_enabled.append(enabled)

    def forward(self, *, batch: dict[str, torch.Tensor], context: object) -> dict:
        self.forward_calls.append((batch, context))
        return {}

    def format_profiling_summary(
        self, *, stage: ExecutionStage, total_frames: int
    ) -> str:
        self.summary_calls.append((stage, total_frames))
        return "profiling-summary"

    def get_input_specs(self) -> dict[str, str]:
        return {"cube": "spec"}

    def get_output_specs(self) -> dict[str, str]:
        return {"node.output": "spec"}

    def save_to_file(self, path: str) -> None:
        self.save_to_file_calls.append(path)

    def unfreeze_nodes_by_name(self, names: list[str]) -> None:
        self.unfreeze_calls.append(list(names))

    def visualize(self, *, format: str, output_path) -> None:
        self.visualize_calls.append({"format": format, "output_path": output_path})


class FakeRestoreDataModule:
    """Unified fake Lightning DataModule for restore_pipeline and restore_trainrun tests.

    ``val_ds`` and ``test_ds`` are configurable sentinels checked by ``restore_trainrun``
    to gate val/test evaluation in train mode. Set to a truthy object to simulate a
    resolved dataset; leave as ``None`` to simulate "no dataset configured".

    ``instances`` is a class-level list so tests can assert on the number of
    DataModules created and inspect the config they were built from.
    """

    instances: list["FakeRestoreDataModule"] = []

    def __init__(
        self,
        data_config=None,
        *,
        val_ds: object = None,
        test_ds: object = None,
    ) -> None:
        self.data_config = data_config
        self.setup_stages: list[str | None] = []
        self.batches = [
            {"cube": torch.tensor([1.0], dtype=torch.float32)},
            {"cube": torch.tensor([2.0], dtype=torch.float32)},
        ]
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.__class__.instances.append(self)

    def setup(self, stage: str | None = None) -> None:
        self.setup_stages.append(stage)

    def predict_dataloader(self):
        return list(self.batches)


class RecordingTrainer:
    """Records fit/validate/test calls; stands in for both StatisticalTrainer and GradientTrainer.

    ``all_instances`` is a class-level list so tests can assert which trainer fired in
    which order (stat_trainer is always created first, grad_trainer second when present).
    Clear it at the start of each test with ``monkeypatch.setattr(RecordingTrainer,
    'all_instances', [])`` or ``RecordingTrainer.all_instances.clear()``.
    """

    all_instances: list["RecordingTrainer"] = []

    def __init__(self, **kwargs) -> None:
        self.calls: list[str] = []
        self.__class__.all_instances.append(self)

    def fit(self) -> None:
        self.calls.append("fit")

    def validate(self) -> None:
        self.calls.append("validate")

    def test(self) -> None:
        self.calls.append("test")
