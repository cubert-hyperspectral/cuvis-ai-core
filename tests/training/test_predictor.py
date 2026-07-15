"""Tests for the Predictor inference orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

import pytorch_lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import cuvis_ai_core.training.predictor as predictor_mod
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


class GatedSinkNode(Node):
    """A sink gated to VAL/TEST only (like a metric node), to exercise stage routing."""

    INPUT_SPECS = {
        "doubled": PortSpec(
            dtype=torch.float32, shape=(-1, -1), description="Input stream"
        ),
    }
    OUTPUT_SPECS = {}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("execution_stages", {ExecutionStage.VAL, ExecutionStage.TEST})
        super().__init__(**kwargs)
        self.forward_calls = 0

    def forward(
        self, doubled: torch.Tensor, context: Context | None = None, **_
    ) -> dict:
        del doubled, context
        self.forward_calls += 1
        return {}


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


class _LenRaises:
    def __iter__(self):
        yield {"value": torch.tensor([[1.0]])}

    def __len__(self) -> int:
        raise TypeError("unknown length")


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


def _build_gated_pipeline() -> tuple[CuvisPipeline, PredictSourceNode, GatedSinkNode]:
    pipeline = CuvisPipeline("predict_gated_pipeline")
    source = PredictSourceNode(name="source")
    gated = GatedSinkNode(name="gated")
    pipeline.connect(source.outputs.doubled, gated.inputs.doubled)
    return pipeline, source, gated


def test_predictor_stage_selects_which_nodes_fire() -> None:
    # Default stage INFERENCE: a VAL/TEST-gated node (e.g. a metric node) does not run.
    pipeline, source, gated = _build_gated_pipeline()
    datamodule = PredictDataModule(values=torch.tensor([[1.0], [2.0]]), batch_size=1)
    Predictor(pipeline=pipeline, datamodule=datamodule).predict(collect_outputs=False)
    assert gated.forward_calls == 0
    assert all(ctx.stage == ExecutionStage.INFERENCE for ctx in source.contexts)

    # stage=TEST: the gated node fires for every batch, in its native stage.
    pipeline, source, gated = _build_gated_pipeline()
    datamodule = PredictDataModule(values=torch.tensor([[1.0], [2.0]]), batch_size=1)
    Predictor(pipeline=pipeline, datamodule=datamodule).predict(
        stage=ExecutionStage.TEST, collect_outputs=False
    )
    assert gated.forward_calls == 2
    assert all(ctx.stage == ExecutionStage.TEST for ctx in source.contexts)


def test_predictor_collect_ports_filters_and_moves_to_cpu() -> None:
    # collect_ports keeps only the named ports and returns them detached on CPU.
    pipeline, _, _ = _build_pipeline()
    datamodule = PredictDataModule(values=torch.tensor([[1.0], [2.0]]), batch_size=1)
    kept = Predictor(pipeline=pipeline, datamodule=datamodule).predict(
        collect_outputs=True, collect_ports={"doubled"}
    )
    assert kept is not None and len(kept) == 2
    for out in kept:
        assert set(out) == {("source", "doubled")}
        value = out[("source", "doubled")]
        assert value.device.type == "cpu"
        assert not value.requires_grad

    # A port name that no node produces yields empty per-batch dicts (nothing retained).
    pipeline, _, _ = _build_pipeline()
    datamodule = PredictDataModule(values=torch.tensor([[1.0], [2.0]]), batch_size=1)
    dropped = Predictor(pipeline=pipeline, datamodule=datamodule).predict(
        collect_outputs=True, collect_ports={"not_a_port"}
    )
    assert dropped is not None and all(out == {} for out in dropped)


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


@pytest.mark.parametrize(
    ("is_tty", "expected_disable"),
    [
        (False, True),
        (True, False),
    ],
)
def test_predictor_tqdm_disable_follows_tty_state(
    monkeypatch: pytest.MonkeyPatch, is_tty: bool, expected_disable: bool
) -> None:
    pipeline, _, sink = _build_pipeline()
    datamodule = PredictDataModule(values=torch.tensor([[1.0], [2.0]]), batch_size=1)
    captured_disable: list[bool] = []

    class _FakePbar:
        def __init__(self, iterable) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def close(self) -> None:
            return None

    def _fake_tqdm(iterable, *args, **kwargs):
        del args
        captured_disable.append(bool(kwargs.get("disable")))
        return _FakePbar(iterable)

    class _FakeStderr:
        def isatty(self) -> bool:
            return is_tty

    monkeypatch.setattr(predictor_mod, "tqdm", _fake_tqdm)
    monkeypatch.setattr(predictor_mod.sys, "stderr", _FakeStderr())

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(collect_outputs=False)

    assert captured_disable == [expected_disable]
    assert sink.forward_calls == 2


def test_predictor_helper_methods_cover_batch_estimation_and_iteration() -> None:
    loader = DataLoader(
        DictDataset(torch.tensor([[1.0], [2.0]])),
        batch_size=1,
        shuffle=False,
    )
    mapping = {"first": loader, "skip": None}
    iterable = [loader, None]

    assert Predictor._estimate_total_batches(loader, None) == 2
    assert Predictor._estimate_total_batches(mapping, max_batches=1) == 1
    assert Predictor._estimate_total_batches(iterable, max_batches=None) == 2
    assert Predictor._estimate_total_batches({"bad": _LenRaises()}, None) is None

    mapping_batches = [
        batch["value"].item() for batch in Predictor._iter_batches(mapping)
    ]
    iterable_batches = [
        batch["value"].item() for batch in Predictor._iter_batches(iterable)
    ]

    assert mapping_batches == [1.0, 2.0]
    assert iterable_batches == [1.0, 2.0]

    with pytest.raises(TypeError, match="predict_dataloader\\(\\) must return"):
        list(Predictor._iter_batches(123))


def test_predictor_progress_bar_disables_for_missing_or_broken_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoIsatty:
        pass

    class _BrokenIsatty:
        def isatty(self) -> bool:
            raise RuntimeError("boom")

    monkeypatch.setattr(predictor_mod.sys, "stderr", _NoIsatty())
    assert Predictor._should_disable_progress_bar() is True

    monkeypatch.setattr(predictor_mod.sys, "stderr", _BrokenIsatty())
    assert Predictor._should_disable_progress_bar() is True


def test_predictor_progress_bar_stays_enabled_inside_ipython(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a Jupyter / IPython kernel `get_ipython()` returns a non-None shell;
    the progress bar must remain enabled even when stderr.isatty() is False."""
    import sys as _sys
    from types import ModuleType

    fake_ipython = ModuleType("IPython")
    fake_ipython.get_ipython = lambda: object()
    monkeypatch.setitem(_sys.modules, "IPython", fake_ipython)

    class _NotATty:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(predictor_mod.sys, "stderr", _NotATty())

    assert Predictor._should_disable_progress_bar() is False


def test_predictor_progress_bar_falls_through_when_ipython_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If IPython is importable but `get_ipython()` returns None (e.g. plain
    Python that has IPython installed), the IPython branch falls through to
    the stderr/TTY check."""
    import sys as _sys
    from types import ModuleType

    fake_ipython = ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None
    monkeypatch.setitem(_sys.modules, "IPython", fake_ipython)

    class _NotATty:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(predictor_mod.sys, "stderr", _NotATty())

    assert Predictor._should_disable_progress_bar() is True


def test_predictor_progress_bar_handles_missing_ipython(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless environments without IPython installed: the import raises
    ImportError, the helper falls through to the TTY check (returns True
    when stderr is non-TTY)."""
    import builtins
    import sys as _sys

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "IPython":
            raise ImportError("no IPython on this Python")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(_sys.modules, "IPython", None)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    class _NotATty:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(predictor_mod.sys, "stderr", _NotATty())

    assert Predictor._should_disable_progress_bar() is True


def test_predictor_moves_batches_using_pipeline_device_and_preserves_non_tensors() -> (
    None
):
    pipeline, source, _ = _build_pipeline()
    source.register_buffer("device_probe", torch.tensor([1.0], dtype=torch.float32))
    predictor = Predictor(
        pipeline=pipeline,
        datamodule=PredictDataModule(values=torch.tensor([[1.0]])),
    )

    assert predictor._get_pipeline_device().type == "cpu"

    moved = predictor._move_batch_to_device(
        {
            "value": torch.tensor([[1.0]], dtype=torch.float32),
            "meta": "keep-me",
        }
    )

    assert moved["value"].device.type == "cpu"
    assert moved["meta"] == "keep-me"


def test_predictor_get_pipeline_device_uses_parameter_device() -> None:
    class _LayerWithParameterDevice:
        def parameters(self):
            return [SimpleNamespace(device=torch.device("cuda:0"))]

        def buffers(self):
            return []

    class _PipelineStub:
        name = "predict_stub"
        torch_layers = [_LayerWithParameterDevice()]

        @staticmethod
        def nodes():
            return []

    predictor = Predictor(
        pipeline=_PipelineStub(),
        datamodule=PredictDataModule(values=torch.tensor([[1.0]])),
    )

    assert predictor._get_pipeline_device() == torch.device("cuda:0")
