"""Tests for the Predictor inference orchestrator."""

from __future__ import annotations


from types import SimpleNamespace

import pytorch_lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import cuvis_ai_core.pipeline.pipeline as pipeline_mod
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

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs) -> None:
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

    assert predictor.pipeline.device.type == "cpu"

    moved = predictor._move_batch_to_device(
        {
            "value": torch.tensor([[1.0]], dtype=torch.float32),
            "meta": "keep-me",
        }
    )

    assert moved["value"].device.type == "cpu"
    assert moved["meta"] == "keep-me"


# ---------------------------------------------------------------------------
# Data-load profiling through the pipeline's profiled batch iterator
# ---------------------------------------------------------------------------


class CountingDataset(DictDataset):
    """Counts fetches; with batch_size=1 and no workers, one fetch is one batch."""

    def __init__(self, values: torch.Tensor) -> None:
        super().__init__(values)
        self.fetches = 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self.fetches += 1
        return super().__getitem__(idx)


class CountingDataModule(PredictDataModule):
    def setup(self, stage: str | None = None) -> None:
        if stage == "predict" or stage is None:
            self.predict_ds = CountingDataset(self.values)


class FailingSinkNode(PredictSinkNode):
    """Fails on its second forward."""

    def forward(
        self, doubled: torch.Tensor, context: Context | None = None, **_
    ) -> dict:
        if self.forward_calls >= 1:
            raise RuntimeError("sink failed")
        return super().forward(doubled, context)


def _data_rows(pipeline: CuvisPipeline) -> dict[str, object]:
    return {s.node_name: s for s in pipeline.get_data_profiling_summary()}


def test_predictor_profiles_the_data_load_when_enabled() -> None:
    values = torch.tensor([[1.0], [2.0], [3.0]])
    plain_pipeline, _, _ = _build_pipeline()
    plain = Predictor(
        pipeline=plain_pipeline, datamodule=PredictDataModule(values=values)
    ).predict(collect_outputs=True)

    pipeline, _, sink = _build_pipeline()
    pipeline.set_profiling(enabled=True)
    profiled = Predictor(
        pipeline=pipeline, datamodule=PredictDataModule(values=values)
    ).predict(collect_outputs=True)

    assert profiled is not None and plain is not None
    assert [out[("source", "doubled")].tolist() for out in profiled] == [
        out[("source", "doubled")].tolist() for out in plain
    ]
    rows = _data_rows(pipeline)
    assert set(rows) == {"data_load", "to_device", "batch_loop"}
    assert {name: r.count for name, r in rows.items()} == {
        "data_load": 2,
        "to_device": 2,
        "batch_loop": 2,
    }
    assert len(pipeline._first_batch_ms["inference"]) == 1
    assert sink.forward_calls == 3
    assert sink.close_calls == 1


@pytest.mark.parametrize("profiling", [False, True])
def test_predictor_max_batches_bounds_fetches_moves_and_forwards(
    monkeypatch: pytest.MonkeyPatch, profiling: bool
) -> None:
    pipeline, source, sink = _build_pipeline()
    pipeline.set_profiling(enabled=profiling)
    datamodule = CountingDataModule(
        values=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    )
    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    moves: list[int] = []
    real_move = predictor._move_batch_to_device
    monkeypatch.setattr(
        predictor,
        "_move_batch_to_device",
        lambda batch: (moves.append(1), real_move(batch))[1],
    )

    predictor.predict(max_batches=2, collect_outputs=False)

    assert datamodule.predict_ds is not None
    assert datamodule.predict_ds.fetches == 2  # not the one-past-the-limit fetch of old
    assert len(moves) == 2
    assert sink.forward_calls == 2
    assert [ctx.batch_idx for ctx in source.contexts] == [0, 1]


def test_predictor_repeat_predict_keeps_one_first_batch_per_pass() -> None:
    pipeline, _, _ = _build_pipeline()
    pipeline.set_profiling(enabled=True)
    values = torch.tensor([[1.0], [2.0], [3.0]])

    Predictor(pipeline=pipeline, datamodule=PredictDataModule(values=values)).predict()
    Predictor(pipeline=pipeline, datamodule=PredictDataModule(values=values)).predict()

    assert len(pipeline._first_batch_ms["inference"]) == 2
    assert _data_rows(pipeline)["data_load"].count == 4  # (3 - 1) per pass


def test_predictor_batch_loop_includes_output_handling_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"now": 0}

    def perf_counter_ns() -> int:
        state["now"] += 1_000_000
        return state["now"]

    monkeypatch.setattr(
        pipeline_mod, "time", SimpleNamespace(perf_counter_ns=perf_counter_ns)
    )

    class _RenderingPbar:
        """A progress bar whose per-item update costs one clock tick."""

        def __init__(self, iterable) -> None:
            self._iterable = iterable

        def __iter__(self):
            for item in self._iterable:
                perf_counter_ns()
                yield item

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        predictor_mod, "tqdm", lambda iterable, **kw: _RenderingPbar(iterable)
    )
    real_select = Predictor._select_outputs
    monkeypatch.setattr(
        Predictor,
        "_select_outputs",
        staticmethod(
            lambda outputs, ports: (perf_counter_ns(), real_select(outputs, ports))[1]
        ),
    )

    pipeline, _, _ = _build_pipeline()
    pipeline.set_profiling(enabled=True)
    Predictor(
        pipeline=pipeline,
        datamodule=PredictDataModule(values=torch.tensor([[1.0], [2.0], [3.0]])),
    ).predict(collect_outputs=True, collect_ports={"doubled"})

    rows = _data_rows(pipeline)
    # data_load sees only the loader's next(): one tick.
    assert rows["data_load"].mean_ms == 1.0
    # batch_loop spans fetch start .. next request: fetch (1) + move (1) + progress render (1)
    # + two node timers (4) + output selection (1) + loop close (1) = 9 ticks.
    assert rows["batch_loop"].mean_ms == 9.0


def test_predictor_forward_error_leaves_no_partial_loop_sample() -> None:
    pipeline = CuvisPipeline("predict_pipeline")
    source = PredictSourceNode(name="source")
    sink = FailingSinkNode(name="sink")
    pipeline.connect(source.outputs.doubled, sink.inputs.doubled)
    pipeline.set_profiling(enabled=True)

    with pytest.raises(RuntimeError, match="sink failed"):
        Predictor(
            pipeline=pipeline,
            datamodule=PredictDataModule(values=torch.tensor([[1.0], [2.0], [3.0]])),
        ).predict()

    rows = _data_rows(pipeline)
    assert rows["data_load"].count == 1  # the second fetch happened
    assert (
        "batch_loop" not in rows or rows["batch_loop"].count == 0
    )  # its loop never closed
    assert sink.close_calls == 1  # the finally ran


def test_predictor_single_batch_shows_only_the_first_batch_line() -> None:
    pipeline, _, _ = _build_pipeline()
    pipeline.set_profiling(enabled=True)

    Predictor(
        pipeline=pipeline, datamodule=PredictDataModule(values=torch.tensor([[1.0]]))
    ).predict()

    table = pipeline.format_profiling_summary()
    assert "First batch data load (inference, excluded from the rows):" in table
    assert "Time per batch" not in table
    assert not any(line.startswith("data_load") for line in table.split("\n"))
