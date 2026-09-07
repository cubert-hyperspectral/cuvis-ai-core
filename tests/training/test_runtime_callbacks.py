"""Runtime callbacks: CUDA memory reporting and the opt-in cache release.

Nothing here needs a GPU: ``torch.cuda`` is monkeypatched so the callbacks'
own logic (phase lines, peak resets, the memory-fraction cap and its failure
paths, the accumulation guard) is what is under test.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from cuvis_ai_core.training import callbacks as callbacks_mod
from cuvis_ai_core.training.callbacks import (
    CUDA_MEMORY_FRACTION_ENV,
    CudaCacheReleaseCallback,
    CudaMemoryLogCallback,
    build_runtime_callbacks,
)
from cuvis_ai_core.training.config import TrainingConfig


@pytest.fixture
def log_lines():
    """Collect every loguru message emitted during the test."""
    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: records.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="DEBUG",
    )
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _cuda_lines(records) -> list[str]:
    return [message for _, message in records if message.startswith("cuda-mem ")]


class _FakeCuda:
    """Records what the callbacks asked of ``torch.cuda``."""

    def __init__(self, *, available: bool = True, total: int = 12 * 1024**3) -> None:
        self.available = available
        self.total = total
        self.fraction_calls: list[float] = []
        self.empty_cache_calls = 0
        self.reset_peak_calls = 0
        self.fraction_raises: Exception | None = None

    # torch.cuda surface the callbacks touch
    def is_available(self) -> bool:
        return self.available

    def current_device(self) -> int:
        return 0

    def get_device_name(self, device=None) -> str:
        return "Fake RTX"

    def get_device_properties(self, device=None):
        return SimpleNamespace(total_memory=self.total)

    def memory_allocated(self) -> int:
        return 3 * 1024**2

    def max_memory_allocated(self) -> int:
        return 7 * 1024**2

    def memory_reserved(self) -> int:
        return 10 * 1024**2

    def reset_peak_memory_stats(self) -> None:
        self.reset_peak_calls += 1

    def memory_summary(self) -> str:
        return "fake summary"

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def set_per_process_memory_fraction(self, fraction: float) -> None:
        if self.fraction_raises is not None:
            raise self.fraction_raises
        self.fraction_calls.append(fraction)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Replace the ``torch.cuda`` functions the callbacks call."""
    fake = _FakeCuda()
    for name in (
        "is_available",
        "current_device",
        "get_device_name",
        "get_device_properties",
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "reset_peak_memory_stats",
        "memory_summary",
        "empty_cache",
        "set_per_process_memory_fraction",
    ):
        monkeypatch.setattr(torch.cuda, name, getattr(fake, name))
    monkeypatch.delenv(CUDA_MEMORY_FRACTION_ENV, raising=False)
    return fake


class _FakeModule:
    """LightningModule stand-in that records ``zero_grad``."""

    def __init__(self) -> None:
        self.zero_grad_calls: list[bool] = []

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_calls.append(set_to_none)


def _trainer(accumulate_grad_batches: int = 1):
    """Trainer stand-in: only ``accumulate_grad_batches`` is read."""
    return SimpleNamespace(accumulate_grad_batches=accumulate_grad_batches)


# ---------------------------------------------------------------------------
# CudaMemoryLogCallback
# ---------------------------------------------------------------------------


def test_no_cuda_logs_nothing(monkeypatch, log_lines):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callback = CudaMemoryLogCallback()
    module = _FakeModule()

    callback.on_fit_start(_trainer(), module)
    callback.on_train_epoch_end(_trainer(), module)
    callback.on_validation_epoch_start(_trainer(), module)
    callback.on_validation_batch_end(_trainer(), module, None, None, 0)
    callback.on_validation_epoch_end(_trainer(), module)

    assert _cuda_lines(log_lines) == []


def test_each_phase_logs_one_line_and_resets_the_peak(fake_cuda, log_lines):
    callback = CudaMemoryLogCallback()
    module = _FakeModule()

    callback.on_fit_start(_trainer(), module)
    callback.on_train_epoch_end(_trainer(), module)
    callback.on_validation_epoch_start(_trainer(), module)
    callback.on_validation_batch_end(_trainer(), module, None, None, 0)
    callback.on_validation_epoch_end(_trainer(), module)

    phases = [
        line.split("phase=")[1].split()[0]
        for line in _cuda_lines(log_lines)
        if "phase=" in line
    ]
    assert phases == [
        "fit-start",
        "train-epoch-end",
        "val-start",
        "val-first-batch",
        "val-end",
    ]
    assert fake_cuda.reset_peak_calls == len(phases)


def test_the_phase_line_carries_the_four_numbers(fake_cuda, log_lines):
    CudaMemoryLogCallback().on_train_epoch_end(_trainer(), _FakeModule())

    line = _cuda_lines(log_lines)[0]
    assert "allocated=3 MiB" in line
    assert "peak=7 MiB" in line
    assert "reserved=10 MiB" in line
    # gap = reserved - allocated: what empty_cache could give back.
    assert "gap=7 MiB" in line


def test_only_the_first_validation_batch_of_an_epoch_logs(fake_cuda, log_lines):
    callback = CudaMemoryLogCallback()
    module = _FakeModule()

    callback.on_validation_epoch_start(_trainer(), module)
    for batch_idx in range(4):
        callback.on_validation_batch_end(_trainer(), module, None, None, batch_idx)

    assert sum("val-first-batch" in line for line in _cuda_lines(log_lines)) == 1

    # The next validation epoch reports its own first batch.
    callback.on_validation_epoch_start(_trainer(), module)
    callback.on_validation_batch_end(_trainer(), module, None, None, 0)
    assert sum("val-first-batch" in line for line in _cuda_lines(log_lines)) == 2


def test_memory_summary_is_logged_once_at_debug(fake_cuda, log_lines):
    callback = CudaMemoryLogCallback()
    module = _FakeModule()

    callback.on_validation_epoch_end(_trainer(), module)
    callback.on_validation_epoch_end(_trainer(), module)

    summaries = [
        (level, message)
        for level, message in log_lines
        if message.startswith("cuda-mem summary")
    ]
    assert len(summaries) == 1
    assert summaries[0][0] == "DEBUG"


def test_fit_start_names_the_device_and_its_size(fake_cuda, log_lines):
    CudaMemoryLogCallback().on_fit_start(_trainer(), _FakeModule())

    device_line = next(line for line in _cuda_lines(log_lines) if "device=" in line)
    assert "Fake RTX" in device_line
    assert "total=12288 MiB" in device_line


# -- the memory fraction ----------------------------------------------------


def test_env_fraction_is_applied_and_reported(fake_cuda, monkeypatch, log_lines):
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "0.66")

    CudaMemoryLogCallback().on_fit_start(_trainer(), _FakeModule())

    assert fake_cuda.fraction_calls == [0.66]
    budget_line = next(line for line in _cuda_lines(log_lines) if "fraction=" in line)
    assert "budget=8110 MiB" in budget_line
    # The caveat travels with the number so nobody reads it as a hard limit.
    assert "soft cap" in budget_line


def test_the_constructor_argument_wins_over_the_env(fake_cuda, monkeypatch):
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "0.5")
    CudaMemoryLogCallback(memory_fraction=0.25).on_fit_start(_trainer(), _FakeModule())
    assert fake_cuda.fraction_calls == [0.25]


def test_no_fraction_leaves_the_device_alone(fake_cuda):
    CudaMemoryLogCallback().on_fit_start(_trainer(), _FakeModule())
    assert fake_cuda.fraction_calls == []


@pytest.mark.parametrize("raw", ["not-a-number", "0", "-0.5", "1.5"])
def test_an_invalid_fraction_warns_and_continues(
    fake_cuda, monkeypatch, log_lines, raw
):
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, raw)

    CudaMemoryLogCallback().on_fit_start(_trainer(), _FakeModule())

    assert fake_cuda.fraction_calls == []
    assert any(level == "WARNING" for level, _ in log_lines)
    # The run continues: the phase line still went out.
    assert any("phase=fit-start" in line for line in _cuda_lines(log_lines))


def test_a_raising_set_fraction_warns_and_continues(fake_cuda, monkeypatch, log_lines):
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "0.5")
    fake_cuda.fraction_raises = RuntimeError("unsupported on this device")

    CudaMemoryLogCallback().on_fit_start(_trainer(), _FakeModule())

    warnings = [message for level, message in log_lines if level == "WARNING"]
    assert any("unsupported on this device" in message for message in warnings)
    assert any("phase=fit-start" in line for line in _cuda_lines(log_lines))


def test_a_blank_env_value_is_not_a_fraction(fake_cuda, monkeypatch, log_lines):
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "   ")

    CudaMemoryLogCallback().on_fit_start(_trainer(), _FakeModule())

    assert fake_cuda.fraction_calls == []
    assert [level for level, _ in log_lines if level == "WARNING"] == []


# ---------------------------------------------------------------------------
# CudaCacheReleaseCallback
# ---------------------------------------------------------------------------


def test_release_drops_gradients_and_empties_the_cache(fake_cuda, monkeypatch):
    collected = []
    monkeypatch.setattr(gc, "collect", lambda *a: collected.append(1))
    callback = CudaCacheReleaseCallback()
    module = _FakeModule()

    callback.on_validation_epoch_start(_trainer(), module)

    assert module.zero_grad_calls == [True]
    assert fake_cuda.empty_cache_calls == 1
    assert len(collected) == 1


def test_release_skips_zero_grad_while_gradients_accumulate(fake_cuda):
    callback = CudaCacheReleaseCallback()
    module = _FakeModule()

    callback.on_validation_epoch_start(_trainer(accumulate_grad_batches=4), module)

    # Mid-accumulation those gradients are live state; only the cache goes.
    assert module.zero_grad_calls == []
    assert fake_cuda.empty_cache_calls == 1


def test_release_on_epoch_end_only_empties_the_cache(fake_cuda):
    callback = CudaCacheReleaseCallback()
    module = _FakeModule()

    callback.on_validation_epoch_end(_trainer(), module)

    assert module.zero_grad_calls == []
    assert fake_cuda.empty_cache_calls == 1


def test_release_is_a_no_op_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    empties = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empties.append(1))
    callback = CudaCacheReleaseCallback()
    module = _FakeModule()

    callback.on_validation_epoch_start(_trainer(), module)
    callback.on_validation_epoch_end(_trainer(), module)

    assert module.zero_grad_calls == []
    assert empties == []


# ---------------------------------------------------------------------------
# build_runtime_callbacks
# ---------------------------------------------------------------------------


def test_the_log_callback_is_always_built():
    built = build_runtime_callbacks(TrainingConfig())
    assert [type(c) for c in built] == [CudaMemoryLogCallback]


def test_the_release_callback_is_opt_in():
    built = build_runtime_callbacks(
        TrainingConfig(release_cuda_cache_on_validation=True)
    )
    assert [type(c) for c in built] == [
        CudaMemoryLogCallback,
        CudaCacheReleaseCallback,
    ]


def test_no_config_still_yields_the_log_callback():
    assert [type(c) for c in build_runtime_callbacks(None)] == [CudaMemoryLogCallback]


def test_the_env_var_name_is_the_documented_one():
    assert callbacks_mod.CUDA_MEMORY_FRACTION_ENV == "CUVIS_CUDA_MEMORY_FRACTION"
