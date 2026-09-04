"""Runtime Lightning callbacks: CUDA memory reporting and cache release.

Both callbacks exist because of one failure: in-app training dying at the
first validation pass on small-VRAM machines with no evidence of why. The
log callback makes the run say what it is holding at the points where the
footprint changes (end of a train epoch, around validation), so a death
leaves numbers behind. The release callback is the mitigation, and it is
opt-in per trainrun (``release_cuda_cache_on_validation``) because every
``empty_cache`` synchronizes the device.

Neither callback does anything without CUDA, so a CPU run pays nothing.
"""

from __future__ import annotations

import gc
import os
from typing import Any

import torch
from loguru import logger
from pytorch_lightning import Callback, LightningModule, Trainer

from cuvis_ai_core.training.config import TrainingConfig

# Soft cap on the process's share of the device, read when the log callback
# starts a fit. See CudaMemoryLogCallback.on_fit_start for what "soft" means.
CUDA_MEMORY_FRACTION_ENV = "CUVIS_CUDA_MEMORY_FRACTION"

_MIB = 1024 * 1024


class CudaMemoryLogCallback(Callback):
    """Log the CUDA footprint at the points where a run's memory changes.

    One ``cuda-mem phase=... allocated ... peak ... reserved ... gap ...``
    line at the end of each train epoch, at the start of validation, after
    validation's first batch, and at the end of validation. ``peak`` is
    reset after every line, so each one reports the high-water mark since
    the previous phase rather than since the run started. ``gap`` is
    ``reserved - allocated``: memory the caching allocator holds but is
    not using, which is what ``empty_cache`` can give back.

    Parameters
    ----------
    memory_fraction : float, optional
        Fraction of the device this process may allocate, applied at fit
        start. ``None`` (the default) reads
        ``$CUVIS_CUDA_MEMORY_FRACTION`` instead, so a run can be squeezed
        into a smaller budget without touching the config.
    """

    def __init__(self, memory_fraction: float | None = None) -> None:
        """Store the requested memory fraction (``None`` reads the env var)."""
        super().__init__()
        self._memory_fraction = memory_fraction
        self._summary_logged = False
        self._logged_first_val_batch = False

    # -- lifecycle ------------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Name the device, its size, and apply the memory fraction."""
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / _MIB
        logger.info(
            f"cuda-mem device={torch.cuda.get_device_name(device)} "
            f"index={device} total={total:.0f} MiB"
        )
        self._apply_memory_fraction(total)
        self._log_phase("fit-start")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Report the footprint the finished train epoch leaves behind."""
        self._log_phase("train-epoch-end")

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Report what validation starts from (train activations included)."""
        self._logged_first_val_batch = False
        self._log_phase("val-start")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Report the first validation batch: the peak that kills small cards."""
        if self._logged_first_val_batch:
            return
        self._logged_first_val_batch = True
        self._log_phase("val-first-batch")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Report the footprint validation leaves, plus one full summary."""
        self._log_phase("val-end")
        if not self._summary_logged and torch.cuda.is_available():
            self._summary_logged = True
            try:
                logger.debug(f"cuda-mem summary\n{torch.cuda.memory_summary()}")
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug(f"cuda-mem summary unavailable: {exc}")

    # -- internals ------------------------------------------------------

    def _apply_memory_fraction(self, total_mib: float) -> None:
        """Cap this process's share of the device, if a fraction was given.

        The cap is soft in two ways worth knowing before reading a number
        off it: it does not cover the CUDA context (a few hundred MiB the
        driver takes before any tensor exists), and transient allocations
        outside the caching allocator can exceed it. It bounds what torch
        will hand out, not what the process occupies.
        """
        fraction = self._resolve_memory_fraction()
        if fraction is None:
            return
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception as exc:
            logger.warning(
                f"Could not apply a CUDA memory fraction of {fraction}: {exc}. "
                "Continuing with the full device."
            )
            return
        logger.info(
            f"cuda-mem fraction={fraction} budget={total_mib * fraction:.0f} MiB "
            "(soft cap: excludes the CUDA context, transients can exceed it)"
        )

    def _resolve_memory_fraction(self) -> float | None:
        """Return the fraction to apply: the constructor's, the env's, or none."""
        if self._memory_fraction is not None:
            return self._validated_fraction(self._memory_fraction, "constructor")
        raw = os.environ.get(CUDA_MEMORY_FRACTION_ENV)
        if raw is None or not raw.strip():
            return None
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                f"{CUDA_MEMORY_FRACTION_ENV}={raw!r} is not a number; ignoring it."
            )
            return None
        return self._validated_fraction(value, CUDA_MEMORY_FRACTION_ENV)

    @staticmethod
    def _validated_fraction(value: float, source: str) -> float | None:
        """Accept a fraction in ``(0, 1]``; warn and ignore anything else."""
        if not 0 < value <= 1:
            logger.warning(
                f"{source} memory fraction {value} is outside (0, 1]; ignoring it."
            )
            return None
        return value

    @staticmethod
    def _log_phase(phase: str) -> None:
        """Log one memory line for ``phase`` and restart the peak counter."""
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / _MIB
        peak = torch.cuda.max_memory_allocated() / _MIB
        reserved = torch.cuda.memory_reserved() / _MIB
        logger.info(
            f"cuda-mem phase={phase} allocated={allocated:.0f} MiB "
            f"peak={peak:.0f} MiB reserved={reserved:.0f} MiB "
            f"gap={reserved - allocated:.0f} MiB"
        )
        torch.cuda.reset_peak_memory_stats()


class CudaCacheReleaseCallback(Callback):
    """Give the caching allocator's unused blocks back around validation.

    A train epoch leaves the allocator holding blocks shaped for training.
    Validation then asks for differently shaped ones, and on a small card
    the two sets do not fit at once even though neither is large by itself.
    Releasing the cache on the way in (and on the way out, so training does
    not inherit validation's blocks) trades a device synchronize per
    validation for that headroom, which is why it is opt-in.

    Gradients are dropped on the way in as well, but only when gradient
    accumulation is off: mid-accumulation those gradients are live state.
    """

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Drop gradients (when safe) and release the allocator's cache."""
        if not torch.cuda.is_available():
            return
        if getattr(trainer, "accumulate_grad_batches", 1) == 1:
            pl_module.zero_grad(set_to_none=True)
        self._release()

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Release validation's blocks so the next train epoch starts clean."""
        if not torch.cuda.is_available():
            return
        self._release()

    @staticmethod
    def _release() -> None:
        """Collect Python garbage, then hand the freed blocks back to the driver."""
        gc.collect()
        torch.cuda.empty_cache()


def build_runtime_callbacks(training_config: TrainingConfig | None) -> list[Callback]:
    """Build the callbacks every gradient run gets, regardless of its config.

    The memory log is always on: it costs a handful of log lines and is the
    only evidence a run leaves when it dies of memory. The cache release is
    added only when the trainrun asks for it.

    Parameters
    ----------
    training_config : TrainingConfig, optional
        The run's trainer config. ``None`` yields the log callback alone.

    Returns
    -------
    list[Callback]
        Callbacks to prepend to the run's callback list.
    """
    callbacks: list[Callback] = [CudaMemoryLogCallback()]
    if training_config is not None and training_config.release_cuda_cache_on_validation:
        callbacks.append(CudaCacheReleaseCallback())
    return callbacks
