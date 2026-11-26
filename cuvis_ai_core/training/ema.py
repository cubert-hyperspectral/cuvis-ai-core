"""Exponential moving average of the trained weights (``TrainingConfig.ema``).

``EmaCallback`` keeps an average of every trainable parameter and floating-point
buffer of the trained module. It updates after every ``update_interval`` optimizer
steps (a batch that only accumulates gradients leaves the weights, and so the average,
alone) and, with ``update_on_epoch_end``, once more at the end of every training epoch.
The ``n``-th update computes ``average = average * d + live * (1 - d)`` with
``d = decay * (1 - exp(-n / tau))`` (``d = decay`` for ``tau == 0``); the first update
copies the live weights. This is the update of ``torch.optim.swa_utils.AveragedModel``
with ``use_buffers=True`` and the decay rule of RF-DETR's and timm's model EMA.

With ``evaluate == "ema"`` validation and test run on the average (swapped in for the
pass, the live weights restored after it), so the logged metrics and the best-checkpoint
choice are the average's; at the end of training the average replaces the live weights,
and ``GradientTrainer.load_best_checkpoint`` applies the average saved with the best
checkpoint. With ``evaluate == "live"`` everything runs on the live weights and the
average is only kept in the checkpoints (``EmaCallback.state_dict``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

#: Key of the callback's state in a Lightning checkpoint (``checkpoint["callbacks"]``).
EMA_STATE_KEY = "EmaCallback"


def ema_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The averaged tensors of ``module`` by name: trainable parameters, float buffers."""
    tensors = {name: p for name, p in module.named_parameters() if p.requires_grad}
    tensors.update(
        {name: b for name, b in module.named_buffers() if b.is_floating_point()}
    )
    return tensors


@torch.no_grad()
def copy_into(module: torch.nn.Module, values: dict[str, torch.Tensor]) -> list[str]:
    """Copy ``values`` into the matching tensors of ``module``; returns the missing names."""
    targets = ema_tensors(module)
    missing = [name for name in values if name not in targets]
    for name, value in values.items():
        target = targets.get(name)
        if target is not None:
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    return missing


class EmaCallback(Callback):
    """Keeps, evaluates and applies the moving average of the trained weights.

    Parameters
    ----------
    config : EmaConfig
        ``decay``, ``tau``, ``update_interval``, ``update_on_epoch_end`` and
        ``evaluate`` (see :class:`cuvis_ai_schemas.training.EmaConfig`).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.decay = float(config.decay)
        self.tau = float(config.tau)
        self.update_interval = int(config.update_interval)
        self.update_on_epoch_end = bool(config.update_on_epoch_end)
        self.evaluate = str(config.evaluate)
        self.shadow: dict[str, torch.Tensor] = {}
        self.updates = 0
        self._last_step = 0
        self._backup: dict[str, torch.Tensor] | None = None
        self._pending: dict[str, Any] | None = None

    @property
    def state_key(self) -> str:
        """Fixed checkpoint key, so the best-checkpoint restore can find the average."""
        return EMA_STATE_KEY

    # -- lifecycle ---------------------------------------------------------------
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the average from the live weights, or from a resumed checkpoint."""
        live = ema_tensors(pl_module)
        if self._pending is not None:
            self.shadow = {
                name: value.to(live[name].device).clone()
                for name, value in self._pending["shadow"].items()
                if name in live
            }
            self.updates = int(self._pending["updates"])
            self._last_step = int(self._pending["last_step"])
            self._pending = None
            return
        self.shadow = {name: t.detach().clone() for name, t in live.items()}
        self.updates = 0
        self._last_step = int(trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update once per optimizer step (every ``update_interval`` steps)."""
        step = int(trainer.global_step)
        if step <= self._last_step:
            return
        self._last_step = step
        if step % self.update_interval == 0:
            self.update(pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """The optional extra update at the end of every training epoch."""
        if self.update_on_epoch_end:
            self.update(pl_module)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Validate on the average."""
        self._swap_in(pl_module)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Restore the live weights before the checkpoint is written."""
        self._swap_out(pl_module)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Test on the average."""
        self._swap_in(pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore the live weights after a test pass."""
        self._swap_out(pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Leave the module on the average when it is the evaluated weights."""
        self._swap_out(pl_module)
        if self.evaluate == "ema" and self.updates > 0:
            copy_into(pl_module, self.shadow)

    # -- averaging ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, pl_module: torch.nn.Module) -> None:
        """Apply one update of the average towards the live weights."""
        live = ema_tensors(pl_module)
        if self.updates == 0:
            for name, value in self.shadow.items():
                value.copy_(live[name].detach())
        else:
            n = self.updates + 1
            d = (
                self.decay * (1.0 - math.exp(-n / self.tau))
                if self.tau > 0
                else self.decay
            )
            for name, value in self.shadow.items():
                value.copy_(value * d + live[name].detach() * (1.0 - d))
        self.updates += 1

    def _swap_in(self, pl_module: torch.nn.Module) -> None:
        """Put the average on the module for an evaluation pass, keeping the live weights."""
        if self.evaluate != "ema" or self.updates == 0 or self._backup is not None:
            return
        live = ema_tensors(pl_module)
        self._backup = {name: live[name].detach().clone() for name in self.shadow}
        copy_into(pl_module, self.shadow)

    def _swap_out(self, pl_module: torch.nn.Module) -> None:
        """Put the live weights kept by :meth:`_swap_in` back on the module."""
        if self._backup is None:
            return
        copy_into(pl_module, self._backup)
        self._backup = None

    # -- checkpoint state -----------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """The average, the update count and the evaluated weights, for the checkpoint."""
        return {
            "shadow": {
                name: value.detach().cpu() for name, value in self.shadow.items()
            },
            "updates": int(self.updates),
            "last_step": int(self._last_step),
            "evaluate": self.evaluate,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Keep a resumed average until the fit starts (the module is not ready yet)."""
        if state_dict:
            self._pending = state_dict
