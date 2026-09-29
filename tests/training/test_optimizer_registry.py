"""Tests for the scheduler factory and its alias resolution."""

from __future__ import annotations

import math

import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, SequentialLR

from cuvis_ai_core.training.config import SchedulerConfig
from cuvis_ai_core.training.optimizer_registry import (
    SUPPORTED_SCHEDULERS,
    create_scheduler,
    get_scheduler_info,
    get_supported_schedulers,
    wrap_scheduler_for_lightning,
)


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)


def _build(name: str, **params):
    return create_scheduler(SchedulerConfig(name=name, **params), _optimizer(), 5)


def test_plateau_alias_builds_reduce_on_plateau():
    assert isinstance(_build("plateau"), ReduceLROnPlateau)


def test_plateau_alias_carries_the_same_parameters_as_the_full_name():
    params = {"factor": 0.5, "patience": 3, "min_lr": 1e-5, "threshold": 1e-3}
    alias = _build("plateau", **params)
    full = _build("reduce_on_plateau", **params)
    observed = (alias.factor, alias.patience, alias.min_lrs, alias.threshold)
    assert observed == (full.factor, full.patience, full.min_lrs, full.threshold)
    assert observed == (0.5, 3, [1e-5], 1e-3)


def test_scheduler_names_are_case_insensitive():
    assert isinstance(_build("Plateau"), ReduceLROnPlateau)


@pytest.mark.parametrize("name", ["none", ""])
def test_disabled_scheduler_names_return_none(name):
    assert _build(name) is None


@pytest.mark.parametrize("name", get_supported_schedulers())
def test_every_advertised_scheduler_name_builds(name):
    assert _build(name) is not None


def test_unknown_scheduler_raises_with_the_supported_list():
    with pytest.raises(ValueError, match="Unsupported scheduler: bogus") as exc_info:
        _build("bogus")
    assert "reduce_on_plateau" in str(exc_info.value)


def test_get_scheduler_info_resolves_the_alias_to_the_registry_entry():
    entry = SUPPORTED_SCHEDULERS["reduce_on_plateau"]
    assert get_scheduler_info("plateau") is entry
    assert get_scheduler_info("PLATEAU") is entry


def test_get_scheduler_info_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown scheduler: bogus"):
        get_scheduler_info("bogus")


# ---------------------------------------------------------------------------
# accepted-but-not-honoured knobs: gamma and warmup_epochs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["step", "exponential"])
def test_an_explicit_zero_gamma_is_honoured(name):
    """``gamma: 0.0`` is a value the schema allows; ``or`` turned it into the default."""
    assert _build(name, gamma=0.0).gamma == 0.0


def _lrs(scheduler, optimizer, steps):
    """The LR in force at each of ``steps`` epochs, stepping like Lightning does."""
    seen = []
    for _ in range(steps):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return seen


def test_warmup_ramps_linearly_then_hands_over_to_the_main_schedule():
    opt = _optimizer()
    sched = create_scheduler(SchedulerConfig(name="cosine", warmup_epochs=2), opt, 5)
    assert isinstance(sched, SequentialLR)
    lrs = _lrs(sched, opt, 5)
    assert lrs[0] == pytest.approx(0.1 / 3)
    assert lrs[1] == pytest.approx(0.1 * 2 / 3)
    assert lrs[2] == pytest.approx(0.1)
    ref_opt = _optimizer()
    ref = torch.optim.lr_scheduler.CosineAnnealingLR(ref_opt, T_max=3)
    # torch's chainable cosine drifts by ~1e-7 from the closed form
    assert lrs[2:] == pytest.approx(_lrs(ref, ref_opt, 3), abs=1e-6)


def test_default_t_max_ends_the_anneal_at_max_epochs():
    sched = create_scheduler(
        SchedulerConfig(name="cosine", warmup_epochs=2), _optimizer(), 5
    )
    assert sched._schedulers[1].T_max == 3


def test_an_explicit_t_max_survives_the_warmup():
    sched = create_scheduler(
        SchedulerConfig(name="cosine", warmup_epochs=2, t_max=10), _optimizer(), 5
    )
    assert sched._schedulers[1].T_max == 10


@pytest.mark.parametrize("name", ["step", "exponential"])
def test_epoch_stepped_schedulers_accept_a_warmup(name):
    assert isinstance(_build(name, warmup_epochs=1), SequentialLR)


def test_no_warmup_returns_the_bare_scheduler():
    assert not isinstance(_build("cosine", warmup_epochs=0), SequentialLR)


@pytest.mark.parametrize("name", ["plateau", "reduce_on_plateau"])
def test_a_metric_stepped_scheduler_rejects_a_warmup(name):
    with pytest.raises(ValueError, match="warmup_epochs"):
        _build(name, warmup_epochs=1)


def test_a_warmup_must_be_shorter_than_the_run():
    with pytest.raises(ValueError, match="warmup_epochs"):
        _build("cosine", warmup_epochs=5)  # _build trains for 5 epochs


def test_warmup_chain_state_survives_a_checkpoint_round_trip():
    opt = _optimizer()
    sched = create_scheduler(SchedulerConfig(name="cosine", warmup_epochs=2), opt, 5)
    _lrs(sched, opt, 2)  # stop right before the milestone
    # Lightning's order on resume: build the optimizer and scheduler, then
    # restore the optimizer state, then the scheduler state.
    resumed_opt = _optimizer()
    resumed = create_scheduler(
        SchedulerConfig(name="cosine", warmup_epochs=2), resumed_opt, 5
    )
    resumed_opt.load_state_dict(opt.state_dict())
    resumed.load_state_dict(sched.state_dict())
    assert _lrs(resumed, resumed_opt, 3) == pytest.approx(_lrs(sched, opt, 3), abs=1e-6)


def test_lightning_steps_the_warmup_chain_once_per_epoch():
    """What the trainer does with the chain: Lightning steps it after every epoch, so
    epoch 0 trains at the warmup value, epoch 1 at the base LR and epoch 2 on the anneal."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    class _Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(1, 1)
            self.seen = []

        def training_step(self, batch, _batch_idx):
            (x,) = batch
            return self.w(x).pow(2).mean()

        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=0.1)
            sched = create_scheduler(
                SchedulerConfig(name="cosine", warmup_epochs=1), opt, 3
            )
            return {
                "optimizer": opt,
                "lr_scheduler": wrap_scheduler_for_lightning(sched),
            }

        def on_train_epoch_start(self):
            self.seen.append(self.trainer.optimizers[0].param_groups[0]["lr"])

    module = _Module()
    loader = DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2)
    pl.Trainer(
        max_epochs=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(module, loader)
    assert module.seen == pytest.approx(
        [0.05, 0.1, 0.1 * (1 + math.cos(math.pi / 2)) / 2], abs=1e-6
    )
