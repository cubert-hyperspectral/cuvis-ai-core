"""Tests for the scheduler factory and its alias resolution."""

from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cuvis_ai_core.training.config import SchedulerConfig
from cuvis_ai_core.training.optimizer_registry import (
    SUPPORTED_SCHEDULERS,
    create_scheduler,
    get_scheduler_info,
    get_supported_schedulers,
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
