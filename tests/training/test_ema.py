"""Weight averaging of the trained weights (``TrainingConfig.ema``, ``EmaCallback``).

The update is checked against ``torch.optim.swa_utils.AveragedModel`` with the same
average function, the timing against a stand-in trainer (one update per optimizer step,
an optional one per epoch end), the evaluation swap, the final application and the
checkpoint state. A real CPU fit checks that the post-fit restore puts the average saved
with the best checkpoint on the pipeline.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.callbacks import build_runtime_callbacks
from cuvis_ai_core.training.ema import EMA_STATE_KEY, EmaCallback, ema_tensors
from cuvis_ai_core.training.post_fit import PostFitCallback
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

pytestmark = pytest.mark.unit

schemas_training = pytest.importorskip("cuvis_ai_schemas.training")
needs_ema_schema = pytest.mark.skipif(
    not hasattr(schemas_training, "EmaConfig"),
    reason="needs a cuvis-ai-schemas with TrainingConfig.ema",
)


def _config(**overrides):
    values = {
        "decay": 0.993,
        "tau": 100.0,
        "update_interval": 1,
        "update_on_epoch_end": False,
        "evaluate": "ema",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))


@torch.no_grad()
def _train_step(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.add_(torch.randn_like(p))
    model[1].running_mean.add_(0.5)


def _start(callback: EmaCallback, model: torch.nn.Module) -> SimpleNamespace:
    trainer = SimpleNamespace(global_step=0)
    callback.on_fit_start(trainer, model)
    return trainer


def _step(callback, trainer, model, steps: int = 1) -> None:
    """One optimizer step: weights change, global_step moves, the batch ends."""
    _train_step(model)
    trainer.global_step += steps
    callback.on_train_batch_end(trainer, model, None, None, 0)


def _floats(module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in ema_tensors(module).items()}


def test_update_matches_averaged_model_with_the_same_rule() -> None:
    decay, tau = 0.993, 100.0

    def avg_fn(averaged, live, num_averaged):
        n = int(num_averaged) + 1
        d = decay * (1 - math.exp(-n / tau))
        return averaged * d + live * (1.0 - d)

    model = _model()
    reference = AveragedModel(model, use_buffers=True, avg_fn=avg_fn)
    callback = EmaCallback(_config(decay=decay, tau=tau))
    trainer = _start(callback, model)
    for _ in range(5):
        _step(callback, trainer, model)
        reference.update_parameters(model)
    expected = {
        k: v for k, v in reference.module.state_dict().items() if v.is_floating_point()
    }
    assert set(callback.shadow) == set(expected)
    for name, value in callback.shadow.items():
        assert torch.allclose(value, expected[name], atol=1e-6), name
    assert callback.updates == 5


def test_first_update_copies_and_constant_decay_without_tau() -> None:
    model = _model()
    callback = EmaCallback(_config(decay=0.5, tau=0.0))
    trainer = _start(callback, model)
    _step(callback, trainer, model)
    assert all(torch.equal(callback.shadow[k], v) for k, v in _floats(model).items())
    before = {k: v.clone() for k, v in callback.shadow.items()}
    _step(callback, trainer, model)
    live = _floats(model)
    for k, v in callback.shadow.items():
        assert torch.allclose(v, 0.5 * before[k] + 0.5 * live[k])


def test_one_update_per_optimizer_step_not_per_batch() -> None:
    model = _model()
    callback = EmaCallback(_config())
    trainer = _start(callback, model)
    _step(callback, trainer, model)
    for _ in range(3):  # gradient accumulation: batches without an optimizer step
        callback.on_train_batch_end(trainer, model, None, None, 0)
    assert callback.updates == 1
    _step(callback, trainer, model)
    assert callback.updates == 2


def test_update_interval_and_epoch_end() -> None:
    model = _model()
    callback = EmaCallback(_config(update_interval=2, update_on_epoch_end=True))
    trainer = _start(callback, model)
    _step(callback, trainer, model)  # step 1: skipped
    assert callback.updates == 0
    _step(callback, trainer, model)  # step 2: update
    assert callback.updates == 1
    callback.on_train_epoch_end(trainer, model)
    assert callback.updates == 2
    plain = EmaCallback(_config())
    plain_trainer = _start(plain, model)
    plain.on_train_epoch_end(plain_trainer, model)
    assert plain.updates == 0


def test_validation_runs_on_the_average_and_restores_the_live_weights() -> None:
    model = _model()
    callback = EmaCallback(_config(decay=0.5, tau=0.0))
    trainer = _start(callback, model)
    _step(callback, trainer, model)
    _step(callback, trainer, model)
    live = _floats(model)
    callback.on_validation_epoch_start(trainer, model)
    assert all(
        torch.equal(_floats(model)[k], callback.shadow[k]) for k in callback.shadow
    )
    callback.on_validation_epoch_end(trainer, model)
    assert all(torch.equal(_floats(model)[k], live[k]) for k in live)


def test_no_swap_before_the_first_update_or_with_live_evaluation() -> None:
    model = _model()
    fresh = EmaCallback(_config())
    trainer = _start(fresh, model)
    live = _floats(model)
    fresh.on_validation_epoch_start(
        trainer, model
    )  # sanity check: nothing averaged yet
    assert fresh._backup is None
    on_live = EmaCallback(_config(evaluate="live"))
    trainer = _start(on_live, model)
    _step(on_live, trainer, model)
    live = _floats(model)
    on_live.on_validation_epoch_start(trainer, model)
    assert all(torch.equal(_floats(model)[k], live[k]) for k in live)


@pytest.mark.parametrize(("evaluate", "applied"), [("ema", True), ("live", False)])
def test_train_end_leaves_the_evaluated_weights(evaluate: str, applied: bool) -> None:
    model = _model()
    callback = EmaCallback(_config(decay=0.5, tau=0.0, evaluate=evaluate))
    trainer = _start(callback, model)
    _step(callback, trainer, model)
    _step(callback, trainer, model)
    live = _floats(model)
    callback.on_train_end(trainer, model)
    target = callback.shadow if applied else live
    assert all(torch.equal(_floats(model)[k], target[k]) for k in target)


def test_checkpoint_state_round_trip_resumes_the_average() -> None:
    model = _model()
    callback = EmaCallback(_config(decay=0.5, tau=0.0))
    trainer = _start(callback, model)
    _step(callback, trainer, model)
    _step(callback, trainer, model)
    state = callback.state_dict()
    assert (
        state["evaluate"] == "ema" and state["updates"] == 2 and state["last_step"] == 2
    )
    assert all(v.device.type == "cpu" for v in state["shadow"].values())
    resumed = EmaCallback(_config(decay=0.5, tau=0.0))
    resumed.load_state_dict(state)
    resumed.on_fit_start(SimpleNamespace(global_step=2), copy.deepcopy(model))
    assert resumed.updates == 2 and resumed._last_step == 2
    assert all(
        torch.equal(resumed.shadow[k], callback.shadow[k]) for k in callback.shadow
    )
    assert callback.state_key == EMA_STATE_KEY


@needs_ema_schema
def test_runtime_callbacks_add_the_average_before_the_post_fit_step() -> None:
    from cuvis_ai_schemas.training import EmaConfig, TrainingConfig

    with_ema = build_runtime_callbacks(TrainingConfig(ema=EmaConfig()))
    kinds = [type(c) for c in with_ema]
    assert EmaCallback in kinds
    assert kinds.index(EmaCallback) < kinds.index(PostFitCallback)
    assert EmaCallback not in [
        type(c) for c in build_runtime_callbacks(TrainingConfig())
    ]


class _Mean(Node):
    """Learns the mean of the data: loss = mean((data - w)^2)."""

    INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, data, **_):
        return {"out": ((data - self.weight) ** 2).mean()}


class _Loss(Node):
    INPUT_SPECS = {"value": PortSpec(dtype=torch.float32, shape=())}
    OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}
    EXECUTION_STAGES = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}

    def forward(self, value, **_):
        return {"loss": value}


class _Data(pl.LightningDataModule):
    def _loader(self, seed: int):
        generator = torch.Generator().manual_seed(seed)
        items = [torch.randn(4, generator=generator) + 3.0 for _ in range(8)]
        return DataLoader(
            items, batch_size=2, collate_fn=lambda b: {"data": torch.stack(b)}
        )

    def train_dataloader(self):
        return self._loader(0)

    def val_dataloader(self):
        return self._loader(1)

    def test_dataloader(self):
        return self._loader(2)


@needs_ema_schema
def test_fit_saves_the_average_of_the_best_checkpoint(tmp_path) -> None:
    from cuvis_ai_core.training.config import (
        CallbacksConfig,
        ModelCheckpointConfig,
        OptimizerConfig,
        TrainingConfig,
    )
    from cuvis_ai_core.training.trainers import GradientTrainer
    from cuvis_ai_schemas.training import EmaConfig

    pipeline = CuvisPipeline("ema_fit")
    node, loss = _Mean(name="mean"), _Loss(name="loss")
    pipeline.connect(node.outputs.out, loss.value)
    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=_Data(),
        loss_nodes=[loss],
        training_config=TrainingConfig(
            seed=0,
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            default_root_dir=str(tmp_path),
            optimizer=OptimizerConfig(name="sgd", lr=0.3),
            ema=EmaConfig(decay=0.5, update_on_epoch_end=True),
            callbacks=CallbacksConfig(
                checkpoint=ModelCheckpointConfig(
                    dirpath=str(tmp_path / "checkpoints"),
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                )
            ),
        ),
    )
    trainer.fit()
    callback = next(c for c in trainer.trainer.callbacks if isinstance(c, EmaCallback))
    best = torch.load(
        trainer.trainer.checkpoint_callback.best_model_path, weights_only=True
    )
    saved = best["callbacks"][EMA_STATE_KEY]
    assert saved["evaluate"] == "ema" and saved["updates"] > 0
    name = next(k for k in saved["shadow"] if k.endswith("weight"))
    # the pipeline holds the best checkpoint's average, not its live weights
    assert torch.allclose(node.weight.detach().cpu(), saved["shadow"][name])
    live = best["state_dict"][name]
    assert not torch.allclose(saved["shadow"][name], live)
    assert callback.updates > 0
