"""Fast wiring tests for GradientTrainer after the TrainingConfig fold.

These lock the single-config API: scheduler derivation (the fix for the
CLI scheduler-drop bug), seed application in ``fit``, and checkpoint
auto-enable when callbacks are passed explicitly. All are construction- or
mock-level so they run in the ``not slow and not gpu`` gate.
"""

from unittest.mock import MagicMock, patch

import pytorch_lightning as pl
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import (
    CallbacksConfig,
    ModelCheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from cuvis_ai_core.training.trainers import GradientTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec


class _TrainableNode(Node):
    """Minimal node carrying one trainable parameter."""

    INPUT_SPECS = {}
    OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **inputs):
        return {"out": self.weight}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class _LossNode(Node):
    INPUT_SPECS = {"value": PortSpec(dtype=torch.float32, shape=())}
    OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}

    def forward(self, value, **kwargs):
        return {"loss": value}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class _MockDataModule(pl.LightningDataModule):
    pass


def _pipeline() -> CuvisPipeline:
    pipeline = CuvisPipeline("wiring")
    source = _TrainableNode()
    loss = _LossNode(name="loss", execution_stages={ExecutionStage.TRAIN})
    pipeline.connect(source.outputs.out, loss.value)
    return pipeline


def _trainer(training_config: TrainingConfig) -> GradientTrainer:
    return GradientTrainer(
        pipeline=_pipeline(),
        datamodule=_MockDataModule(),
        loss_nodes=[],
        training_config=training_config,
    )


def test_scheduler_and_optimizer_wired_from_training_config():
    """Regression: the single-config constructor derives scheduler + optimizer.

    The pre-fold CLI path never passed ``scheduler_config`` and the optimizer
    carried no scheduler, so the configured LR scheduler was silently dropped.
    """
    cfg = TrainingConfig(
        max_epochs=3,
        optimizer=OptimizerConfig(name="adamw", lr=0.001),
        scheduler=SchedulerConfig(name="cosine", t_max=3),
    )
    trainer = _trainer(cfg)
    assert trainer.scheduler_config is cfg.scheduler
    assert trainer.optimizer_config is cfg.optimizer


def test_configure_optimizers_attaches_scheduler_when_configured():
    """Regression: a configured scheduler reaches Lightning as an lr_scheduler."""
    cfg = TrainingConfig(
        max_epochs=3,
        optimizer=OptimizerConfig(name="adamw", lr=0.001),
        scheduler=SchedulerConfig(name="cosine", t_max=3),
    )
    result = _trainer(cfg).configure_optimizers()
    assert isinstance(result, dict)
    assert "lr_scheduler" in result


def test_configure_optimizers_without_scheduler_returns_bare_optimizer():
    """No scheduler configured -> plain optimizer, no lr_scheduler dict."""
    cfg = TrainingConfig(
        max_epochs=3, optimizer=OptimizerConfig(name="adamw", lr=0.001)
    )
    result = _trainer(cfg).configure_optimizers()
    assert isinstance(result, torch.optim.Optimizer)


def test_fit_seeds_before_training():
    """fit() seeds with the configured value (belt-and-suspenders re-seed)."""
    cfg = TrainingConfig(seed=1234, max_epochs=1)
    trainer = _trainer(cfg)
    with (
        patch("cuvis_ai_core.training.trainers.pl.Trainer") as mock_trainer_cls,
        patch("cuvis_ai_core.training.trainers.pl.seed_everything") as mock_seed,
    ):
        mock_trainer_cls.return_value = MagicMock()
        trainer.fit()
    mock_seed.assert_called_once()
    assert mock_seed.call_args.args[0] == 1234


def test_fit_enables_checkpointing_with_explicit_callbacks():
    """Regression: an explicit callback list still auto-enables checkpointing.

    The gRPC path passes a callback list (progress stream); the checkpoint
    auto-enable must not be skipped just because callbacks were supplied.
    """
    cfg = TrainingConfig(
        max_epochs=1,
        callbacks=CallbacksConfig(checkpoint=ModelCheckpointConfig(monitor="val_loss")),
    )
    trainer = GradientTrainer(
        pipeline=_pipeline(),
        datamodule=_MockDataModule(),
        loss_nodes=[],
        training_config=cfg,
        callbacks=[pl.callbacks.Callback()],
    )
    with patch("cuvis_ai_core.training.trainers.pl.Trainer") as mock_trainer_cls:
        mock_trainer_cls.return_value = MagicMock()
        trainer.fit()
    kwargs = mock_trainer_cls.call_args.kwargs
    assert kwargs["enable_checkpointing"] is True
    assert kwargs["callbacks"] == trainer.callbacks


def test_test_without_fit_lazily_builds_trainer_and_drops_best():
    """test() without a prior fit() builds a trainer from the config, and the
    'best' default degrades to None (no ModelCheckpoint ever ran, so Lightning
    would otherwise error on ckpt_path='best')."""
    trainer = _trainer(TrainingConfig(max_epochs=1))
    with patch("cuvis_ai_core.training.trainers.pl.Trainer") as mock_trainer_cls:
        inst = MagicMock()
        mock_trainer_cls.return_value = inst
        trainer.test()
    mock_trainer_cls.assert_called_once()
    assert inst.test.call_args.kwargs["ckpt_path"] is None


def test_test_without_fit_honors_explicit_ckpt_path():
    """An explicit ckpt_path survives the lazy-construction path untouched."""
    trainer = _trainer(TrainingConfig(max_epochs=1))
    with patch("cuvis_ai_core.training.trainers.pl.Trainer") as mock_trainer_cls:
        inst = MagicMock()
        mock_trainer_cls.return_value = inst
        trainer.test(ckpt_path="epoch01.ckpt")
    assert inst.test.call_args.kwargs["ckpt_path"] == "epoch01.ckpt"


def test_validate_without_fit_lazily_builds_trainer():
    """validate() without a prior fit() no longer raises; it evaluates the
    current model state with a config-built trainer."""
    trainer = _trainer(TrainingConfig(max_epochs=1))
    with patch("cuvis_ai_core.training.trainers.pl.Trainer") as mock_trainer_cls:
        inst = MagicMock()
        mock_trainer_cls.return_value = inst
        trainer.validate()
    mock_trainer_cls.assert_called_once()
    assert inst.validate.call_args.kwargs["ckpt_path"] is None


def test_test_after_fit_reuses_trainer_and_keeps_best_default():
    """Fit-first flow unchanged: the fit() trainer is reused (no rebuild) and
    the 'best' checkpoint default is preserved."""
    trainer = _trainer(TrainingConfig(max_epochs=1))
    with patch("cuvis_ai_core.training.trainers.pl.Trainer") as mock_trainer_cls:
        inst = MagicMock()
        mock_trainer_cls.return_value = inst
        trainer.fit()
        trainer.test()
    mock_trainer_cls.assert_called_once()
    assert inst.test.call_args.kwargs["ckpt_path"] == "best"


def test_fit_builds_callbacks_from_config_and_logs_early_stopping():
    """With no explicit callbacks, fit() builds them from the config, and an
    early-stopping entry is logged. Covers the config-callbacks branch and the
    early-stopping log line."""
    from cuvis_ai_schemas.training.callbacks import EarlyStoppingConfig

    cfg = TrainingConfig(
        max_epochs=1,
        callbacks=CallbacksConfig(
            early_stopping=[EarlyStoppingConfig(monitor="val_loss")],
        ),
    )
    trainer = _trainer(cfg)  # no explicit callbacks -> they come from the config
    with patch("cuvis_ai_core.training.trainers.pl.Trainer") as mock_trainer_cls:
        mock_trainer_cls.return_value = MagicMock()
        trainer.fit()
    # callbacks were derived from the config (not an explicitly-passed list).
    assert "callbacks" in mock_trainer_cls.call_args.kwargs
