"""Training infrastructure for cuvis.ai PyTorch Lightning integration.

This module provides:
- Training configuration dataclasses with Hydra support
- GraphDataModule base class for data loading
- Port-based loss and metric nodes for training
- Internal Lightning module for training orchestration
"""

from cuvis_ai_core.training.datamodule import CuvisDataModule
from cuvis_ai_schemas.pipeline import PipelineConfig
from cuvis_ai_schemas.training import (
    CallbacksConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_core.training.callback_factory import create_callbacks_from_config
from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer
from cuvis_ai_schemas.execution import Context

__all__ = [
    # Configuration
    "TrainerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "SchedulerConfig",
    "CallbacksConfig",
    "DataConfig",
    "PipelineConfig",
    "TrainRunConfig",
    # Data Module
    "CuvisDataModule",
    # Context
    "Context",
    # Helpers
    "create_callbacks_from_config",
    # External Trainers (Phase 4.7)
    "GradientTrainer",
    "StatisticalTrainer",
]
