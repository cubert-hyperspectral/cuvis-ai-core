"""Training infrastructure for cuvis.ai PyTorch Lightning integration.

This module provides:
- Training configuration dataclasses with Hydra support
- Port-based loss and metric nodes for training
- Internal Lightning module for training orchestration
"""

from cuvis_ai_core.training.config import (
    CallbacksConfig,
    DataConfig,
    OptimizerConfig,
    PipelineConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_core.training.predictor import Predictor
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
    # Context
    "Context",
    # Inference
    "Predictor",
    # Trainers
    "GradientTrainer",
    "StatisticalTrainer",
]
