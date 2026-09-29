"""Training infrastructure for cuvis.ai PyTorch Lightning integration.

This package provides:
- Training configuration dataclasses with Hydra support (``config``)
- The gradient and statistical trainers (``trainers``) and the ``Predictor``
- Post-training decider calibration (``calibration``)
- The optimizer and scheduler registry (``optimizer_registry``) and the runtime
  callbacks (``callbacks``)
"""

from cuvis_ai_core.training.config import (
    CallbacksConfig,
    DataConfig,
    OptimizerConfig,
    PipelineConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_core.training.calibration import (
    CalibrationOutcome,
    calibrate_pipeline_deciders,
)
from cuvis_ai_core.training.predictor import Predictor
from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer

__all__ = [
    # Configuration
    "OptimizerConfig",
    "TrainingConfig",
    "SchedulerConfig",
    "CallbacksConfig",
    "DataConfig",
    "PipelineConfig",
    "TrainRunConfig",
    # Inference
    "Predictor",
    # Trainers
    "GradientTrainer",
    "StatisticalTrainer",
    # Post-training calibration
    "CalibrationOutcome",
    "calibrate_pipeline_deciders",
]
