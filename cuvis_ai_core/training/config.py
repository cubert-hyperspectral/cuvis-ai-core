"""Training configuration models — re-exported from cuvis-ai-schemas.

All config classes are defined in cuvis-ai-schemas. This module re-exports them
for backward compatibility so that ``from cuvis_ai_core.training.config import X``
continues to work.
"""

from __future__ import annotations

from cuvis_ai_schemas.pipeline.config import (
    ConnectionConfig,
    NodeConfig,
    PipelineConfig,
    PipelineMetadata,
)
from cuvis_ai_schemas.training import (
    CallbacksConfig,
    DataConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
    TrainRunConfig,
    create_callbacks_from_config,
)


__all__ = [
    "ConnectionConfig",
    "NodeConfig",
    "EarlyStoppingConfig",
    "ModelCheckpointConfig",
    "LearningRateMonitorConfig",
    "CallbacksConfig",
    "TrainerConfig",
    "SchedulerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "PipelineMetadata",
    "PipelineConfig",
    "DataConfig",
    "TrainRunConfig",
    "create_callbacks_from_config",
]
