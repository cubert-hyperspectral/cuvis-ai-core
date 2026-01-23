"""Reusable test fixtures for cuvis-ai-core tests."""

from tests.fixtures.mock_nodes import (
    LentilsAnomalyDataNode,
    MinMaxNormalizer,
    MockStatisticalTrainableNode,
    SimpleLossNode,
    SoftChannelSelector,
)
from tests.fixtures.registry_test_nodes import MockMetricNode

__all__ = [
    "LentilsAnomalyDataNode",
    "MinMaxNormalizer",
    "MockMetricNode",
    "MockStatisticalTrainableNode",
    "SimpleLossNode",
    "SoftChannelSelector",
]
