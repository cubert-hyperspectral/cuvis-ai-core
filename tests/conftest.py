"""Root conftest.py for cuvis-ai-core tests.

This file provides shared fixtures and configuration for all tests.
Fixtures are imported from the fixtures/ directory and made available to all tests.
"""

from __future__ import annotations

import pytest

# Import all fixtures from fixtures/ modules
# This makes them available to all tests without explicit imports
pytest_plugins = [
    "tests.fixtures.basic_nodes",
    "tests.fixtures.basic_pipelines",
    "tests.fixtures.mock_models",
    "tests.fixtures.mock_nodes",
    "tests.fixtures.data_factory",
    "tests.fixtures.grpc",
    "tests.fixtures.config_factory",
    "tests.fixtures.sessions",
    "tests.fixtures.mock_sdk",
    "tests.fixtures.workflow_fixtures",
]


@pytest.fixture(scope="session")
def test_data_path():
    """Path to test data directory.
    
    Returns:
        Path to test data directory (data/)
    """
    from pathlib import Path
    return Path("data")


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary configuration directory for testing.
    
    Args:
        tmp_path: Pytest's temporary directory fixture
    
    Returns:
        Path to temporary config directory
    """
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def tmp_weights_dir(tmp_path):
    """Create a temporary weights directory for testing.
    
    Args:
        tmp_path: Pytest's temporary directory fixture
    
    Returns:
        Path to temporary weights directory
    """
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory for testing.
    
    Args:
        tmp_path: Pytest's temporary directory fixture
    
    Returns:
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory for testing.
    
    Args:
        tmp_path: Pytest's temporary directory fixture
    
    Returns:
        Path to temporary workspace directory
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def mock_pipeline_dir(tmp_path):
    """Create a temporary pipeline directory for testing.
    
    Args:
        tmp_path: Pytest's temporary directory fixture
    
    Returns:
        Path to temporary pipeline directory
    """
    pipeline_dir = tmp_path / "pipelines"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    return pipeline_dir


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state between tests.
    
    This fixture runs automatically before each test to ensure test isolation.
    """
    from cuvis_ai_core.utils.node_registry import NodeRegistry
    from tests.fixtures.registry_test_nodes import (
        MockMinMaxNormalizer,
        MockSoftChannelSelector,
        MockTrainablePCA,
        MockLossNode,
        MockMetricNode,
    )
    # Import after pytest plugin registration to avoid assertion rewrite warning
    from tests.fixtures import mock_nodes
    
    # Register mock nodes with clean names for YAML loading
    # This allows tests to use "MinMaxNormalizer" instead of "MockMinMaxNormalizer"
    NodeRegistry.clear()
    
    # Register with clean names (remove "Mock" prefix)
    NodeRegistry._builtin_registry["MinMaxNormalizer"] = MockMinMaxNormalizer
    NodeRegistry._builtin_registry["SoftChannelSelector"] = mock_nodes.SoftChannelSelector
    NodeRegistry._builtin_registry["TrainablePCA"] = MockTrainablePCA
    NodeRegistry._builtin_registry["LossNode"] = MockLossNode
    NodeRegistry._builtin_registry["MetricNode"] = MockMetricNode
    NodeRegistry._builtin_registry["LentilsAnomalyDataNode"] = mock_nodes.LentilsAnomalyDataNode
    NodeRegistry._builtin_registry["SimpleMSELoss"] = MockLossNode  # Alias for loss node
    NodeRegistry._builtin_registry["SimpleMetric"] = MockMetricNode  # Alias for metric node
    
    # Also register with Mock prefix for tests that use full names
    NodeRegistry._builtin_registry["MockMinMaxNormalizer"] = MockMinMaxNormalizer
    NodeRegistry._builtin_registry["MockSoftChannelSelector"] = MockSoftChannelSelector
    NodeRegistry._builtin_registry["MockTrainablePCA"] = MockTrainablePCA
    NodeRegistry._builtin_registry["MockLossNode"] = MockLossNode
    NodeRegistry._builtin_registry["MockMetricNode"] = MockMetricNode
    
    yield
    
    # Cleanup after test
    NodeRegistry.clear()
