"""Root conftest.py for cuvis-ai-core tests.

This file provides shared fixtures and configuration for all tests.
Fixtures are imported from the fixtures/ directory and made available to all tests.
"""

from __future__ import annotations

import warnings
import pytest

# Filter PyTorch Lightning internal deprecation warnings
# TreeSpec deprecation in pytorch_lightning.utilities._pytree
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="pytorch_lightning.utilities._pytree",
)

# Import all fixtures from fixtures/ modules
# This makes them available to all tests without explicit imports
# Note: mock_nodes is imported at runtime in reset_global_state to avoid assertion rewrite issues
pytest_plugins = [
    "tests.fixtures.basic_nodes",
    "tests.fixtures.basic_pipelines",
    "tests.fixtures.mock_models",
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


@pytest.fixture
def create_plugin_pyproject():
    """Fixture providing a helper to create PEP 621 compliant pyproject.toml for test plugins.

    Returns:
        Callable that takes a plugin directory Path and creates a minimal pyproject.toml

    Example:
        def test_my_plugin(tmp_path, create_plugin_pyproject):
            plugin_dir = tmp_path / "my_plugin"
            plugin_dir.mkdir()
            create_plugin_pyproject(plugin_dir)
    """
    from pathlib import Path

    def _create_pyproject_toml(plugin_dir: Path) -> None:
        """Create PEP 621 compliant pyproject.toml for test plugin.

        Args:
            plugin_dir: Directory where the plugin is located
        """
        (plugin_dir / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{plugin_dir.name}"\n'
            'version = "0.1.0"\n'
            'description = "Test plugin"\n'
            'requires-python = ">=3.11"\n'
            "dependencies = []\n"
        )

    return _create_pyproject_toml


@pytest.fixture
def mock_statistical_trainable_node():
    """Fixture providing MockStatisticalTrainableNode class for tests.

    Returns:
        MockStatisticalTrainableNode class (not an instance)
    """
    from tests.fixtures.mock_nodes import MockStatisticalTrainableNode

    return MockStatisticalTrainableNode


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state between tests.

    This fixture runs automatically before each test to ensure test isolation.
    All mock nodes are imported from tests.fixtures.mock_nodes at runtime
    to avoid pytest assertion rewrite warnings.
    """
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    # Import at runtime to avoid assertion rewrite issues
    from tests.fixtures import mock_nodes

    NodeRegistry.clear()

    # Register with clean names for YAML loading
    NodeRegistry._builtin_registry["MinMaxNormalizer"] = mock_nodes.MinMaxNormalizer
    NodeRegistry._builtin_registry["SoftChannelSelector"] = (
        mock_nodes.SoftChannelSelector
    )
    NodeRegistry._builtin_registry["TrainablePCA"] = mock_nodes.MockTrainablePCA
    NodeRegistry._builtin_registry["LossNode"] = mock_nodes.MockLossNode
    NodeRegistry._builtin_registry["MetricNode"] = mock_nodes.MockMetricNode
    NodeRegistry._builtin_registry["LentilsAnomalyDataNode"] = (
        mock_nodes.LentilsAnomalyDataNode
    )
    NodeRegistry._builtin_registry["SimpleLossNode"] = mock_nodes.SimpleLossNode
    NodeRegistry._builtin_registry["SimpleMSELoss"] = mock_nodes.MockLossNode
    NodeRegistry._builtin_registry["SimpleMetric"] = mock_nodes.MockMetricNode

    # Also register with Mock prefix for tests that use full names
    NodeRegistry._builtin_registry["MockMinMaxNormalizer"] = (
        mock_nodes.MockMinMaxNormalizer
    )
    NodeRegistry._builtin_registry["MockSoftChannelSelector"] = (
        mock_nodes.MockSoftChannelSelector
    )
    NodeRegistry._builtin_registry["MockTrainablePCA"] = mock_nodes.MockTrainablePCA
    NodeRegistry._builtin_registry["MockLossNode"] = mock_nodes.MockLossNode
    NodeRegistry._builtin_registry["MockMetricNode"] = mock_nodes.MockMetricNode

    yield

    # Cleanup after test
    NodeRegistry.clear()
