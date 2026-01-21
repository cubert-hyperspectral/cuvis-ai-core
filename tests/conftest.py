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
    "tests.fixtures.data_factory",
    "tests.fixtures.grpc",
    "tests.fixtures.config_factory",
    "tests.fixtures.sessions",
]


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


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state between tests.
    
    This fixture runs automatically before each test to ensure test isolation.
    """
    # Reset any registries or global state here
    # For example, clear the node registry if needed
    yield
    # Cleanup after test if needed
