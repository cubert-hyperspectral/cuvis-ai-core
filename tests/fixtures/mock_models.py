"""Mock model file fixtures for testing without real model dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture
def mock_pt_file(tmp_path):
    """Create a dummy .pt file for testing weight loading.

    Args:
        tmp_path: Pytest's temporary directory fixture

    Returns:
        Path to the created .pt file
    """
    model_path = tmp_path / "model.pt"

    # Create a simple state dict
    state_dict = {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.zeros(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.zeros(5),
    }

    torch.save(state_dict, model_path)
    return model_path


@pytest.fixture
def mock_checkpoint_file(tmp_path):
    """Create a full checkpoint file with model state and metadata.

    Args:
        tmp_path: Pytest's temporary directory fixture

    Returns:
        Path to the created checkpoint file
    """
    checkpoint_path = tmp_path / "checkpoint.pt"

    checkpoint = {
        "state_dict": {
            "model.weight": torch.randn(20, 20),
            "model.bias": torch.zeros(20),
        },
        "epoch": 10,
        "optimizer_state": {},
        "metadata": {
            "version": "1.0.0",
            "created_at": "2026-01-20",
        },
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_pt_file_factory(tmp_path):
    """Factory fixture for creating customizable .pt files.

    Returns:
        Callable that creates .pt files with specified parameters
    """

    def _create_pt_file(
        filename: str = "model.pt",
        layer_sizes: list[tuple[int, int]] | None = None,
        include_metadata: bool = False,
    ) -> Path:
        """Create a .pt file with configurable structure.

        Args:
            filename: Name of the .pt file
            layer_sizes: List of (input_dim, output_dim) tuples for layers
            include_metadata: Whether to include training metadata

        Returns:
            Path to the created .pt file
        """
        if layer_sizes is None:
            layer_sizes = [(10, 10), (10, 5)]

        file_path = tmp_path / filename

        # Build state dict from layer sizes
        state_dict = {}
        for i, (in_dim, out_dim) in enumerate(layer_sizes):
            state_dict[f"layer{i}.weight"] = torch.randn(out_dim, in_dim)
            state_dict[f"layer{i}.bias"] = torch.zeros(out_dim)

        # Optionally wrap in checkpoint format
        if include_metadata:
            checkpoint = {
                "state_dict": state_dict,
                "epoch": 1,
                "version": "1.0",
            }
            torch.save(checkpoint, file_path)
        else:
            torch.save(state_dict, file_path)

        return file_path

    return _create_pt_file


@pytest.fixture
def mock_pipeline_weights(tmp_path):
    """Create mock weights for a complete pipeline.

    Returns:
        Path to pipeline weights file
    """
    weights_path = tmp_path / "pipeline_weights.pt"

    # Simulate a multi-node pipeline's state dict
    pipeline_state = {
        "normalizer.running_mean": torch.zeros(10),
        "normalizer.running_std": torch.ones(10),
        "transform.weight": torch.randn(5, 10),
        "transform.bias": torch.zeros(5),
        "classifier.weight": torch.randn(2, 5),
        "classifier.bias": torch.zeros(2),
    }

    torch.save(pipeline_state, weights_path)
    return weights_path


@pytest.fixture
def mock_statistical_params(tmp_path):
    """Create mock statistical parameters (mean, std, etc.) file.

    Returns:
        Path to statistical parameters file
    """
    stats_path = tmp_path / "statistics.pt"

    # Common statistical parameters
    stats = {
        "mean": torch.randn(10),
        "std": torch.rand(10) + 0.5,  # Ensure positive
        "min": torch.randn(10) - 1.0,
        "max": torch.randn(10) + 1.0,
        "count": torch.tensor(1000),
    }

    torch.save(stats, stats_path)
    return stats_path


@pytest.fixture
def mock_mismatched_weights(tmp_path):
    """Create a .pt file with intentionally mismatched keys for testing.

    Returns:
        Path to mismatched weights file
    """
    weights_path = tmp_path / "mismatched.pt"

    # Create state dict with wrong/unexpected keys
    state_dict = {
        "completely_wrong_key.weight": torch.randn(5, 5),
        "another_bad_key.bias": torch.zeros(5),
        "unexpected_layer.param": torch.randn(3, 3),
    }

    torch.save(state_dict, weights_path)
    return weights_path


@pytest.fixture
def mock_empty_weights(tmp_path):
    """Create an empty .pt file for edge case testing.

    Returns:
        Path to empty weights file
    """
    weights_path = tmp_path / "empty.pt"
    torch.save({}, weights_path)
    return weights_path


@pytest.fixture
def mock_corrupted_weights(tmp_path):
    """Create a corrupted .pt file for error handling testing.

    Returns:
        Path to corrupted file
    """
    weights_path = tmp_path / "corrupted.pt"

    # Write invalid data
    with open(weights_path, "wb") as f:
        f.write(b"This is not a valid PyTorch file!")

    return weights_path
