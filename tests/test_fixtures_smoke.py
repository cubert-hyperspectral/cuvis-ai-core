"""Smoke tests to verify test infrastructure is working correctly."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
def test_simple_input_node_fixture(simple_input_node):
    """Verify simple_input_node fixture works."""
    NodeClass = simple_input_node
    node = NodeClass(batch_size=2, channels=5, height=64, width=64)
    
    result = node.forward()
    
    assert "output" in result
    assert result["output"].shape == (2, 64, 64, 5)


@pytest.mark.unit
def test_simple_transform_node_fixture(simple_transform_node):
    """Verify simple_transform_node fixture works."""
    NodeClass = simple_transform_node
    node = NodeClass(scale=3.0)
    
    input_data = torch.randn(2, 64, 64, 5)
    result = node.forward(input=input_data)
    
    assert "output" in result
    assert torch.allclose(result["output"], input_data * 3.0)


@pytest.mark.unit
def test_mock_pt_file_fixture(mock_pt_file):
    """Verify mock_pt_file fixture creates valid .pt file."""
    assert mock_pt_file.exists()
    
    state_dict = torch.load(mock_pt_file)
    
    assert "layer1.weight" in state_dict
    assert "layer1.bias" in state_dict
    assert state_dict["layer1.weight"].shape == (10, 10)


@pytest.mark.unit
def test_hyperspectral_batch_fixture(hyperspectral_batch):
    """Verify hyperspectral_batch fixture generates correct data."""
    assert "cube" in hyperspectral_batch
    assert "wavelengths" in hyperspectral_batch
    assert "mask" in hyperspectral_batch
    
    assert hyperspectral_batch["cube"].shape == (2, 64, 64, 5)
    assert hyperspectral_batch["wavelengths"].shape == (2, 5)
    assert hyperspectral_batch["mask"].shape == (2, 64, 64)


@pytest.mark.unit
def test_batch_factory_fixture(batch_factory):
    """Verify batch_factory fixture creates customizable batches."""
    batch = batch_factory(
        batch_size=8,
        height=128,
        width=128,
        channels=10,
        include_labels=True
    )
    
    assert batch["cube"].shape == (8, 128, 128, 10)
    assert batch["wavelengths"].shape == (8, 10)
    assert "mask" in batch
    assert "labels" in batch


@pytest.mark.unit
def test_tmp_config_dir_fixture(tmp_config_dir):
    """Verify tmp_config_dir fixture creates directory."""
    assert tmp_config_dir.exists()
    assert tmp_config_dir.is_dir()
    
    # Can write files to it
    test_file = tmp_config_dir / "test.yaml"
    test_file.write_text("test: content")
    assert test_file.exists()


@pytest.mark.unit
def test_minimal_pipeline_dict_fixture(minimal_pipeline_dict):
    """Verify minimal_pipeline_dict fixture structure."""
    assert "version" in minimal_pipeline_dict
    assert "nodes" in minimal_pipeline_dict
    assert "connections" in minimal_pipeline_dict
    
    assert "input" in minimal_pipeline_dict["nodes"]
    assert "output" in minimal_pipeline_dict["nodes"]
