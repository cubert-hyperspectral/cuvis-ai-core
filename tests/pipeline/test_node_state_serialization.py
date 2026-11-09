"""Comprehensive tests for node state serialization using only state_dict."""

import pytest
import torch
import torch.nn as nn

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


class TestNodeStateSerialization:
    """Test suite for node state serialization using only state_dict."""

    def test_buffer_and_parameter_serialization(self, mock_statistical_trainable_node):
        """Test that both buffers and parameters are correctly serialized."""
        # Create and initialize node
        node = mock_statistical_trainable_node(input_dim=4, hidden_dim=3)

        # Create mock data for fitting
        mock_data = [
            {"data": torch.randn(2, 8, 8, 4)},
            {"data": torch.randn(3, 8, 8, 4)},
        ]

        # Fit the node (creates buffers)
        node.statistical_initialization(iter(mock_data))

        # Verify fitted state exists
        assert node.fitted_mean is not None
        assert node.fitted_std is not None
        assert node.fitted_transform is not None
        assert node._statistically_initialized is True

        # Unfreeze to create parameters
        node.unfreeze()

        # Verify parameters were created
        assert isinstance(node.linear_weight, nn.Parameter)
        assert isinstance(node.linear_bias, nn.Parameter)
        assert node.linear_weight.requires_grad is True
        assert node.linear_bias.requires_grad is True

        # Modify parameters to test serialization
        with torch.no_grad():
            node.linear_weight.fill_(42.0)
            node.linear_bias.fill_(7.0)

        # Get original state
        original_state = node.state_dict()

        # Verify state contains all expected keys
        expected_keys = {
            "fitted_mean",
            "fitted_std",
            "fitted_transform",
            "linear_weight",
            "linear_bias",
        }
        assert set(original_state.keys()) == expected_keys

        # Verify non-persistent buffer is not included
        assert "temp_cache" not in original_state

        # Create new node and load state
        new_node = mock_statistical_trainable_node(input_dim=4, hidden_dim=3)
        new_node.load_state_dict(original_state, strict=False)

        # Verify all state was loaded correctly
        assert torch.equal(new_node.fitted_mean, node.fitted_mean)
        assert torch.equal(new_node.fitted_std, node.fitted_std)
        assert torch.equal(new_node.fitted_transform, node.fitted_transform)
        assert torch.equal(new_node.linear_weight, node.linear_weight)
        assert torch.equal(new_node.linear_bias, node.linear_bias)

        # After loading, weights are buffers (PyTorch behavior)
        # Need to call unfreeze() to convert to parameters
        assert not isinstance(new_node.linear_weight, nn.Parameter)
        assert not isinstance(new_node.linear_bias, nn.Parameter)

        # Unfreeze to convert back to parameters
        new_node.unfreeze()

        # Now they should be parameters again
        assert isinstance(new_node.linear_weight, nn.Parameter)
        assert isinstance(new_node.linear_bias, nn.Parameter)

        # Verify fitted state is preserved
        new_node._statistically_initialized = True  # Set manually after loading

        # Test forward pass works identically
        test_input = torch.randn(1, 4, 4, 4)
        original_output = node(test_input)
        loaded_output = new_node(test_input)

        assert torch.allclose(original_output["output"], loaded_output["output"])

    def test_buffer_to_parameter_conversion_preserved(
        self, mock_statistical_trainable_node
    ):
        """Test that buffer->parameter conversion is preserved through serialization."""
        node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)

        # Initially, weights should be buffers
        assert not isinstance(node.linear_weight, nn.Parameter)
        assert not isinstance(node.linear_bias, nn.Parameter)

        # Fit the node
        mock_data = [{"data": torch.randn(1, 4, 4, 3)}]
        node.statistical_initialization(iter(mock_data))

        # Unfreeze to convert to parameters
        node.unfreeze()

        # Verify conversion
        assert isinstance(node.linear_weight, nn.Parameter)
        assert isinstance(node.linear_bias, nn.Parameter)

        # Serialize and deserialize
        state = node.state_dict()
        new_node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        new_node.load_state_dict(state, strict=False)

        # After loading, they're loaded as buffers (PyTorch behavior)
        assert not isinstance(new_node.linear_weight, nn.Parameter)
        assert not isinstance(new_node.linear_bias, nn.Parameter)

        # Call unfreeze again to convert to parameters
        new_node.unfreeze()

        # Now they should be parameters again
        assert isinstance(new_node.linear_weight, nn.Parameter)
        assert isinstance(new_node.linear_bias, nn.Parameter)
        assert new_node.linear_weight.requires_grad is True
        assert new_node.linear_bias.requires_grad is True

    def test_partial_state_loading(self, mock_statistical_trainable_node):
        """Test loading with missing or extra keys."""
        node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        mock_data = [{"data": torch.randn(1, 4, 4, 3)}]
        node.statistical_initialization(iter(mock_data))
        node.unfreeze()

        # Get full state
        full_state = node.state_dict()

        # Test loading with missing key (non-strict)
        partial_state = {k: v for k, v in full_state.items() if k != "linear_bias"}
        new_node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)

        # Should work with strict=False
        new_node.load_state_dict(partial_state, strict=False)

        # Should fail with strict=True
        with pytest.raises(RuntimeError):
            new_node.load_state_dict(partial_state, strict=True)

        # Test loading with extra key (non-strict)
        extra_state = {**full_state, "extra_key": torch.tensor([1.0])}
        new_node2 = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)

        # Should work with strict=False
        new_node2.load_state_dict(extra_state, strict=False)

        # Should fail with strict=True
        with pytest.raises(RuntimeError):
            new_node2.load_state_dict(extra_state, strict=True)

    def test_pipeline_integration_with_mock_node(
        self, mock_statistical_trainable_node, tmp_path
    ):
        """Test that mock node works correctly in pipeline serialization."""
        # Create pipeline with mock node
        pipeline = CuvisPipeline("test_pipeline")
        mock_node = mock_statistical_trainable_node(input_dim=4, hidden_dim=3)
        pipeline._graph.add_node(mock_node)

        # Fit the node
        mock_data = [{"data": torch.randn(2, 8, 8, 4)}]
        mock_node.statistical_initialization(iter(mock_data))
        mock_node.unfreeze()

        # Modify parameters
        with torch.no_grad():
            mock_node.linear_weight.fill_(99.0)
            mock_node.linear_bias.fill_(11.0)

        # Save pipeline
        config_path = tmp_path / "test_pipeline.yaml"
        pipeline.save_to_file(config_path)

        # Verify files were created
        assert config_path.exists()
        assert config_path.with_suffix(".pt").exists()

        # Load pipeline (use non-strict loading for nodes with fitted state)
        loaded_pipeline = CuvisPipeline.load_pipeline(
            config_path,
            weights_path=config_path.with_suffix(".pt"),
            strict_weight_loading=False,
        )

        # Get loaded node
        loaded_node = list(loaded_pipeline.nodes)[0]

        # Verify state was preserved
        assert torch.all(loaded_node.linear_weight == 99.0)
        assert torch.all(loaded_node.linear_bias == 11.0)
        assert torch.equal(loaded_node.fitted_mean, mock_node.fitted_mean)
        assert torch.equal(loaded_node.fitted_std, mock_node.fitted_std)
        assert torch.equal(loaded_node.fitted_transform, mock_node.fitted_transform)

        # After loading, weights are buffers (PyTorch behavior)
        # Would need to call unfreeze() to convert back to parameters
        assert not isinstance(loaded_node.linear_weight, nn.Parameter)
        assert not isinstance(loaded_node.linear_bias, nn.Parameter)

        # Call unfreeze to convert back to parameters
        loaded_node.unfreeze()

        # Now they should be parameters
        assert isinstance(loaded_node.linear_weight, nn.Parameter)
        assert isinstance(loaded_node.linear_bias, nn.Parameter)

    def test_restore_weights_round_trip_marks_statistical_initialization(
        self, mock_statistical_trainable_node, tmp_path
    ):
        """save -> rebuild from YAML -> _restore_weights_from_checkpoint restores
        the exact fitted buffers and marks the node statistically initialized,
        so evaluation can skip re-deriving statistics."""
        pipeline = CuvisPipeline("roundtrip")
        node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        pipeline._graph.add_node(node)
        node.statistical_initialization(iter([{"data": torch.randn(1, 4, 4, 3)}]))

        config_path = tmp_path / "roundtrip.yaml"
        pipeline.save_to_file(config_path)
        weights_path = config_path.with_suffix(".pt")

        # Rebuild from config only: fresh node, no weights, not initialized.
        rebuilt = CuvisPipeline.load_pipeline(config_path)
        rebuilt_node = list(rebuilt.nodes)[0]
        assert rebuilt_node._statistically_initialized is False
        assert not torch.equal(rebuilt_node.fitted_mean, node.fitted_mean)

        missing = rebuilt._restore_weights_from_checkpoint(weights_path)

        assert missing == []
        assert rebuilt_node._statistically_initialized is True
        assert torch.equal(rebuilt_node.fitted_mean, node.fitted_mean)
        assert torch.equal(rebuilt_node.fitted_std, node.fitted_std)
        assert torch.equal(rebuilt_node.fitted_transform, node.fitted_transform)

    def test_restore_weights_returns_missing_nodes(
        self, mock_statistical_trainable_node, tmp_path
    ):
        """Nodes absent from the checkpoint are returned to the caller (a
        stale or renamed pipeline must be detectable, not silent)."""
        pipeline = CuvisPipeline("missing")
        node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        pipeline._graph.add_node(node)
        node.statistical_initialization(iter([{"data": torch.randn(1, 4, 4, 3)}]))

        config_path = tmp_path / "missing.yaml"
        pipeline.save_to_file(config_path)
        weights_path = config_path.with_suffix(".pt")

        checkpoint = torch.load(weights_path, weights_only=False)
        checkpoint["state_dict"].pop(node.name)
        torch.save(checkpoint, weights_path)

        assert pipeline._restore_weights_from_checkpoint(weights_path) == [node.name]

    def test_restore_weights_corrupt_file_raises(
        self, mock_statistical_trainable_node, tmp_path
    ):
        """A file that is not a torch checkpoint fails loudly, not silently."""
        import pickle

        pipeline = CuvisPipeline("corrupt")
        pipeline._graph.add_node(
            mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        )
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"this is not a torch checkpoint")

        with pytest.raises((RuntimeError, pickle.UnpicklingError, EOFError)):
            pipeline._restore_weights_from_checkpoint(bad)

    def test_save_pipeline_yaml_only(self, mock_statistical_trainable_node, tmp_path):
        """Test save_to_file(save_weights=False) writes only YAML config."""
        pipeline = CuvisPipeline("test_pipeline_yaml_only")
        mock_node = mock_statistical_trainable_node(input_dim=4, hidden_dim=3)
        pipeline._graph.add_node(mock_node)

        config_path = tmp_path / "yaml_only_pipeline.yaml"
        pipeline.save_to_file(config_path, save_weights=False)

        assert config_path.exists()
        assert not config_path.with_suffix(".pt").exists()

    @pytest.mark.gpu
    def test_device_and_dtype_handling(self, mock_statistical_trainable_node):
        """Test that serialization handles device and dtype correctly."""
        # Create node on CPU
        node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        mock_data = [{"data": torch.randn(1, 4, 4, 3)}]
        node.statistical_initialization(iter(mock_data))
        node.unfreeze()

        # Move to GPU
        node = node.cuda()

        # Verify on GPU
        assert node.fitted_mean.device.type == "cuda"
        assert node.linear_weight.device.type == "cuda"

        # Serialize
        state = node.state_dict()

        # Create new node on CPU
        new_node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)

        # Load state (should handle device transfer)
        new_node.load_state_dict(state, strict=False)

        # Verify loaded on CPU
        assert new_node.fitted_mean.device.type == "cpu"
        assert new_node.linear_weight.device.type == "cpu"

        # Values should be equal (ignoring device)
        assert torch.equal(new_node.fitted_mean.cpu(), node.fitted_mean.cpu())
        assert torch.equal(new_node.linear_weight.cpu(), node.linear_weight.cpu())


class _StatelessNode(Node):
    """A node with no parameters, buffers or extra state: nothing to load."""

    def forward(self, **kwargs):
        return {}


class TestRestoreInitialTrainWeights:
    """Tests for _restore_initial_train_weights (train-mode initial weights)."""

    def _write_checkpoint(self, path, state_dict: dict) -> None:
        torch.save({"state_dict": state_dict, "metadata": {}}, path)

    def test_fitted_node_is_never_loaded_even_when_present(
        self, mock_statistical_trainable_node, tmp_path
    ):
        """A node this run fits is skipped even though the checkpoint has a row
        for it (the row may have an incompatible shape: a 7,200-row coreset
        against this run's 80,000-row bank)."""
        pipeline = CuvisPipeline("train_initial")
        fitted_node = mock_statistical_trainable_node(
            input_dim=3, hidden_dim=2, name="fitted_node"
        )
        pipeline._graph.add_node(fitted_node)

        checkpoint_node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        checkpoint_node.statistical_initialization(
            iter([{"data": torch.randn(1, 4, 4, 3)}])
        )
        stale_state = checkpoint_node.state_dict()
        weights_path = tmp_path / "weights.pt"
        self._write_checkpoint(weights_path, {"fitted_node": stale_state})

        before = {k: v.clone() for k, v in fitted_node.state_dict().items()}
        skipped = pipeline._restore_initial_train_weights(
            weights_path, fitted_node_names={"fitted_node"}
        )

        assert skipped == ["fitted_node"]
        for key, value in before.items():
            assert torch.equal(fitted_node.state_dict()[key], value)

    def test_other_stateful_node_loads_strictly(
        self, mock_statistical_trainable_node, tmp_path
    ):
        """A node not fitted by this run loads its initial weights strictly."""
        pipeline = CuvisPipeline("train_initial")
        other_node = mock_statistical_trainable_node(
            input_dim=3, hidden_dim=2, name="other_node"
        )
        pipeline._graph.add_node(other_node)

        source_node = mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        source_node.statistical_initialization(
            iter([{"data": torch.randn(1, 4, 4, 3)}])
        )
        weights_path = tmp_path / "weights.pt"
        self._write_checkpoint(weights_path, {"other_node": source_node.state_dict()})

        skipped = pipeline._restore_initial_train_weights(
            weights_path, fitted_node_names=set()
        )

        assert skipped == []
        assert torch.equal(other_node.fitted_mean, source_node.fitted_mean)
        assert other_node._statistically_initialized is True

    def test_missing_required_stateful_node_raises(
        self, mock_statistical_trainable_node, tmp_path
    ):
        """A stateful node this run does not fit must be present in the
        checkpoint; a missing one raises, naming it."""
        pipeline = CuvisPipeline("train_initial")
        other_node = mock_statistical_trainable_node(
            input_dim=3, hidden_dim=2, name="other_node"
        )
        pipeline._graph.add_node(other_node)

        weights_path = tmp_path / "weights.pt"
        self._write_checkpoint(weights_path, {})

        with pytest.raises(ValueError, match="other_node"):
            pipeline._restore_initial_train_weights(
                weights_path, fitted_node_names=set()
            )

    def test_stateless_node_is_skipped_without_requiring_a_checkpoint_row(
        self, tmp_path
    ):
        """A stateless node needs no checkpoint entry, fitted or not."""
        pipeline = CuvisPipeline("train_initial")
        stateless = _StatelessNode(name="stateless")
        pipeline._graph.add_node(stateless)

        weights_path = tmp_path / "weights.pt"
        self._write_checkpoint(weights_path, {})

        skipped = pipeline._restore_initial_train_weights(
            weights_path, fitted_node_names=set()
        )

        assert skipped == []

    def test_missing_weights_file_raises_file_not_found(
        self, mock_statistical_trainable_node, tmp_path
    ):
        pipeline = CuvisPipeline("train_initial")
        pipeline._graph.add_node(
            mock_statistical_trainable_node(input_dim=3, hidden_dim=2)
        )

        with pytest.raises(FileNotFoundError):
            pipeline._restore_initial_train_weights(
                tmp_path / "absent.pt", fitted_node_names=set()
            )
