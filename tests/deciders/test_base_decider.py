"""Tests for BinaryDecider base class."""

import torch
import pytest

from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


class ConcreteBinaryDecider(BinaryDecider):
    """Concrete implementation of BinaryDecider for testing."""

    def forward(
        self, logits: torch.Tensor, context: Context, **_
    ) -> dict[str, torch.Tensor]:
        """Apply simple thresholding logic."""
        decisions = (logits > self.threshold).to(torch.bool)
        # Ensure output is (B, H, W, 1)
        if decisions.shape[-1] != 1:
            decisions = decisions[..., :1]
        return {"decisions": decisions}


class TestBinaryDecider:
    """Test suite for BinaryDecider base class."""

    def test_init_default_threshold(self):
        """Test BinaryDecider initialization with default threshold."""
        decider = ConcreteBinaryDecider()
        assert decider.threshold == 0.5

    def test_init_custom_threshold(self):
        """Test BinaryDecider initialization with custom threshold."""
        decider = ConcreteBinaryDecider(threshold=0.7)
        assert decider.threshold == 0.7

    def test_input_specs_defined(self):
        """Test that INPUT_SPECS are properly defined."""
        assert "logits" in BinaryDecider.INPUT_SPECS
        spec = BinaryDecider.INPUT_SPECS["logits"]
        assert isinstance(spec, PortSpec)
        assert spec.dtype == torch.float32
        assert spec.shape == (-1, -1, -1, -1)
        assert "logits" in spec.description.lower()

    def test_output_specs_defined(self):
        """Test that OUTPUT_SPECS are properly defined."""
        assert "decisions" in BinaryDecider.OUTPUT_SPECS
        spec = BinaryDecider.OUTPUT_SPECS["decisions"]
        assert isinstance(spec, PortSpec)
        assert spec.dtype == torch.bool
        assert spec.shape == (-1, -1, -1, 1)
        assert "decision" in spec.description.lower()

    def test_forward_with_context(self):
        """Test forward method executes with context parameter."""
        decider = ConcreteBinaryDecider(threshold=0.5)
        context = Context()

        # Create test logits (B=1, H=4, W=4, C=1)
        logits = torch.tensor(
            [
                [
                    [[0.3], [0.6], [0.4], [0.8]],
                    [[0.2], [0.7], [0.5], [0.9]],
                    [[0.1], [0.55], [0.45], [0.75]],
                    [[0.0], [1.0], [0.25], [0.65]],
                ]
            ]
        )

        result = decider.forward(logits, context)

        assert "decisions" in result
        decisions = result["decisions"]
        assert decisions.dtype == torch.bool
        assert decisions.shape == (1, 4, 4, 1)

        # Check threshold logic
        expected = logits > 0.5
        torch.testing.assert_close(decisions.float(), expected.float())

    def test_forward_different_thresholds(self):
        """Test that different thresholds produce different results."""
        context = Context()
        logits = torch.rand(1, 3, 3, 1)

        decider_low = ConcreteBinaryDecider(threshold=0.3)
        decider_high = ConcreteBinaryDecider(threshold=0.7)

        decisions_low = decider_low.forward(logits, context)["decisions"]
        decisions_high = decider_high.forward(logits, context)["decisions"]

        # Lower threshold should have more True values (or equal)
        assert decisions_low.sum() >= decisions_high.sum()

    def test_abstract_forward_not_implemented(self):
        """Test that BinaryDecider cannot be instantiated directly."""
        # This should fail because forward is abstract
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BinaryDecider()

    def test_kwargs_forwarded_to_super(self):
        """Test that kwargs are forwarded to parent Node class."""
        # Node class accepts arbitrary kwargs for serialization
        decider = ConcreteBinaryDecider(threshold=0.6, custom_param="test")
        assert decider.threshold == 0.6
        # The custom_param should be stored by the Node's Serializable base
