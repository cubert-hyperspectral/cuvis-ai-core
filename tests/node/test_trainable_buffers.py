"""Tests for TRAINABLE_BUFFERS, frozen property, and freeze/unfreeze buffer conversion."""

import pytest
import torch
from torch import nn

from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


class _BufferNode(Node):
    """Minimal node with TRAINABLE_BUFFERS for testing."""

    TRAINABLE_BUFFERS = ("weight", "bias")

    INPUT_SPECS = {
        "data": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }
    OUTPUT_SPECS = {
        "output": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }

    def __init__(self, dim: int = 4, **kwargs):
        self.dim = dim
        super().__init__(dim=dim, **kwargs)
        self.register_buffer("weight", torch.randn(dim, dim))
        self.register_buffer("bias", torch.zeros(dim))

    def forward(self, data, **_):
        return {"output": data @ self.weight + self.bias}


# -- __init_subclass__ validation -------------------------------------------


def test_valid_trainable_buffers():
    """TRAINABLE_BUFFERS as tuple of strings is accepted."""
    # _BufferNode already defined above — no TypeError raised
    node = _BufferNode()
    assert node.TRAINABLE_BUFFERS == ("weight", "bias")


def test_invalid_trainable_buffers_list():
    """TRAINABLE_BUFFERS as list raises TypeError."""
    with pytest.raises(TypeError, match="must be a tuple of strings"):

        class _Bad(Node):
            TRAINABLE_BUFFERS = ["weight"]

            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **_):
                return {}


def test_invalid_trainable_buffers_non_string():
    """TRAINABLE_BUFFERS containing non-strings raises TypeError."""
    with pytest.raises(TypeError, match="must be a tuple of strings"):

        class _Bad(Node):
            TRAINABLE_BUFFERS = (1, 2)

            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **_):
                return {}


# -- frozen property --------------------------------------------------------


def test_frozen_default_false():
    node = _BufferNode()
    assert node.frozen is False


def test_frozen_after_freeze():
    node = _BufferNode()
    node.freeze()
    assert node.frozen is True


def test_frozen_after_unfreeze():
    node = _BufferNode()
    node.freeze()
    node.unfreeze()
    assert node.frozen is False


# -- unfreeze: buffer → nn.Parameter ----------------------------------------


def test_unfreeze_converts_buffers_to_parameters():
    node = _BufferNode()
    assert "weight" in dict(node.named_buffers())
    assert "bias" in dict(node.named_buffers())

    node.unfreeze()

    assert isinstance(node.weight, nn.Parameter)
    assert isinstance(node.bias, nn.Parameter)
    assert "weight" not in dict(node.named_buffers())
    assert "bias" not in dict(node.named_buffers())


def test_unfreeze_preserves_values():
    node = _BufferNode()
    orig_weight = node.weight.clone()
    orig_bias = node.bias.clone()

    node.unfreeze()

    assert torch.equal(node.weight.data, orig_weight)
    assert torch.equal(node.bias.data, orig_bias)


# -- freeze: nn.Parameter → buffer -----------------------------------------


def test_freeze_converts_parameters_to_buffers():
    node = _BufferNode()
    node.unfreeze()
    assert isinstance(node.weight, nn.Parameter)

    node.freeze()

    assert "weight" in dict(node.named_buffers())
    assert "bias" in dict(node.named_buffers())
    assert not isinstance(node.weight, nn.Parameter)


def test_freeze_unfreeze_roundtrip_preserves_values():
    node = _BufferNode()
    orig_weight = node.weight.clone()

    node.unfreeze()
    node.freeze()

    assert torch.equal(node.weight, orig_weight)


# -- error paths ------------------------------------------------------------


def test_unfreeze_missing_buffer_raises():
    class _Missing(Node):
        TRAINABLE_BUFFERS = ("nonexistent",)

        INPUT_SPECS = {}
        OUTPUT_SPECS = {}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self, **_):
            return {}

    node = _Missing()
    with pytest.raises(AttributeError, match="nonexistent"):
        node.unfreeze()


def test_freeze_missing_buffer_raises():
    class _Missing(Node):
        TRAINABLE_BUFFERS = ("nonexistent",)

        INPUT_SPECS = {}
        OUTPUT_SPECS = {}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self, **_):
            return {}

    node = _Missing()
    with pytest.raises(AttributeError, match="nonexistent"):
        node.freeze()
