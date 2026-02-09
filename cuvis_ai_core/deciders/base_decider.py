from abc import abstractmethod
from typing import Any

from torch import Tensor

from cuvis_ai_core.node.node import Node


import torch
from cuvis_ai_schemas.pipeline import PortSpec
from cuvis_ai_schemas.execution.context import Context


class BinaryDecider(Node):
    """
    Abstract class for Decision Making Nodes.

    The decider nodes transform a prediction state into a final prediction
    based on the task that needs to be accomplished.
    """

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits to threshold (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        self.threshold = threshold
        # Accept arbitrary args/kwargs so subclasses can forward their init
        # parameters to the Serializable base for automatic hparam capture.
        super().__init__(threshold=threshold, **kwargs)

    @abstractmethod
    def forward(  # type: ignore[override]
        self,
        logits: Tensor,
        context: Context,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply decisioning on channels-last data.

        Args:
            logits: Tensor shaped (B, H, W, C) containing logits.

        Returns:
            Dictionary with "decisions" key containing (B, H, W, 1) decision mask.
        """

        pass
