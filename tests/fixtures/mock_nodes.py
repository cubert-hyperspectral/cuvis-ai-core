"""Mock nodes for testing serialization patterns."""

import pytest
import torch
import torch.nn as nn

from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec


class MockStatisticalTrainableNode(Node):
    """Mock node with both buffers and parameters for testing serialization."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input data [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "output": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Processed output [B, H, W, C]",
        )
    }

    def __init__(self, input_dim: int = 10, hidden_dim: int = 5, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)

        # Statistical state (buffers) - fitted during statistical training
        # Initialize as zero tensors so load_state_dict works properly
        self.register_buffer("fitted_mean", torch.zeros(input_dim))
        self.register_buffer("fitted_std", torch.ones(input_dim))
        self.register_buffer("fitted_transform", torch.zeros(hidden_dim, input_dim))

        # Trainable parameters (start as buffers, become parameters after unfreeze)
        # Note: These operate on hidden_dim outputs from fitted_transform
        initial_weights = torch.randn(hidden_dim, hidden_dim) * 0.01
        initial_bias = torch.zeros(hidden_dim)
        self.register_buffer("linear_weight", initial_weights)
        self.register_buffer("linear_bias", initial_bias)

        # Non-persistent buffer (won't be saved)
        self.register_buffer("temp_cache", torch.zeros(10), persistent=False)

        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Statistical initialization - compute mean, std, and PCA-like transform."""
        all_data = []

        for batch in input_stream:
            data = batch.get("data")
            if data is not None:
                # Flatten spatial dimensions: [B, H, W, C] -> [B*H*W, C]
                flat_data = data.reshape(-1, data.shape[-1])
                all_data.append(flat_data)

        if not all_data:
            raise RuntimeError("No data provided for fitting")

        # Concatenate all data
        X = torch.cat(all_data, dim=0)  # [N, C]

        # Compute statistics
        self.fitted_mean = X.mean(dim=0)  # [C]
        self.fitted_std = X.std(dim=0) + 1e-6  # [C]

        # Compute simple PCA-like transform (just random projection for testing)
        # Handle case where hidden_dim might be larger than input_dim
        if self.hidden_dim <= self.input_dim:
            U, _, _ = torch.svd(X.T @ X)
            self.fitted_transform = U[
                : self.hidden_dim, :
            ].clone()  # [hidden_dim, input_dim]
        else:
            # When hidden_dim > input_dim, pad with random values
            U, _, _ = torch.svd(X.T @ X)
            transform = torch.randn(self.hidden_dim, self.input_dim) * 0.01
            transform[: self.input_dim, :] = U.clone()
            self.fitted_transform = transform

        self._statistically_initialized = True

    def unfreeze(self) -> None:
        """Convert trainable buffers to nn.Parameters."""
        if self.linear_weight is not None and self.linear_bias is not None:
            # Convert buffers to parameters
            self.linear_weight = nn.Parameter(self.linear_weight.clone())
            self.linear_bias = nn.Parameter(self.linear_bias.clone())
        super().unfreeze()

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Forward pass using both statistical and trainable components."""
        if not self._statistically_initialized:
            raise RuntimeError("Node must be initialized before forward pass")

        B, H, W, C = data.shape
        flat_data = data.reshape(-1, C)  # [B*H*W, C]

        # Apply statistical normalization
        normalized = (flat_data - self.fitted_mean) / self.fitted_std

        # Apply statistical transform
        transformed = normalized @ self.fitted_transform.T  # [B*H*W, hidden_dim]

        # Apply trainable linear layer
        output = (
            transformed @ self.linear_weight.T + self.linear_bias
        )  # [B*H*W, hidden_dim]

        # Reshape back
        output = output.reshape(B, H, W, self.hidden_dim)

        return {"output": output}


class LentilsAnomalyDataNode(Node):
    """Mock data node for testing - replaces cuvis-ai LentilsAnomalyDataNode.

    This is a pass-through node that accepts cube and wavelengths inputs
    and outputs them along with a generated mask for anomaly detection testing.
    Automatically converts uint16 cubes to float32.
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.uint16,  # Accept uint16 like original
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "wavelengths": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1),
            description="Wavelength metadata [B, C]",
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "wavelengths": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1),
            description="Wavelength metadata [B, C]",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1),
            description="Anomaly mask [B, H, W]",
        ),
    }

    def __init__(self, normal_class_ids=None, **kwargs):
        self.normal_class_ids = normal_class_ids or [0, 1]
        super().__init__(normal_class_ids=normal_class_ids, **kwargs)

    def forward(
        self, cube: torch.Tensor, wavelengths: torch.Tensor, **_
    ) -> dict[str, torch.Tensor]:
        """Pass through cube and wavelengths, generate dummy mask.

        Converts cube to float32 if needed (e.g., from uint16).
        """
        B, H, W, C = cube.shape

        # Convert to float32 if needed
        if cube.dtype != torch.float32:
            cube = cube.to(torch.float32)

        # Generate a simple random mask for testing
        mask = torch.rand(B, H, W) > 0.8  # ~20% anomalies

        return {
            "cube": cube,
            "wavelengths": wavelengths,
            "mask": mask,
        }


class SoftChannelSelector(Node):
    """Mock channel selector for testing - replaces cuvis-ai SoftChannelSelector.

    This node performs soft channel selection on hyperspectral data with
    trainable parameters for statistical training.
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input data [B, H, W, C]",
        ),
    }

    OUTPUT_SPECS = {
        "selected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Channel-selected data [B, H, W, C]",
        ),
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights [C]",
        ),
    }

    def __init__(
        self,
        n_select: int = 3,
        input_channels: int = 61,
        init_method: str = "variance",
        temperature_init: float = 5.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.9,
        hard: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.n_select = n_select
        self.input_channels = input_channels
        self.init_method = init_method
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.hard = hard
        self.eps = eps

        super().__init__(
            n_select=n_select,
            input_channels=input_channels,
            init_method=init_method,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            hard=hard,
            eps=eps,
            **kwargs,
        )

        # Initialize channel logits as buffer (becomes parameter after unfreeze)
        initial_logits = torch.zeros(input_channels)
        self.register_buffer("channel_logits", initial_logits)

        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Statistical initialization - compute channel importance based on variance."""
        all_data = []

        for batch in input_stream:
            data = batch.get("data")
            if data is not None:
                # Keep data as [B, H, W, C]
                all_data.append(data)

        if not all_data:
            raise RuntimeError("No data provided for statistical initialization")

        # Concatenate all data
        X = torch.cat(all_data, dim=0)  # [B, H, W, C]

        # Compute variance across all spatial and batch dimensions
        # Flatten to [B*H*W, C]
        X_flat = X.reshape(-1, X.shape[-1])
        channel_variance = X_flat.var(dim=0)  # [C]

        # Initialize logits based on variance (higher variance = higher logit)
        if self.init_method == "variance":
            # Use log of variance as logits
            self.channel_logits = torch.log(channel_variance + self.eps)
        else:
            # Uniform initialization (zeros)
            self.channel_logits = torch.zeros(self.input_channels)

        self._statistically_initialized = True

    def unfreeze(self) -> None:
        """Convert channel logits buffer to nn.Parameter for gradient training."""
        if self.channel_logits is not None:
            self.channel_logits = nn.Parameter(self.channel_logits.clone())
        super().unfreeze()

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Forward pass - apply soft channel selection."""
        # Compute selection weights from logits
        weights = torch.softmax(self.channel_logits, dim=0)

        # Apply soft weighting to channels
        # data: [B, H, W, C], weights: [C]
        selected = data * weights.view(1, 1, 1, -1)

        return {
            "selected": selected,
            "weights": weights,
        }


class MinMaxNormalizer(Node):
    """Mock min-max normalizer for testing - replaces cuvis-ai MinMaxNormalizer.

    Can operate in two modes:
    1. Per-sample normalization (default): min/max computed per batch
    2. Global normalization: uses running statistics from initialization
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input data tensor to normalize (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Normalized output tensor",
        )
    }

    def __init__(self, eps: float = 1e-6, use_running_stats: bool = True, **kwargs):
        self.eps = float(eps)
        self.use_running_stats = use_running_stats
        super().__init__(eps=eps, use_running_stats=use_running_stats, **kwargs)

        # Running statistics for global normalization
        self.register_buffer("running_min", torch.tensor(float("nan")))
        self.register_buffer("running_max", torch.tensor(float("nan")))

        # Only require initialization when running stats are requested
        self._requires_initial_fit_override = self.use_running_stats

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Compute global min/max from data iterator."""
        all_mins = []
        all_maxs = []

        for batch_data in input_stream:
            # Extract data from port-based dict
            x = batch_data.get("data")
            if x is not None:
                # Flatten spatial dimensions
                flat = x.reshape(x.shape[0], -1)
                batch_min = flat.min()
                batch_max = flat.max()
                all_mins.append(batch_min)
                all_maxs.append(batch_max)

        if all_mins:
            self.running_min = torch.stack(all_mins).min()
            self.running_max = torch.stack(all_maxs).max()
            self._statistically_initialized = True

    def _is_initialized(self) -> bool:
        """Check if running statistics have been initialized."""
        return not torch.isnan(self.running_min).item()

    def validate_serialization_support(self) -> tuple[bool, str]:
        """Validate that this node can be properly serialized."""
        return True, "OK"

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Normalize input data (BHWC only)."""
        B, H, W, C = data.shape
        flat = data.view(B, -1, C)

        # Use running stats if available and initialized
        if self.use_running_stats and self._is_initialized():
            mins = self.running_min
            maxs = self.running_max
            ranges = torch.clamp(maxs - mins, min=self.eps)
            scaled = (flat - mins) / ranges
        else:
            # Per-sample normalization
            mins = flat.min(dim=1, keepdim=True).values
            maxs = flat.max(dim=1, keepdim=True).values
            ranges = torch.clamp(maxs - mins, min=self.eps)
            scaled = (flat - mins) / ranges

        normalized = scaled.view(B, H, W, C)
        return {"normalized": normalized}


class SimpleLossNode(Node):
    """Simple loss node for testing training workflows.

    Computes MSE loss between predictions and targets.
    Only executes during TRAIN, VAL, and TEST stages.
    """

    INPUT_SPECS = {
        "predictions": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Model predictions",
        ),
        "targets": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Ground truth targets",
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(
            dtype=torch.float32,
            shape=(),
            description="Scalar loss value",
        ),
    }

    def __init__(self, weight: float = 1.0, **kwargs):
        from cuvis_ai_schemas.enums import ExecutionStage

        self.weight = weight
        super().__init__(weight=weight, **kwargs)

        # Loss nodes only execute during training/validation/test
        self.execution_stages = {
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        }

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, **_
    ) -> dict[str, torch.Tensor]:
        """Compute MSE loss between predictions and targets."""
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predictions, targets)

        # Apply weight
        loss = loss * self.weight

        return {"loss": loss}


# Pytest Fixtures
# ----------------


@pytest.fixture
def mock_statistical_trainable_node():
    """Factory fixture for MockStatisticalTrainableNode.

    Returns the class itself so tests can instantiate it with custom parameters.

    Usage:
        node = mock_statistical_trainable_node(input_dim=4, hidden_dim=3)
    """
    return MockStatisticalTrainableNode
