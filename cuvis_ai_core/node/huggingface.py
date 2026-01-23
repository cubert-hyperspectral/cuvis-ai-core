"""HuggingFace model integration nodes for cuvis.ai pipeline system.

This module provides nodes for interfacing with HuggingFace models via:
1. API backend - gradio_client for HF Spaces (Phase 1)
2. Local backend - transformers for local models (Phase 2)

Example
-------
>>> # API inference (Phase 1)
>>> adaclip_api = AdaCLIPAPINode(space_url="Caoyunkang/AdaCLIP")
>>> result = adaclip_api.forward(image=rgb_tensor)
"""

from __future__ import annotations

import os
from typing import Any

import torch
from dotenv import load_dotenv
from gradio_client import Client
from loguru import logger
from transformers import AutoModel

from cuvis_ai_core.node.node import Node

load_dotenv(override=True)


class HuggingFaceAPINode(Node):
    """Base class for HuggingFace Spaces API integration.

    This node enables calling HuggingFace Spaces via their API using gradio_client.
    API calls are NOT differentiable and should only be used for inference/testing.

    Parameters
    ----------
    space_url : str
        HuggingFace Space URL (e.g., "username/space-name")
    use_hf_token : bool, optional
        Whether to use HF_TOKEN from environment (default: True)
    api_timeout : float, optional
        Timeout for API calls in seconds (default: 60.0)
    **kwargs
        Additional arguments passed to Node base class

    Attributes
    ----------
    space_url : str
        The HuggingFace Space URL
    client : gradio_client.Client or None
        The Gradio client instance (lazy loaded)

    Notes
    -----
    - API calls are not differentiable (no gradient flow)
    - Requires internet connection
    - Subject to HF Space rate limits
    - Use local backend (Phase 2) for gradient training
    """

    def __init__(
        self,
        space_url: str,
        api_timeout: float = 60.0,
        **kwargs,
    ) -> None:
        self.space_url = space_url
        self.api_timeout = api_timeout

        super().__init__(
            space_url=space_url,
            api_timeout=api_timeout,
            **kwargs,
        )

        # Lazy load client (only when needed)
        self._client = None

    @property
    def client(self) -> Client:
        """Lazy load Gradio client."""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client

    def _initialize_client(self) -> Client:
        """Initialize Gradio client with optional HF token.

        Returns
        -------
        gradio_client.Client
            Initialized Gradio client

        Raises
        ------
        ImportError
            If gradio_client is not installed
        RuntimeError
            If Space connection fails
        """

        # Get HF token from environment if requested
        hf_token = os.getenv("HF_TOKEN")

        # Initialize client
        try:
            logger.info(f"Connecting to HuggingFace Space: {self.space_url}")
            client = Client(self.space_url, hf_token=hf_token)

            logger.success(f"Successfully connected to {self.space_url}")
            return client
        except TypeError as e:
            raise RuntimeError(
                f"Failed to connect to HuggingFace Space '{self.space_url}': {e}"
            ) from e

    def forward(self, **inputs: Any) -> dict[str, Any]:
        """Execute API call to HuggingFace Space.

        This method should be overridden by subclasses to implement
        model-specific API calls.

        Parameters
        ----------
        **inputs
            Input tensors/data for the model

        Returns
        -------
        dict[str, Any]
            Model outputs

        Raises
        ------
        NotImplementedError
            If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward() method"
        )


# ============================================================================
# Phase 2: Local Model Loading
# ============================================================================


class HuggingFaceLocalNode(Node):
    """Base class for local HuggingFace model integration."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            **kwargs,
        )

        self._model: torch.nn.Module | None = None

    @property
    def model(self) -> torch.nn.Module:
        """Lazy load and cache the HF model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> torch.nn.Module:
        """Load model from HuggingFace Hub (frozen by default)."""
        hf_token = os.getenv("HF_TOKEN")

        try:
            logger.info(f"Loading HF model locally: {self.model_name}")
            logger.info(f"Cache dir: {self.cache_dir or 'default'}")

            model = AutoModel.from_pretrained(
                self.model_name,
                token=hf_token,
                cache_dir=self.cache_dir,
            )

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            logger.success(f"Model loaded and frozen: {self.model_name}")
            return model
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(
                f"Failed to load model '{self.model_name}': {exc}\n"
                "Check model name, network access, or HF_TOKEN for private models."
            ) from exc

    def freeze(self) -> None:
        if self._model is not None:
            logger.info(f"Freezing {self.__class__.__name__}")
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False
        super().freeze()

    def unfreeze(self) -> None:
        if self._model is not None:
            logger.info(f"Unfreezing {self.__class__.__name__}")
            self._model.train()
            for param in self._model.parameters():
                param.requires_grad = True
        super().unfreeze()

    def forward(self, **inputs: Any) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")


__all__ = [
    "HuggingFaceAPINode",
    "HuggingFaceLocalNode",
]
