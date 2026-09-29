"""Data utilities for CUVIS.AI.

Heavy imports (cv2, torch) are deferred so that lightweight modules like
``public_datasets`` can be imported without pulling in the full stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuvis_ai_core.data.model_weights import (
        TRAINED_PIPELINES,
        ModelDownloadError,
        ModelRegistryConflict,
        ModelStatus,
        ModelWeights,
        ModelWeightsMissingError,
        RegisteredWeight,
    )
    from cuvis_ai_core.data.public_datasets import PublicDatasets
    from cuvis_ai_core.data.rle import (
        coco_rle_area,
        coco_rle_decode,
        coco_rle_encode,
        coco_rle_to_bbox,
        rle_list_to_mask,
    )
    from cuvis_ai_core.data.video import (
        VideoFrameDataModule,
        VideoFrameDataset,
        VideoIterator,
    )

__all__ = [
    "ModelDownloadError",
    "ModelRegistryConflict",
    "ModelStatus",
    "ModelWeights",
    "ModelWeightsMissingError",
    "PublicDatasets",
    "RegisteredWeight",
    "TRAINED_PIPELINES",
    "coco_rle_area",
    "coco_rle_decode",
    "coco_rle_encode",
    "coco_rle_to_bbox",
    "rle_list_to_mask",
    "VideoFrameDataModule",
    "VideoFrameDataset",
    "VideoIterator",
]

_SUBMODULE_MAP: dict[str, str] = {
    "ModelDownloadError": "cuvis_ai_core.data.model_weights",
    "ModelRegistryConflict": "cuvis_ai_core.data.model_weights",
    "ModelStatus": "cuvis_ai_core.data.model_weights",
    "ModelWeights": "cuvis_ai_core.data.model_weights",
    "ModelWeightsMissingError": "cuvis_ai_core.data.model_weights",
    "RegisteredWeight": "cuvis_ai_core.data.model_weights",
    "TRAINED_PIPELINES": "cuvis_ai_core.data.model_weights",
    "PublicDatasets": "cuvis_ai_core.data.public_datasets",
    "coco_rle_area": "cuvis_ai_core.data.rle",
    "coco_rle_decode": "cuvis_ai_core.data.rle",
    "coco_rle_encode": "cuvis_ai_core.data.rle",
    "coco_rle_to_bbox": "cuvis_ai_core.data.rle",
    "rle_list_to_mask": "cuvis_ai_core.data.rle",
    "VideoFrameDataModule": "cuvis_ai_core.data.video",
    "VideoFrameDataset": "cuvis_ai_core.data.video",
    "VideoIterator": "cuvis_ai_core.data.video",
}


def __getattr__(name: str):
    """Import the submodule that defines ``name`` on first access."""
    if name in _SUBMODULE_MAP:
        return getattr(importlib.import_module(_SUBMODULE_MAP[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
