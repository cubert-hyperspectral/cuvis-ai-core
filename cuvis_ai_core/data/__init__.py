"""Data utilities for CUVIS.AI.

Heavy imports (cv2, torch) are deferred so that lightweight modules like
``public_datasets`` can be imported without pulling in the full stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    "PublicDatasets",
    "coco_rle_area",
    "coco_rle_decode",
    "coco_rle_encode",
    "coco_rle_to_bbox",
    "rle_list_to_mask",
    "VideoFrameDataModule",
    "VideoFrameDataset",
    "VideoIterator",
]

_SUBMODULE_MAP: dict[str, tuple[str, str]] = {
    "PublicDatasets": ("cuvis_ai_core.data.public_datasets", "PublicDatasets"),
    "coco_rle_area": ("cuvis_ai_core.data.rle", "coco_rle_area"),
    "coco_rle_decode": ("cuvis_ai_core.data.rle", "coco_rle_decode"),
    "coco_rle_encode": ("cuvis_ai_core.data.rle", "coco_rle_encode"),
    "coco_rle_to_bbox": ("cuvis_ai_core.data.rle", "coco_rle_to_bbox"),
    "rle_list_to_mask": ("cuvis_ai_core.data.rle", "rle_list_to_mask"),
    "VideoFrameDataModule": ("cuvis_ai_core.data.video", "VideoFrameDataModule"),
    "VideoFrameDataset": ("cuvis_ai_core.data.video", "VideoFrameDataset"),
    "VideoIterator": ("cuvis_ai_core.data.video", "VideoIterator"),
}


def __getattr__(name: str):
    if name in _SUBMODULE_MAP:
        module_path, attr = _SUBMODULE_MAP[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
