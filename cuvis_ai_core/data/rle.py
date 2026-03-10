"""RLE (Run-Length Encoding) utilities for COCO mask formats."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def rle_list_to_mask(rle: list[int], height: int, width: int) -> np.ndarray:
    """Decode a plain list-based RLE to a boolean mask.

    Values alternate between background and foreground run lengths,
    starting with background.  Data is in column-major (Fortran) order
    matching the COCO convention.

    Parameters
    ----------
    rle : list[int]
        Run-length counts alternating [bg, fg, bg, fg, ...].
    height, width : int
        Spatial dimensions matching COCO ``size`` field ``[H, W]``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(height, width)``.
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    value = 0
    for count in rle:
        mask[pos : pos + count] = value
        value = not value
        pos += count
    return mask.reshape((height, width), order="F").astype(bool, copy=False)


def coco_rle_encode(mask_np: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask to COCO compressed RLE dict.

    Parameters
    ----------
    mask_np : np.ndarray
        Binary mask of shape ``(H, W)``, dtype ``uint8``.

    Returns
    -------
    dict
        COCO RLE dict with ``"size"`` ``[H, W]`` and ``"counts"`` (str).
    """
    import pycocotools.mask as mask_util

    encoded = mask_util.encode(np.asfortranarray(mask_np))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    size = [int(v) for v in encoded["size"]]
    return {"size": size, "counts": counts}


def coco_rle_decode(rle: dict[str, Any]) -> np.ndarray:
    """Decode a COCO RLE dict to a binary mask.

    Handles both compressed (str/bytes) and uncompressed (list) counts.

    Parameters
    ----------
    rle : dict
        COCO RLE dict with ``"size"`` and ``"counts"`` keys.

    Returns
    -------
    np.ndarray
        Binary mask of shape ``(H, W)``, dtype ``uint8``.
    """
    import pycocotools.mask as mask_util

    counts = rle["counts"]
    if isinstance(counts, list):
        # Uncompressed RLE — convert via frPyObjects
        h, w = rle["size"]
        rle = mask_util.frPyObjects(rle, h, w)
    elif isinstance(counts, str):
        rle = {"size": rle["size"], "counts": counts.encode("utf-8")}
    return mask_util.decode(rle).astype(np.uint8)


def coco_rle_area(rle: dict[str, Any]) -> int:
    """Compute the area (foreground pixel count) from a COCO RLE dict."""
    import pycocotools.mask as mask_util

    counts = rle["counts"]
    if isinstance(counts, list):
        h, w = rle["size"]
        rle = mask_util.frPyObjects(rle, h, w)
    elif isinstance(counts, str):
        rle = {"size": rle["size"], "counts": counts.encode("utf-8")}
    return int(mask_util.area(rle))


def coco_rle_to_bbox(rle: dict[str, Any]) -> list[float]:
    """Compute ``[x, y, w, h]`` bounding box from a COCO RLE dict."""
    import pycocotools.mask as mask_util

    counts = rle["counts"]
    if isinstance(counts, list):
        h, w = rle["size"]
        rle = mask_util.frPyObjects(rle, h, w)
    elif isinstance(counts, str):
        rle = {"size": rle["size"], "counts": counts.encode("utf-8")}
    return mask_util.toBbox(rle).tolist()


def RLE2mask(rle: list[int], mask_width: int, mask_height: int) -> np.ndarray:
    """Deprecated: use :func:`rle_list_to_mask` instead.

    .. deprecated::
        ``RLE2mask`` will be removed in the next release.
        Note: the old signature used ``(rle, mask_width, mask_height)``
        which was misleading — callers typically passed COCO ``size[0]``
        (height) as ``mask_width``.  The replacement uses explicit
        ``(rle, height, width)`` order.
    """
    warnings.warn(
        "RLE2mask is deprecated; use rle_list_to_mask(rle, height, width) instead. "
        "Note the (height, width) argument order matches COCO size [H, W].",
        DeprecationWarning,
        stacklevel=2,
    )
    # Preserve old (buggy) behavior: mask_width was actually height in callers
    return rle_list_to_mask(rle, height=mask_width, width=mask_height)
