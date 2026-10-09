"""Boundary padding and sliding-window inference utilities (pure torch).

Mechanics mirror nnU-Net v2's verified inference pipeline:

- inputs are padded with constant zeros (``pad_nd_image(..., 'constant', 0)``),
  centered, and the padding is reverted on the output;
- tile offsets follow ``compute_steps_for_sliding_window``: first tile at 0,
  last tile flush with the far edge, remaining tiles spread evenly;
- overlapping tile logits are blended with a Gaussian importance map
  (``sigma = tile * 1/8``, scaled to max 10, clamped away from zero) and
  accumulated in float32.

Everything here is layout-native NCHW ``[B, C, H, W]``; the BHWC port contract
lives in the node wrapper. The helpers came from the cuvis-ai-unet plugin, whose
``DynUNet`` tiles its input with them; nodes that tile a frame (for example an
image tiler and its stitcher) build on the same offsets and weights, so there is
one tiling rule across the node library and the plugins.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def _pad_hw_centered(
    x: Tensor, target_h: int, target_w: int
) -> tuple[Tensor, tuple[slice, ...]]:
    """Center-pad the last two dims of ``x`` with zeros to ``(target_h, target_w)``.

    Returns the padded tensor and the slices that revert an output of the padded
    size back to the original H, W (leading dims untouched).
    """
    H, W = x.shape[-2], x.shape[-1]
    dh, dw = target_h - H, target_w - W
    top, left = dh // 2, dw // 2
    if dh or dw:
        x = F.pad(
            x, (left, dw - left, top, dh - top)
        )  # last dim first: (w_l, w_r, h_l, h_r)
    revert = (Ellipsis, slice(top, top + H), slice(left, left + W))
    return x, revert


def pad_hw_to_multiple(
    x: Tensor, grid_hw: tuple[int, int]
) -> tuple[Tensor, tuple[slice, ...]]:
    """Center-pad H, W with zeros to the next multiple of ``grid_hw``."""
    gh, gw = grid_hw
    H, W = x.shape[-2], x.shape[-1]
    return _pad_hw_centered(x, H + (-H) % gh, W + (-W) % gw)


def pad_hw_to_min(
    x: Tensor, min_hw: tuple[int, int]
) -> tuple[Tensor, tuple[slice, ...]]:
    """Center-pad H, W with zeros so each is at least ``min_hw``."""
    H, W = x.shape[-2], x.shape[-1]
    return _pad_hw_centered(x, max(H, min_hw[0]), max(W, min_hw[1]))


def compute_tile_offsets(size: int, tile: int, overlap: float) -> list[int]:
    """Tile start offsets along one axis (nnU-Net ``compute_steps_for_sliding_window``).

    First offset is 0 and the last is flush at ``size - tile``; intermediate
    offsets are spread evenly. ``overlap`` is the fraction of a tile shared by
    neighbours (0.5 -> step of half a tile). Requires ``size >= tile``.
    """
    if size < tile:
        raise ValueError(f"size ({size}) must be >= tile ({tile}); pad the input first")
    if size == tile:
        return [0]
    target_step = tile * (1.0 - overlap)
    num_steps = math.ceil((size - tile) / target_step) + 1
    actual_step = (size - tile) / (num_steps - 1)
    return [int(round(i * actual_step)) for i in range(num_steps)]


def gaussian_weight_map(
    tile_hw: tuple[int, int],
    device: torch.device,
    sigma_scale: float = 0.125,
    value_scaling: float = 10.0,
) -> Tensor:
    """Gaussian importance map ``[1, 1, th, tw]`` (float32) for tile blending.

    nnU-Net's recipe: per-axis Gaussian centered on the tile with
    ``sigma = tile * sigma_scale``, normalized to max 1, scaled by
    ``value_scaling``, and clamped away from zero so the final division can
    never see a zero weight.
    """
    weights_1d = []
    for size in tile_hw:
        coords = torch.arange(size, dtype=torch.float32, device=device)
        center = (size - 1) / 2.0
        sigma = max(size * sigma_scale, 1e-8)
        weights_1d.append(torch.exp(-((coords - center) ** 2) / (2.0 * sigma**2)))
    w = torch.outer(weights_1d[0], weights_1d[1])
    w = w / w.max() * value_scaling
    return w.clamp_min(torch.finfo(torch.float32).tiny).reshape(1, 1, *tile_hw)


def sliding_window_inference(
    net: Callable[[Tensor], Tensor],
    x: Tensor,
    tile_hw: tuple[int, int],
    overlap: float = 0.5,
    gaussian: bool = True,
    weight_map: Tensor | None = None,
    amp: bool = True,
    tile_batch: int = 1,
) -> Tensor:
    """Tiled forward over NCHW ``x`` with Gaussian-blended overlaps.

    Pads ``x`` (centered, constant 0) so each axis is at least one tile, runs
    ``net`` on every tile, accumulates ``logits * weight`` and the weights in
    float32, divides, and crops the padding back off. Runs under ``no_grad``:
    this is an inference path, not a training path.

    ``weight_map`` lets the caller pass a cached Gaussian (``[1, 1, th, tw]``);
    when ``None`` and ``gaussian`` is set, one is built on the fly.

    ``amp`` runs each tile's forward under ``torch.autocast`` (fp16) when ``x`` is
    on CUDA, matching nnU-Net's inference path; it is a no-op on CPU. Logits are
    cast back to float32 for the accumulation regardless.

    ``tile_batch`` is how many tiles are stacked into a single ``net`` call. The
    default ``1`` reproduces the original one-tile-per-forward behaviour exactly.
    A 128px tile barely occupies a modern GPU, so processing tiles one at a time
    is launch/underfill bound; batching packs several tiles onto the batch axis so
    each kernel does more work. It changes utilisation, not the result: the tiles,
    weights and accumulation are identical, so outputs match the serial path to
    floating-point tolerance. Higher values trade GPU memory for speed.
    """
    th, tw = tile_hw
    with torch.no_grad():
        x_p, revert = pad_hw_to_min(x, tile_hw)
        Hp, Wp = x_p.shape[-2], x_p.shape[-1]

        if gaussian:
            w = (
                weight_map
                if weight_map is not None
                else gaussian_weight_map(tile_hw, x_p.device)
            )
            w = w.to(device=x_p.device, dtype=torch.float32)
        else:
            w = torch.ones(1, 1, th, tw, dtype=torch.float32, device=x_p.device)

        offsets = [
            (oy, ox)
            for oy in compute_tile_offsets(Hp, th, overlap)
            for ox in compute_tile_offsets(Wp, tw, overlap)
        ]
        b = x_p.shape[0]
        acc: Tensor | None = None
        wsum = torch.zeros(1, 1, Hp, Wp, dtype=torch.float32, device=x_p.device)
        use_amp = amp and x_p.is_cuda
        step = max(1, int(tile_batch))
        for i in range(0, len(offsets), step):
            chunk = offsets[i : i + step]
            # Stack this chunk's tiles onto the batch axis: [len(chunk) * b, C, th, tw].
            tiles = torch.cat(
                [x_p[..., oy : oy + th, ox : ox + tw] for oy, ox in chunk], dim=0
            )
            with torch.autocast(device_type=x_p.device.type, enabled=use_amp):
                logits = net(tiles)
            logits = logits.float()
            if acc is None:
                acc = torch.zeros(
                    b, logits.shape[1], Hp, Wp, dtype=torch.float32, device=x_p.device
                )
            # Rows [j*b : (j+1)*b] are tile j; scatter each back to its window.
            for j, (oy, ox) in enumerate(chunk):
                acc[..., oy : oy + th, ox : ox + tw] += logits[j * b : (j + 1) * b] * w
                wsum[..., oy : oy + th, ox : ox + tw] += w
        assert acc is not None  # at least one tile always runs
        return (acc / wsum)[revert]
