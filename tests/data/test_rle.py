"""Tests for cuvis_ai_core.data.rle utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cuvis_ai_core.data.rle import (
    RLE2mask,
    coco_rle_area,
    coco_rle_decode,
    coco_rle_encode,
    coco_rle_to_bbox,
    decode_rle_mask_for_canvas,
    rle_list_to_mask,
)


class TestRleListToMask:
    """Plain list-based RLE decoding."""

    def test_basic(self):
        rle = [5, 2, 3, 10, 80]
        mask = rle_list_to_mask(rle, height=10, width=10)
        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        assert mask.sum() == 12

    def test_all_zeros(self):
        mask = rle_list_to_mask([100], height=10, width=10)
        assert mask.sum() == 0

    def test_all_ones(self):
        mask = rle_list_to_mask([0, 100], height=10, width=10)
        assert mask.sum() == 100

    def test_non_square(self):
        """Non-square mask: 3 rows × 5 cols, first column all foreground."""
        # Fortran order: column-major.  3 rows per column, 5 columns.
        # First column (3 pixels) foreground, rest (12 pixels) background.
        rle = [0, 3, 12]
        mask = rle_list_to_mask(rle, height=3, width=5)
        assert mask.shape == (3, 5)
        # First column should be all True
        np.testing.assert_array_equal(mask[:, 0], [True, True, True])
        # Rest should be False
        assert mask[:, 1:].sum() == 0

    def test_empty(self):
        mask = rle_list_to_mask([], height=2, width=3)
        assert mask.shape == (2, 3)
        assert mask.sum() == 0


class TestCocoRleEncodeDecode:
    """COCO compressed RLE encode/decode roundtrip."""

    def test_roundtrip_square(self):
        original = np.zeros((10, 10), dtype=np.uint8)
        original[2:5, 3:7] = 1
        rle = coco_rle_encode(original)
        decoded = coco_rle_decode(rle)
        np.testing.assert_array_equal(original, decoded)

    def test_roundtrip_non_square(self):
        original = np.zeros((3, 7), dtype=np.uint8)
        original[0, :] = 1
        rle = coco_rle_encode(original)
        assert rle["size"] == [3, 7]
        decoded = coco_rle_decode(rle)
        assert decoded.shape == (3, 7)
        np.testing.assert_array_equal(original, decoded)

    def test_empty_mask(self):
        original = np.zeros((5, 5), dtype=np.uint8)
        rle = coco_rle_encode(original)
        decoded = coco_rle_decode(rle)
        np.testing.assert_array_equal(original, decoded)

    def test_full_mask(self):
        original = np.ones((4, 6), dtype=np.uint8)
        rle = coco_rle_encode(original)
        decoded = coco_rle_decode(rle)
        np.testing.assert_array_equal(original, decoded)

    def test_decode_str_counts(self):
        """Counts as str (compressed, JSON-serialized form)."""
        original = np.zeros((5, 5), dtype=np.uint8)
        original[1:3, 1:3] = 1
        rle = coco_rle_encode(original)
        assert isinstance(rle["counts"], str)
        decoded = coco_rle_decode(rle)
        np.testing.assert_array_equal(original, decoded)

    def test_decode_bytes_counts(self):
        """Counts as bytes (raw pycocotools form)."""
        original = np.zeros((5, 5), dtype=np.uint8)
        original[1:3, 1:3] = 1
        rle = coco_rle_encode(original)
        rle_bytes = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        decoded = coco_rle_decode(rle_bytes)
        np.testing.assert_array_equal(original, decoded)

    def test_decode_uncompressed_list_counts(self):
        """Counts as list (uncompressed RLE)."""
        import pycocotools.mask as mask_util

        original = np.zeros((5, 5), dtype=np.uint8)
        original[2, 2] = 1
        # Get the uncompressed form
        encoded = mask_util.encode(np.asfortranarray(original))
        mask_util.decode(encoded)
        # Build uncompressed RLE manually
        flat = original.flatten(order="F")
        counts = []
        current = flat[0]
        run = 0
        for val in flat:
            if val == current:
                run += 1
            else:
                counts.append(run)
                current = val
                run = 1
        counts.append(run)
        if flat[0] != 0:
            counts.insert(0, 0)
        uncompressed_rle = {"size": [5, 5], "counts": counts}
        decoded = coco_rle_decode(uncompressed_rle)
        np.testing.assert_array_equal(original, decoded)


class TestCocoRleArea:
    def test_known_area(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1  # 3 rows × 4 cols = 12 pixels
        rle = coco_rle_encode(mask)
        assert coco_rle_area(rle) == 12

    def test_empty_area(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        rle = coco_rle_encode(mask)
        assert coco_rle_area(rle) == 0


class TestCocoRleToBbox:
    def test_known_bbox(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        rle = coco_rle_encode(mask)
        bbox = coco_rle_to_bbox(rle)
        # bbox is [x, y, w, h] = [3, 2, 4, 3]
        assert bbox == pytest.approx([3.0, 2.0, 4.0, 3.0])

    def test_empty_bbox(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        rle = coco_rle_encode(mask)
        bbox = coco_rle_to_bbox(rle)
        assert bbox == pytest.approx([0.0, 0.0, 0.0, 0.0])

    def test_area_and_bbox_from_uncompressed_counts(self):
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:5] = 1

        flat = mask.flatten(order="F")
        counts: list[int] = []
        current = 0
        run = 0
        for value in flat:
            if int(value) == current:
                run += 1
                continue
            counts.append(run)
            current = int(value)
            run = 1
        counts.append(run)

        rle = {"size": [4, 5], "counts": counts}
        assert coco_rle_area(rle) == 6
        assert coco_rle_to_bbox(rle) == pytest.approx([2.0, 1.0, 3.0, 2.0])


class TestDecodeRleMaskForCanvas:
    def test_rejects_non_list_counts(self):
        with pytest.raises(TypeError, match="list-based RLE"):
            decode_rle_mask_for_canvas(
                {"size": [2, 2], "counts": "compressed"},
                target_height=2,
                target_width=2,
            )

    def test_warns_for_missing_or_invalid_size(self, monkeypatch, caplog):
        import cuvis_ai_core.data.rle as rle_mod

        monkeypatch.setattr(rle_mod, "_RLE_SIZE_MISMATCH_WARNING_EMITTED", False)
        caplog.set_level("WARNING", logger="cuvis_ai_core.data.rle")

        mask = decode_rle_mask_for_canvas(
            {"counts": [0, 4], "size": "bad-size"},
            target_height=2,
            target_width=2,
        )

        assert mask.shape == (2, 2)
        assert mask.sum() == 4
        assert "<missing or invalid>" in caplog.text

    def test_warns_for_non_numeric_size_entries(self, monkeypatch, caplog):
        import cuvis_ai_core.data.rle as rle_mod

        monkeypatch.setattr(rle_mod, "_RLE_SIZE_MISMATCH_WARNING_EMITTED", False)
        caplog.set_level("WARNING", logger="cuvis_ai_core.data.rle")

        mask = decode_rle_mask_for_canvas(
            {"counts": [0, 1, 3], "size": ["bad", 2]},
            target_height=2,
            target_width=2,
        )

        assert mask.shape == (2, 2)
        assert mask.sum() == 1
        assert "<missing or invalid>" in caplog.text


class TestDeprecatedAlias:
    def test_rle2mask_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RLE2mask([5, 2, 3], mask_width=5, mask_height=2)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "rle_list_to_mask" in str(w[0].message)
