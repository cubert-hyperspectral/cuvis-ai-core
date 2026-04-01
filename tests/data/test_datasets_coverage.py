"""Tests for SingleCu3sDataset edge cases to close codecov gaps.

Covers: FPS extraction, processing mode fallback, SpectralRadiance validation.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import numpy as np
import pytest

import cuvis_ai_core.data.rle as rle_utils
from cuvis_ai_core.data.coco_labels import Annotation
from cuvis_ai_core.data.datasets import SingleCu3sDataset, create_mask


class TestDatasetFpsExtraction:
    """Cover FPS try/except in __init__."""

    def test_fps_available(self, mock_cuvis_sdk, tmp_path):
        mock_cuvis_sdk["session"].fps = 30.0
        cu3s = tmp_path / "test.cu3s"
        cu3s.touch()

        ds = SingleCu3sDataset(str(cu3s), processing_mode="Raw")
        assert ds.fps == 30.0

    def test_fps_unavailable(self, mock_cuvis_sdk, tmp_path):
        type(mock_cuvis_sdk["session"]).fps = PropertyMock(
            side_effect=AttributeError("no fps")
        )
        cu3s = tmp_path / "test.cu3s"
        cu3s.touch()

        ds = SingleCu3sDataset(str(cu3s), processing_mode="Raw")
        assert ds.fps is None


class TestDatasetProcessingModeFallback:
    """Cover getattr for string processing mode (lines 62-63)."""

    def test_string_mode_resolved_via_getattr(self, mock_cuvis_sdk, tmp_path):
        """Valid string mode is resolved via getattr on ProcessingMode."""
        mock_cuvis_sdk["session"].fps = 30.0
        cu3s = tmp_path / "test.cu3s"
        cu3s.touch()

        SingleCu3sDataset(str(cu3s), processing_mode="Raw")
        # ProcessingMode.Raw was resolved through getattr and assigned
        assert mock_cuvis_sdk["processing_context"].processing_mode == "Raw"


class TestDatasetSpectralRadiance:
    """Cover SpectralRadiance mode validation."""

    def test_spectral_radiance_with_dark_ref_succeeds(self, mock_cuvis_sdk, tmp_path):
        mock_cuvis_sdk["session"].fps = 30.0
        cu3s = tmp_path / "test.cu3s"
        cu3s.touch()

        # Dark ref is available via the mock_cuvis_sdk fixture
        ds = SingleCu3sDataset(str(cu3s), processing_mode="SpectralRadiance")
        assert ds is not None

    def test_spectral_radiance_without_dark_ref_fails(self, mock_cuvis_sdk, tmp_path):
        mock_cuvis_sdk["session"].fps = 30.0

        def _no_dark_ref(idx, ref_type):
            if ref_type == "Dark":
                return None
            return Mock()

        mock_cuvis_sdk["session"].get_reference = Mock(side_effect=_no_dark_ref)

        cu3s = tmp_path / "test.cu3s"
        cu3s.touch()

        with pytest.raises(AssertionError, match="Dark reference"):
            SingleCu3sDataset(str(cu3s), processing_mode="SpectralRadiance")


def test_dataset_uses_cube_shape_when_coco_image_metadata_is_unavailable(
    mock_cuvis_sdk, tmp_path
):
    del mock_cuvis_sdk
    cu3s = tmp_path / "mock.cu3s"
    cu3s.touch()
    annotation_json = tmp_path / "mock.json"
    annotation_json.write_text("{}", encoding="utf-8")

    ds = SingleCu3sDataset(
        str(cu3s),
        annotation_json_path=str(annotation_json),
        processing_mode="Raw",
    )

    sample = ds[0]

    assert sample["mask"].shape == sample["cube"].shape[:2]
    assert np.count_nonzero(sample["mask"]) == 0


def test_resolve_annotation_canvas_size_uses_cube_shape_without_coco(
    mock_cuvis_sdk, tmp_path
):
    del mock_cuvis_sdk
    cu3s = tmp_path / "plain.cu3s"
    cu3s.touch()

    ds = SingleCu3sDataset(str(cu3s), processing_mode="Raw")
    ds._coco = None

    assert ds._resolve_annotation_canvas_size(image_id=1, cube_shape=(3, 5)) == (
        3,
        5,
    )


def test_resolve_annotation_canvas_size_uses_matching_image_metadata(
    mock_cuvis_sdk, tmp_path
):
    del mock_cuvis_sdk
    cu3s = tmp_path / "meta.cu3s"
    cu3s.touch()

    ds = SingleCu3sDataset(str(cu3s), processing_mode="Raw")
    ds._coco = SimpleNamespace(images=[SimpleNamespace(id=7, height=11, width=13)])

    assert ds._resolve_annotation_canvas_size(image_id=7, cube_shape=(3, 5)) == (
        11,
        13,
    )


def test_resolve_annotation_canvas_size_falls_back_to_backend_lookup(
    mock_cuvis_sdk, tmp_path
):
    del mock_cuvis_sdk
    cu3s = tmp_path / "backend.cu3s"
    cu3s.touch()

    ds = SingleCu3sDataset(str(cu3s), processing_mode="Raw")
    ds._coco = SimpleNamespace(
        images=[
            SimpleNamespace(id=1, height=8, width=8),
            SimpleNamespace(id=9, height="bad", width=13),
        ],
        _coco=SimpleNamespace(imgs={9: {"height": 17, "width": 19}}),
    )

    assert ds._resolve_annotation_canvas_size(image_id=9, cube_shape=(3, 5)) == (
        17,
        19,
    )


def test_resolve_annotation_canvas_size_falls_back_to_cube_shape_on_invalid_backend_data(
    mock_cuvis_sdk, tmp_path
):
    del mock_cuvis_sdk
    cu3s = tmp_path / "backend-invalid.cu3s"
    cu3s.touch()

    ds = SingleCu3sDataset(str(cu3s), processing_mode="Raw")
    ds._coco = SimpleNamespace(
        images=[SimpleNamespace(id=9, height="bad", width=13)],
        _coco=SimpleNamespace(imgs={9: {"height": "still-bad"}}),
    )

    assert ds._resolve_annotation_canvas_size(image_id=9, cube_shape=(3, 5)) == (
        3,
        5,
    )


def _encode_uncompressed_rle(mask: np.ndarray) -> list[int]:
    """Encode binary mask to uncompressed COCO RLE counts (column-major)."""
    flat = mask.astype(np.uint8).flatten(order="F")
    if flat.size == 0:
        return []

    counts: list[int] = []
    current = 0
    run_length = 0

    for value in flat:
        if int(value) == current:
            run_length += 1
            continue
        counts.append(run_length)
        current = int(value)
        run_length = 1

    counts.append(run_length)
    return counts


class TestCreateMaskRleSizeTolerance:
    def test_create_mask_autocorrects_swapped_rle_size(self):
        binary = np.zeros((3, 5), dtype=np.uint8)
        binary[0, 0] = 1
        binary[1, 3] = 1
        binary[2, 4] = 1
        counts = _encode_uncompressed_rle(binary)

        ann = Annotation(
            id=1,
            image_id=1,
            category_id=7,
            mask={"size": [5, 3], "counts": counts},
        )

        out = create_mask([ann], image_height=3, image_width=5)

        expected = np.zeros((3, 5), dtype=np.int32)
        expected[binary.astype(bool)] = 7
        np.testing.assert_array_equal(out, expected)

    def test_create_mask_keeps_canonical_rle_size_behavior(self):
        binary = np.zeros((3, 5), dtype=np.uint8)
        binary[0, 1:3] = 1
        counts = _encode_uncompressed_rle(binary)

        ann = Annotation(
            id=1,
            image_id=1,
            category_id=4,
            mask={"size": [3, 5], "counts": counts},
        )

        out = create_mask([ann], image_height=3, image_width=5)

        expected = np.zeros((3, 5), dtype=np.int32)
        expected[binary.astype(bool)] = 4
        np.testing.assert_array_equal(out, expected)

    def test_create_mask_warns_once_for_mismatched_rle_size(self, caplog, monkeypatch):
        monkeypatch.setattr(rle_utils, "_RLE_SIZE_MISMATCH_WARNING_EMITTED", False)
        caplog.set_level(logging.WARNING, logger="cuvis_ai_core.data.rle")

        binary = np.zeros((2, 4), dtype=np.uint8)
        binary[:, 0] = 1
        counts = _encode_uncompressed_rle(binary)
        ann = Annotation(
            id=1,
            image_id=1,
            category_id=9,
            mask={"size": [4, 2], "counts": counts},
        )

        create_mask([ann], image_height=2, image_width=4)
        create_mask([ann], image_height=2, image_width=4)

        messages = [record.getMessage() for record in caplog.records]
        assert (
            len(
                [
                    msg
                    for msg in messages
                    if "COCO RLE size" in msg and "target canvas" in msg
                ]
            )
            == 1
        )

    def test_annotation_to_torchvision_autocorrects_swapped_rle_size(self):
        binary = np.zeros((3, 5), dtype=np.uint8)
        binary[1:, 2] = 1
        counts = _encode_uncompressed_rle(binary)
        ann = Annotation(
            id=1,
            image_id=1,
            category_id=5,
            mask={"size": [5, 3], "counts": counts},
        )

        result = ann.to_torchvision(size=(3, 5))
        out_mask = result["mask"].numpy()
        np.testing.assert_array_equal(out_mask, binary.astype(bool))
