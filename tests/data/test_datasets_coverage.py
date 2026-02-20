"""Tests for SingleCu3sDataset edge cases to close codecov gaps.

Covers: FPS extraction, processing mode fallback, SpectralRadiance validation.
"""

from unittest.mock import Mock, PropertyMock

import pytest

from cuvis_ai_core.data.datasets import SingleCu3sDataset


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
