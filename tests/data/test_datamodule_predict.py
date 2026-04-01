"""Tests for SingleCu3sDataModule predict-mode support."""

from __future__ import annotations

import pytest

from cuvis_ai_core.data.datasets import SingleCu3sDataModule


def test_datamodule_allows_explicit_cu3s_without_annotation(
    mock_cuvis_sdk, tmp_path
) -> None:
    del mock_cuvis_sdk
    cu3s_path = tmp_path / "test.cu3s"
    cu3s_path.touch()

    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        annotation_json_path=None,
        processing_mode="Raw",
        batch_size=1,
    )

    assert datamodule.annotation_json_path is None


def test_predict_setup_and_dataloader_support_predict_ids(
    mock_cuvis_sdk, tmp_path
) -> None:
    del mock_cuvis_sdk
    cu3s_path = tmp_path / "test.cu3s"
    cu3s_path.touch()

    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode="Raw",
        batch_size=2,
        predict_ids=[0, 2, 4],
    )

    datamodule.setup(stage="predict")

    assert datamodule.predict_ds is not None
    assert len(datamodule.predict_ds) == 3

    batch = next(iter(datamodule.predict_dataloader()))
    assert "cube" in batch
    assert "wavelengths" in batch
    assert int(batch["cube"].shape[0]) <= 2


def test_predict_setup_without_predict_ids_uses_all_measurements(
    mock_cuvis_sdk, tmp_path
) -> None:
    del mock_cuvis_sdk
    cu3s_path = tmp_path / "test.cu3s"
    cu3s_path.touch()

    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode="Raw",
        batch_size=2,
    )

    datamodule.setup(stage="predict")

    assert datamodule.predict_ds is not None
    assert datamodule.predict_ids is None
    assert len(datamodule.predict_ds) == 7
    assert datamodule.predict_ds.measurement_indices == list(range(7))


def test_predict_dataloader_requires_predict_setup(mock_cuvis_sdk, tmp_path) -> None:
    del mock_cuvis_sdk
    cu3s_path = tmp_path / "test.cu3s"
    cu3s_path.touch()

    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode="Raw",
        batch_size=1,
    )

    with pytest.raises(RuntimeError, match=r"setup\('predict'\)"):
        datamodule.predict_dataloader()
