from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import cuvis
import numpy as np
from loguru import logger
from skimage.draw import polygon2mask
from torch.utils.data import Dataset

from cuvis_ai_core.data.coco_labels import Annotation, COCOData, RLE2mask
from cuvis_ai_core.utils.general import _resolve_measurement_indices



import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SingleCu3sDataset(Dataset):
    """Load cube frames from .cu3s sessions with optional COCO-derived masks."""

    def __init__(
        self,
        cu3s_file_path: str,
        annotation_json_path: str | None = None,
        processing_mode: cuvis.ProcessingMode | str | None = "Raw",
        measurement_indices: Sequence[int] | Iterable[int] | None = None,
        normalize_to_unit: bool = False,
    ) -> None:
        self.cu3s_file_path = cu3s_file_path
        assert os.path.exists(cu3s_file_path), (
            f"Dataset path does not exist: {cu3s_file_path}"
        )
        assert Path(cu3s_file_path).suffix == ".cu3s", (
            f"Dataset path must point to a .cu3s file: {cu3s_file_path}"
        )

        self.session = cuvis.SessionFile(cu3s_file_path)
        self.pc = cuvis.ProcessingContext(self.session)

        has_white_ref = (
            self.session.get_reference(0, cuvis.ReferenceType.White) is not None
        )
        has_dark_ref = (
            self.session.get_reference(0, cuvis.ReferenceType.Dark) is not None
        )
        if processing_mode is not None:
            if isinstance(processing_mode, str):
                processing_mode = getattr(cuvis.ProcessingMode, processing_mode, "Raw")

            if processing_mode == cuvis.ProcessingMode.Reflectance:
                assert has_white_ref and has_dark_ref, (
                    "Reflectance processing mode requires both White and Dark references "
                    "in the cu3s file."
                )
            self.pc.processing_mode = processing_mode

        mesu0 = self.session.get_measurement(0)
        self.num_channels = mesu0.cube.channels
        self.wavelengths = np.array(mesu0.cube.wavelength).ravel()
        self._total_measurements = len(self.session)
        self.measurement_indices = _resolve_measurement_indices(
            measurement_indices, max_index=self._total_measurements
        )
        # Backwards compatibility for legacy callers.
        self.mes_ids = self.measurement_indices

        logger.info(
            f"Loaded cu3s dataset from {cu3s_file_path} with {len(self.measurement_indices)} "
            f"measurements: {self.measurement_indices}"
        )
        self.has_labels = (
            annotation_json_path is not None
            and Path(annotation_json_path).exists()
            or Path(cu3s_file_path).with_suffix(".json").exists()
        )
        if self.has_labels and annotation_json_path is None:
            # Sane fallback: label file is named the same as the Session File
            annotation_json_path = Path(cu3s_file_path).with_suffix(".json")
        self._coco: COCOData | None = None
        self.class_labels: dict[int, str] | None = None
        self.normalize_to_unit = normalize_to_unit
        if self.has_labels:
            try:
                self._coco = COCOData.from_path(annotation_json_path)
            except Exception as e:
                logger.warning(
                    f"Could not load annotation for {annotation_json_path}:", e
                )
                self.has_labels = False
            logger.info(f"Category map: {self._coco.category_id_to_name}")
            self.class_labels = self._coco.category_id_to_name

    def __len__(self) -> int:
        return len(self.measurement_indices)

    @property
    def wavelengths_nm(self) -> np.ndarray:
        mesu = self.session.get_measurement(0)  # starts the cound from 0
        wavelengths = np.array(mesu.cube.wavelength, dtype=np.int32).ravel()
        return wavelengths

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        mesu_index = self.measurement_indices[idx]
        mesu = self.session.get_measurement(mesu_index)  # starts the cound from 0
        if "cube" not in mesu.data:
            mesu = self.pc.apply(mesu)
        cube_array: np.ndarray = mesu.cube.array

        wavelengths = np.array(mesu.cube.wavelength, dtype=np.int32).ravel()

        out: dict[str, np.ndarray | int] = {
            "cube": cube_array,
            "mesu_index": mesu_index,
            "wavelengths": wavelengths,
        }

        if self.has_labels and self._coco is not None:
            # Check if we have a valid COCO image_id for this frame index
            if mesu_index in self._coco.image_ids:
                image_id = self._coco.image_ids[self._coco.image_ids.index(mesu_index)]
                anns = self._coco.annotations.where(image_id=image_id)
                category_mask = create_mask(
                    annotations=anns,
                    image_height=cube_array.shape[0],
                    image_width=cube_array.shape[1],
                )
            else:
                # Frame index not in available annotations
                category_mask = np.zeros(cube_array.shape[:2], dtype=np.int32)
            out["mask"] = category_mask

        return out

class SingleCu3sDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cu3s_file_path: str | None = None,
        annotation_json_path: str | None = None,
        data_dir: str | None = None,
        dataset_name: str | None = None,
        train_ids: list[int] | None = None,
        val_ids: list[int] | None = None,
        test_ids: list[int] | None = None,
        batch_size: int = 2,
        processing_mode: str = "Reflectance",
        normalize_to_unit: bool = False,
    ) -> None:
        """Initialize SingleCu3sDataModule.

        Two modes of operation:
        1. Explicit paths: Provide both cu3s_file_path AND annotation_json_path
        2. Auto-resolve: Provide both data_dir AND dataset_name

        Args:
            cu3s_file_path: Direct path to .cu3s file (takes precedence)
            annotation_json_path: Direct path to .json annotation file (takes precedence)
            data_dir: Directory containing dataset files
            dataset_name: Name of dataset (e.g., "Lentils")
            train_ids: List of measurement indices for training
            val_ids: List of measurement indices for validation
            test_ids: List of measurement indices for testing
            batch_size: Batch size for dataloaders
            processing_mode: Cuvis processing mode string ("Raw", "Reflectance")
            normalize_to_unit: If True, normalize cube per-channel to [0, 1].
                For band selection workflows, keep False to preserve spectral ratios.

        Raises:
            ValueError: If neither (cu3s_file_path, annotation_json_path) nor (data_dir, dataset_name) provided
        """
        super().__init__()

        # Priority 1: Explicit paths
        if cu3s_file_path and annotation_json_path:
            self.cu3s_file_path = Path(cu3s_file_path)
            self.annotation_json_path = Path(annotation_json_path)
        # Priority 2: Auto-resolve from data_dir + dataset_name
        elif data_dir and dataset_name:
            self.cu3s_file_path, self.annotation_json_path = _resolve_assets(
                Path(data_dir), dataset_name
            )
        else:
            raise ValueError(
                "Must provide either (cu3s_file_path AND annotation_json_path) OR (data_dir AND dataset_name)"
            )

        self.batch_size = batch_size
        self.train_ids = train_ids or []
        self.val_ids = val_ids or []
        self.test_ids = test_ids or []
        self.processing_mode = processing_mode
        self.normalize_to_unit = normalize_to_unit
        self.train_ds: SingleCu3sDataset | None = None
        self.val_ds: SingleCu3sDataset | None = None
        self.test_ds: SingleCu3sDataset | None = None

    def prepare_data(self) -> None:
        # Only download if using auto-resolve mode with Lentils dataset
        # Skip for explicit paths (gRPC clients provide their own data)
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.train_ids:
                self.train_ds = SingleCu3sDataset(
                    cu3s_file_path=str(self.cu3s_file_path),
                    annotation_json_path=str(self.annotation_json_path),
                    processing_mode=self.processing_mode,
                    measurement_indices=self.train_ids,
                    normalize_to_unit=self.normalize_to_unit,
                )
            else:
                self.train_ds = None

            if self.val_ids:
                self.val_ds = SingleCu3sDataset(
                    cu3s_file_path=str(self.cu3s_file_path),
                    annotation_json_path=str(self.annotation_json_path),
                    processing_mode=self.processing_mode,
                    measurement_indices=self.val_ids,
                    normalize_to_unit=self.normalize_to_unit,
                )
            else:
                self.val_ds = None

        if stage == "test" or stage is None:
            if not self.test_ids:
                raise ValueError("test_ids must be provided to build the test dataset.")
            self.test_ds = SingleCu3sDataset(
                cu3s_file_path=str(self.cu3s_file_path),
                annotation_json_path=str(self.annotation_json_path),
                processing_mode=self.processing_mode,
                measurement_indices=self.test_ids,
                normalize_to_unit=self.normalize_to_unit,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError(
                "Train dataset is not initialized. Call setup('fit') first."
            )
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Validation dataset is not initialized.")
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError(
                "Test dataset is not initialized. Call setup('test') first."
            )
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)

def create_mask(
    annotations: Iterable[Annotation],
    image_height: int,
    image_width: int,
    overlap_strategy: str = "overwrite",
) -> np.ndarray:
    category_mask = np.zeros((image_height, image_width), dtype=np.int32)
    for ann in annotations:
        segs = ann.segmentation
        mask = ann.mask
        cat_id = int(ann.category_id)
        if not segs and not mask:
            continue

        if (
            isinstance(segs, list)
            and len(segs) > 0
            and isinstance(segs[0], (list, tuple))
        ):
            for seg in segs:
                if len(seg) < 6:
                    continue
                xy = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                # polygon2mask expects coords in (row, col) format and returns a filled boolean mask
                poly_mask = polygon2mask(
                    (image_height, image_width), xy[:, [1, 0]]
                )  # Swap x,y to row,col
                if overlap_strategy == "overwrite":
                    category_mask[poly_mask] = cat_id
                else:
                    write_idx = poly_mask & (category_mask == 0)
                    category_mask[write_idx] = cat_id
        if isinstance(mask, dict) and len(mask.get("counts", lambda: [])) > 0:
            mask_width, mask_height = mask.get("size")
            # decode RLE mask
            decoded = RLE2mask(
                mask.get("counts"), mask_width=mask_width, mask_height=mask_height
            )

            if overlap_strategy == "overwrite":
                write_mask = decoded
            else:
                write_mask = decoded & (category_mask == 0)
            category_mask[write_mask] = cat_id

    return category_mask


def _first_available(path: Path, pattern: str) -> Path:
    """Return the first file matching pattern within path."""
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any files matching '{pattern}' in {path}"
        )
    return matches[0]


def _resolve_assets(root: Path, dataset_name: str) -> tuple[Path, Path]:
    """Locate cube and label files for given dataset name.

    Args:
        root: Root directory containing dataset files
        dataset_name: Name of the dataset (e.g., "Lentils")

    Returns:
        Tuple of (cu3s_file_path, annotation_json_path)
    """
    default_cube = root / f"{dataset_name}.cu3s"
    default_label = root / f"{dataset_name}.json"

    cu3s = (
        default_cube
        if default_cube.exists()
        else _first_available(root, f"{dataset_name}*.cu3s")
    )
    label = (
        default_label
        if default_label.exists()
        else _first_available(root, f"{dataset_name}*.json")
    )
    return cu3s, label
