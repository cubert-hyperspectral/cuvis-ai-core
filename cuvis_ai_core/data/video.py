"""Video utilities: frame iteration, dataset, and Lightning DataModule."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def _import_torchcodec() -> type:
    """Lazy import for torchcodec (requires FFmpeg native libraries at runtime)."""
    try:
        return importlib.import_module("torchcodec.decoders").SimpleVideoDecoder
    except (ImportError, OSError) as exc:
        msg = (
            "torchcodec requires FFmpeg system libraries which are not available.\n"
            "Install FFmpeg for your operating system:\n"
            "  - Linux (apt):   sudo apt install ffmpeg\n"
            "  - Linux (conda): conda install -c conda-forge ffmpeg\n"
            "  - macOS:         brew install ffmpeg\n"
            "  - Windows:       download from https://ffmpeg.org/download.html and add to PATH"
        )
        raise ImportError(msg) from exc


# ---------------------------------------------------------------------------
# VideoIterator — frame-level access to MP4/AVI files via torchcodec
# ---------------------------------------------------------------------------
class VideoIterator:
    """Iterate over frames of an MP4/AVI video.

    Tries torchcodec (random-access, GPU-friendly) first; falls back to
    OpenCV ``cv2.VideoCapture`` when torchcodec/FFmpeg is unavailable.
    """

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        assert Path(source_path).exists(), f"Video file {source_path} does not exist"
        self.basename: str = Path(source_path).stem
        self._backend: str = "torchcodec"

        try:
            _SimpleVideoDecoder = _import_torchcodec()
            self.video_decoder = _SimpleVideoDecoder(source_path)
            self.num_frames: int = len(self.video_decoder)
            self.enable_random_access = True
            if self.num_frames < 0:
                logger.error(
                    "Cannot determine number of frames. Random access is disabled: {}",
                    source_path,
                )
                self.enable_random_access = False
                self.num_frames = 0
            self.frame_rate: float = self.video_decoder.metadata.average_fps
            first_frame = self.video_decoder[0]
            self.image_width: int = first_frame.shape[2]
            self.image_height: int = first_frame.shape[1]
        except ImportError as err:
            logger.info("torchcodec unavailable, falling back to cv2.VideoCapture")
            self._backend = "cv2"
            self.video_decoder = None  # type: ignore[assignment]
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                raise RuntimeError(
                    f"cv2.VideoCapture failed to open {source_path}"
                ) from err
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
            self.image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.enable_random_access = True
            cap.release()

    def __len__(self) -> int:
        return self.num_frames

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for frame_id in range(len(self)):
            yield self.get_frame(frame_id)

    def get_frame(self, frame_id: int) -> dict[str, Any]:
        if self._backend == "cv2":
            return self._get_frame_cv2(frame_id)

        try:
            frame = self.video_decoder[frame_id].permute(1, 2, 0).numpy()
        except Exception as e:
            logger.error("Error reading frame {}: {}", frame_id, e)
            return {"frame_id": frame_id, "image": np.zeros((1, 1, 3), dtype=np.uint8)}

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return {"frame_id": frame_id, "image": frame_bgr, "basename": self.basename}

    def _get_frame_cv2(self, frame_id: int) -> dict[str, Any]:
        cap = cv2.VideoCapture(self.source_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            logger.error("cv2: Error reading frame {}", frame_id)
            return {"frame_id": frame_id, "image": np.zeros((1, 1, 3), dtype=np.uint8)}
        return {"frame_id": frame_id, "image": frame_bgr, "basename": self.basename}


# ---------------------------------------------------------------------------
# VideoFrameDataset — torch Dataset wrapping VideoIterator
# ---------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    """Torch map-style dataset that yields RGB float32 tensors from a video."""

    def __init__(self, video_iter: VideoIterator, end_frame: int = -1) -> None:
        self.video_iter = video_iter
        total = len(video_iter)
        self.n_frames = min(end_frame, total) if end_frame > 0 else total
        self.fps: float = video_iter.frame_rate

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame_data = self.video_iter.get_frame(idx)
        bgr = frame_data["image"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_f32 = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return {
            "rgb_image": rgb_f32,
            "frame_id": torch.tensor(idx, dtype=torch.int64),
        }


# ---------------------------------------------------------------------------
# VideoFrameDataModule — LightningDataModule for the Predictor
# ---------------------------------------------------------------------------
class VideoFrameDataModule(pl.LightningDataModule):
    """DataModule that reads MP4 frames for use with cuvis-ai Predictor."""

    def __init__(
        self, video_path: str, end_frame: int = -1, batch_size: int = 1
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.end_frame = end_frame
        self.batch_size = batch_size
        self.predict_ds: VideoFrameDataset | None = None
        self.fps: float = 10.0

    def setup(self, stage: str | None = None) -> None:
        if stage == "predict" or stage is None:
            video_iter = VideoIterator(self.video_path)
            self.predict_ds = VideoFrameDataset(video_iter, end_frame=self.end_frame)
            self.fps = video_iter.frame_rate
            if self.fps <= 0:
                self.fps = 10.0

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise RuntimeError(
                "Predict dataset not initialized. Call setup('predict') first."
            )
        return DataLoader(
            self.predict_ds, shuffle=False, batch_size=self.batch_size, num_workers=0
        )


__all__ = [
    "VideoFrameDataModule",
    "VideoFrameDataset",
    "VideoIterator",
]
