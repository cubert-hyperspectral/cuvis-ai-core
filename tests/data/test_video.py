"""Tests for video frame iteration and datamodule helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def _install_fake_cv2() -> None:
    if "cv2" in sys.modules:
        return

    module = ModuleType("cv2")
    module.COLOR_RGB2BGR = 1
    module.COLOR_BGR2RGB = 2
    module.CAP_PROP_FRAME_COUNT = 3
    module.CAP_PROP_FPS = 4
    module.CAP_PROP_FRAME_WIDTH = 5
    module.CAP_PROP_FRAME_HEIGHT = 6
    module.CAP_PROP_POS_FRAMES = 7

    def _cvt_color(image: np.ndarray, _code: int) -> np.ndarray:
        return image[..., ::-1].copy()

    module.cvtColor = _cvt_color
    module.VideoCapture = lambda _path: None
    sys.modules["cv2"] = module


_install_fake_cv2()
video_mod = importlib.import_module("cuvis_ai_core.data.video")


class _DecoderWithFrames:
    def __init__(self, _source_path: str) -> None:
        self.metadata = SimpleNamespace(average_fps=12.5)
        self.frames = [
            torch.tensor(
                [
                    [[255, 0], [0, 255]],
                    [[0, 255], [255, 0]],
                    [[10, 20], [30, 40]],
                ],
                dtype=torch.int32,
            )
        ]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.frames[idx]


class _DecoderWithReadError(_DecoderWithFrames):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx == 1:
            raise RuntimeError("bad frame")
        return super().__getitem__(0)


class _FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        frame: np.ndarray | None = None,
        ok: bool = True,
        fps: float = 0.0,
        frame_count: int = 3,
        width: int = 4,
        height: int = 5,
    ) -> None:
        self._opened = opened
        self._frame = frame
        self._ok = ok
        self._fps = fps
        self._frame_count = frame_count
        self._width = width
        self._height = height
        self.released = False
        self.position: int | None = None

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        mapping = {
            video_mod.cv2.CAP_PROP_FRAME_COUNT: self._frame_count,
            video_mod.cv2.CAP_PROP_FPS: self._fps,
            video_mod.cv2.CAP_PROP_FRAME_WIDTH: self._width,
            video_mod.cv2.CAP_PROP_FRAME_HEIGHT: self._height,
        }
        return float(mapping[prop])

    def set(self, prop: int, value: int) -> None:
        assert prop == video_mod.cv2.CAP_PROP_POS_FRAMES
        self.position = value

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._ok, self._frame

    def release(self) -> None:
        self.released = True


class _FakeVideoIter:
    def __init__(self, frame_rate: float = 24.0) -> None:
        self.frame_rate = frame_rate

    def __len__(self) -> int:
        return 3

    def get_frame(self, idx: int) -> dict[str, np.ndarray]:
        image = np.array([[[0, 64, 255]]], dtype=np.uint8)
        return {"frame_id": idx, "image": image}


def test_import_torchcodec_raises_helpful_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_mod.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing ffmpeg")),
    )

    with pytest.raises(ImportError, match="Install FFmpeg"):
        video_mod._import_torchcodec()


def test_video_iterator_torchcodec_backend_and_frame_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.touch()

    monkeypatch.setattr(video_mod, "_import_torchcodec", lambda: _DecoderWithReadError)

    iterator = video_mod.VideoIterator(str(video_path))

    assert len(iterator) == 2
    assert iterator.enable_random_access is True
    assert iterator.frame_rate == 12.5
    assert iterator.image_width == 2
    assert iterator.image_height == 2

    first = iterator.get_frame(0)
    assert first["basename"] == "video"
    assert first["image"].dtype == np.uint8
    assert first["image"].shape == (2, 2, 3)

    second = iterator.get_frame(1)
    assert second["frame_id"] == 1
    assert second["image"].shape == (1, 1, 3)

    frames = list(iterator)
    assert [frame["frame_id"] for frame in frames] == [0, 1]


def test_video_iterator_falls_back_to_cv2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "fallback.mp4"
    video_path.touch()
    init_cap = _FakeCapture(
        fps=0.0,
        frame_count=4,
        width=7,
        height=9,
    )
    read_cap = _FakeCapture(
        frame=np.full((2, 3, 3), 7, dtype=np.uint8),
        ok=True,
    )
    fail_cap = _FakeCapture(frame=None, ok=False)
    caps = [init_cap, read_cap, fail_cap]

    monkeypatch.setattr(
        video_mod,
        "_import_torchcodec",
        lambda: (_ for _ in ()).throw(ImportError("missing torchcodec")),
    )
    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda _path: caps.pop(0))

    iterator = video_mod.VideoIterator(str(video_path))

    assert iterator._backend == "cv2"
    assert iterator.frame_rate == 10.0
    assert iterator.image_width == 7
    assert iterator.image_height == 9
    assert init_cap.released is True

    ok_frame = iterator.get_frame(2)
    assert ok_frame["basename"] == "fallback"
    assert ok_frame["image"].shape == (2, 3, 3)
    assert read_cap.position == 2
    assert read_cap.released is True

    missing_frame = iterator.get_frame(3)
    assert missing_frame["image"].shape == (1, 1, 3)
    assert fail_cap.position == 3
    assert fail_cap.released is True


def test_video_frame_dataset_and_datamodule_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = video_mod.VideoFrameDataset(_FakeVideoIter(frame_rate=15.0), end_frame=2)

    assert len(dataset) == 2
    item = dataset[1]
    assert item["frame_id"].item() == 1
    assert item["rgb_image"].dtype == torch.float32
    torch.testing.assert_close(
        item["rgb_image"],
        torch.tensor([[[1.0, 64.0 / 255.0, 0.0]]], dtype=torch.float32),
    )

    monkeypatch.setattr(video_mod, "VideoIterator", lambda _path: _FakeVideoIter(0.0))

    datamodule = video_mod.VideoFrameDataModule(
        video_path="movie.mp4",
        end_frame=2,
        batch_size=2,
    )
    with pytest.raises(RuntimeError, match="setup\\('predict'\\)"):
        datamodule.predict_dataloader()

    datamodule.setup(stage="predict")
    first_predict_ds = datamodule.predict_ds
    datamodule.setup(stage="predict")

    assert datamodule.predict_ds is first_predict_ds
    assert datamodule.fps == 10.0

    batch = next(iter(datamodule.predict_dataloader()))
    assert batch["rgb_image"].shape[0] == 2
    assert batch["frame_id"].tolist() == [0, 1]
