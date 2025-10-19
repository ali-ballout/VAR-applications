"""Video I/O utilities with decord/torchvision fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torchvision.io import read_video

from .sampling import temporal_sampling

_USE_TORCHVISION = os.environ.get("SHOPLIFT_VISION_USE_TORCHVISION_IO") == "1"

try:  # pragma: no cover - exercised in integration tests
    if not _USE_TORCHVISION:
        from decord import VideoReader, cpu

        _HAS_DECORD = True
    else:
        _HAS_DECORD = False
except ImportError:  # pragma: no cover - environment dependent
    _HAS_DECORD = False


@dataclass
class VideoData:
    """Container for decoded video frames."""

    frames: torch.Tensor
    fps: float
    duration: float


def _load_with_decord(path: str) -> Tuple[np.ndarray, float, float]:
    reader = VideoReader(path, ctx=cpu(0))
    fps = float(reader.get_avg_fps())
    total_frames = len(reader)
    duration = total_frames / fps if fps > 0 else float(total_frames)
    frames = reader.get_batch(list(range(total_frames))).asnumpy()
    return frames, fps, duration


def _load_with_torchvision(path: str) -> Tuple[np.ndarray, float, float]:
    video, _, info = read_video(path, pts_unit="sec")
    fps = float(info.get("video_fps", 0.0))
    total_frames = video.shape[0]
    duration = total_frames / fps if fps > 0 else float(total_frames)
    frames = video.numpy()
    return frames, fps, duration


def _load_with_pillow(path: str) -> Tuple[np.ndarray, float, float]:
    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        frames = [np.array(frame.convert("RGB")) for frame in ImageSequence.Iterator(img)]
        if not frames:
            raise ValueError(f"No frames found in animated image: {path}")
        duration_ms = float(img.info.get("duration", 100))
    fps = 1000.0 / duration_ms if duration_ms > 0 else 10.0
    duration = len(frames) / fps if fps > 0 else float(len(frames))
    return np.stack(frames), fps, duration


def _load_from_numpy(path: str) -> Tuple[np.ndarray, float, float]:
    array = np.load(path)
    if isinstance(array, np.lib.npyio.NpzFile):
        if "frames" not in array:
            raise KeyError("Expected 'frames' key in npz container")
        frames = array["frames"]
    else:
        frames = array
    if frames.ndim != 4:
        raise ValueError("NumPy video array must have shape (T, H, W, C)")
    fps = 30.0
    duration = frames.shape[0] / fps if fps > 0 else float(frames.shape[0])
    return frames.astype(np.float32), fps, duration


def load_video_frames(
    path: str,
    num_frames: int,
    strategy: str = "uniform",
    fps: float | None = None,
    stride: int | None = None,
) -> VideoData:
    """Load and sample frames from a video file.

    Returns frames as a tensor of shape ``(num_frames, H, W, C)`` in float32 [0, 1].
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in {".gif", ".apng"}:
        frames_np, native_fps, duration = _load_with_pillow(path)
    elif ext in {".npy", ".npz"}:
        frames_np, native_fps, duration = _load_from_numpy(path)
    else:
        frames_np, native_fps, duration = (
            _load_with_decord(path) if _HAS_DECORD else _load_with_torchvision(path)
        )
    total_frames = frames_np.shape[0]
    effective_fps = fps or native_fps or 30.0
    indices = temporal_sampling(total_frames, num_frames, strategy=strategy, stride=stride)
    sampled = frames_np[indices]
    tensor = torch.from_numpy(sampled).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return VideoData(frames=tensor, fps=effective_fps, duration=duration)


__all__ = ["load_video_frames", "VideoData"]
