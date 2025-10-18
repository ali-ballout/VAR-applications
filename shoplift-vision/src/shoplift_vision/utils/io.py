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
