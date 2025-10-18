"""Frame sampling utilities for video processing."""

from __future__ import annotations

from typing import List

import numpy as np


def _pad_indices(indices: List[int], num_samples: int) -> List[int]:
    """Repeat indices until reaching the desired number of samples."""

    if not indices:
        return [0] * num_samples
    padded = list(indices)
    while len(padded) < num_samples:
        padded.extend(indices)
    return padded[:num_samples]


def uniform_sample_indices(total_frames: int, num_samples: int) -> List[int]:
    """Return frame indices sampled uniformly across a video."""

    if total_frames <= 0:
        return [0] * num_samples
    if total_frames < num_samples:
        indices = list(range(total_frames))
        return _pad_indices(indices, num_samples)

    ticks = np.linspace(0, total_frames - 1, num_samples)
    return [int(round(t)) for t in ticks]


def strided_sample_indices(total_frames: int, num_samples: int, stride: int) -> List[int]:
    """Return frame indices sampled with a fixed stride."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    if total_frames <= 0:
        return [0] * num_samples

    indices = list(range(0, total_frames, stride))
    return _pad_indices(indices, num_samples)


def temporal_sampling(
    total_frames: int,
    num_samples: int,
    strategy: str = "uniform",
    stride: int | None = None,
) -> List[int]:
    """Sample frame indices according to the selected strategy."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    strategy = strategy.lower()
    if strategy == "uniform":
        return uniform_sample_indices(total_frames, num_samples)
    if strategy == "strided":
        if stride is None:
            raise ValueError("stride must be provided for strided sampling")
        return strided_sample_indices(total_frames, num_samples, stride)

    raise ValueError(f"Unsupported sampling strategy: {strategy}")


__all__ = [
    "temporal_sampling",
    "uniform_sample_indices",
    "strided_sample_indices",
]
