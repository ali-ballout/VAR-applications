"""Transform helpers for video inputs."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEImageProcessor


class VideoTransform:
    """Callable wrapper to process frames using a Hugging Face image processor."""

    def __init__(self, processor: VideoMAEImageProcessor):
        self.processor = processor

    def __call__(self, frames: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
        arrays: List[np.ndarray] = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                arrays.append(frame.detach().cpu().numpy())
            else:
                arrays.append(np.asarray(frame))
        inputs = self.processor(videos=arrays, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]
        return {"pixel_values": pixel_values}


def build_videomae_transform(name: str, image_size: int | None = None) -> VideoTransform:
    processor = VideoMAEImageProcessor.from_pretrained(name)
    if image_size is not None:
        processor.size = {"shortest_edge": image_size}
    return VideoTransform(processor)


def build_generic_image_processor(name: str) -> AutoImageProcessor:
    return AutoImageProcessor.from_pretrained(name)


__all__ = ["build_videomae_transform", "build_generic_image_processor", "VideoTransform"]
