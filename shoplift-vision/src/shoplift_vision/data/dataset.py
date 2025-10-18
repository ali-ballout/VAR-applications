"""Dataset utilities for video classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..utils.io import VideoData, load_video_frames

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


@dataclass
class DatasetConfig:
    root: Path
    split: str
    labels: List[str]
    num_frames: int
    sampling_strategy: str = "uniform"
    fps: Optional[float] = None
    stride: Optional[int] = None


class VideoClassificationDataset(Dataset[Dict[str, object]]):
    """Video dataset that reads from a folder structure."""

    def __init__(
        self,
        config: DatasetConfig,
        transform: Optional[Callable[[Iterable[torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.label_to_id = {label: idx for idx, label in enumerate(config.labels)}
        self.samples: List[Tuple[Path, int]] = []
        self._populate_samples()

    def _populate_samples(self) -> None:
        base_dir = self.config.root / self.config.split
        if not base_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {base_dir}")
        for label in self.config.labels:
            class_dir = base_dir / label
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in _VIDEO_EXTS:
                    self.samples.append((path, self.label_to_id[label]))
        if not self.samples:
            raise RuntimeError(f"No video files found in {base_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path, label_id = self.samples[index]
        video: VideoData = load_video_frames(
            str(path),
            num_frames=self.config.num_frames,
            strategy=self.config.sampling_strategy,
            fps=self.config.fps,
            stride=self.config.stride,
        )
        frames = [frame for frame in video.frames]
        if self.transform is not None:
            processed = self.transform(frames)
        else:
            tensor = torch.stack([frame.permute(2, 0, 1) for frame in frames])
            processed = {"pixel_values": tensor}
        processed.update({"labels": label_id, "path": str(path)})
        return processed


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    paths = [item["path"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "paths": paths}


__all__ = ["VideoClassificationDataset", "DatasetConfig", "collate_fn"]
