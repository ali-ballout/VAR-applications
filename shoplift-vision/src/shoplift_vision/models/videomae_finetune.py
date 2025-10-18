"""VideoMAE fine-tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from transformers import VideoMAEForVideoClassification


def create_videomae_model(model_name: str, label_names: List[str]) -> VideoMAEForVideoClassification:
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model


@dataclass
class TrainingResult:
    metrics: Dict[str, float]
    best_model_path: str


__all__ = ["create_videomae_model", "TrainingResult"]
