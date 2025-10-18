"""Evaluate a directory of videos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from shoplift_vision.infer_finetuned import _load_config as load_train_config
from shoplift_vision.infer_finetuned import _resolve_device as resolve_device
from shoplift_vision.infer_finetuned import _run_inference as run_finetuned
from shoplift_vision.infer_zeroshot import _load_config as load_zero_config
from shoplift_vision.models.videomae_finetune import create_videomae_model
from shoplift_vision.models.xclip_zeroshot import XCLIPZeroShot
from shoplift_vision.utils.io import load_video_frames
from shoplift_vision.data.transforms import build_videomae_transform
from shoplift_vision.utils.metrics import compute_classification_metrics


def _collect_dataset(root: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for label_dir in root.iterdir():
        if label_dir.is_dir():
            for video in label_dir.glob("*"):
                if video.is_file():
                    samples.append((video, label_dir.name))
    if not samples:
        raise ValueError(f"No videos found in {root}")
    return samples


def evaluate_zeroshot(args: argparse.Namespace) -> None:
    cfg = load_zero_config(Path(args.config))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels = cfg.get("labels")
    if labels is None:
        raise ValueError("Labels missing from zero-shot config")
    prompts = cfg.get("prompts")
    if prompts is None:
        raise ValueError("Prompts missing from config")

    model = XCLIPZeroShot(cfg["model"]["name"], device=device)
    embeddings = model.encode_prompts({label: prompts[label] for label in labels})

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    for video_path, label in _collect_dataset(Path(args.directory)):
        video = load_video_frames(
            str(video_path),
            num_frames=int(cfg["video"]["num_frames"]),
            strategy=cfg["video"].get("sampling_strategy", "uniform"),
            fps=cfg["video"].get("fps"),
            stride=cfg["video"].get("stride"),
        )
        probs = model.predict(video.frames, embeddings)
        prob_list = [probs[lbl] for lbl in labels]
        y_prob.append(prob_list)
        y_true.append(labels.index(label))
        y_pred.append(int(np.argmax(prob_list)))
        print(f"{video_path.name}: predicted={labels[y_pred[-1]]} prob={max(prob_list):.3f}")

    metrics = compute_classification_metrics(y_true, y_pred, np.array(y_prob), labels)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def evaluate_finetuned(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for finetuned evaluation")
    cfg = load_train_config(Path(args.config))
    device_str = resolve_device(args.device or cfg.get("device"))
    device = torch.device(device_str)
    labels = cfg.get("labels", ["shoplifting", "no_shoplifting"])

    model = create_videomae_model(cfg["model"]["name"], labels)
    state = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = build_videomae_transform(cfg["processor"]["name"], cfg["video"].get("img_size"))

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    for video_path, label in _collect_dataset(Path(args.directory)):
        probs = run_finetuned(model, transform, labels, video_path, device, cfg)
        prob_list = [probs[lbl] for lbl in labels]
        y_prob.append(prob_list)
        y_true.append(labels.index(label))
        y_pred.append(int(np.argmax(prob_list)))
        print(f"{video_path.name}: predicted={labels[y_pred[-1]]} prob={max(prob_list):.3f}")

    metrics = compute_classification_metrics(y_true, y_pred, np.array(y_prob), labels)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a directory of labeled videos")
    parser.add_argument("--mode", choices=["zeroshot", "finetuned"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for finetuned mode")
    parser.add_argument("--device", type=str, help="Device override")
    args = parser.parse_args()

    if args.mode == "zeroshot":
        evaluate_zeroshot(args)
    else:
        evaluate_finetuned(args)


if __name__ == "__main__":
    main()
