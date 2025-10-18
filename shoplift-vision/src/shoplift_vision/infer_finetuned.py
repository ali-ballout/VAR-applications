"""Inference script for finetuned VideoMAE models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from .data.transforms import build_videomae_transform
from .models.videomae_finetune import create_videomae_model
from .utils.io import load_video_frames


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_device(device: str | None) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _run_inference(
    model: torch.nn.Module,
    transform,
    labels: List[str],
    video_path: Path,
    device: torch.device,
    cfg: Dict[str, object],
) -> Dict[str, float]:
    video = load_video_frames(
        str(video_path),
        num_frames=int(cfg["video"]["num_frames"]),
        strategy=cfg["video"].get("sampling_strategy", "uniform"),
        fps=cfg["video"].get("fps"),
        stride=cfg["video"].get("stride"),
    )
    processed = transform([frame for frame in video.frames])
    pixel_values = processed["pixel_values"].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    return {label: float(prob) for label, prob in zip(labels, probs.cpu())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a finetuned VideoMAE model")
    parser.add_argument("--config", type=str, required=True, help="Config file used during training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or HF checkpoint")
    parser.add_argument("--video", type=str, help="Path to a video file")
    parser.add_argument("--directory", type=str, help="Directory of videos to evaluate")
    parser.add_argument("--output", type=str, help="Optional path to write JSON results")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    args = parser.parse_args()

    if not args.video and not args.directory:
        raise ValueError("Provide --video or --directory")

    cfg = _load_config(Path(args.config))
    device_str = _resolve_device(args.device or cfg.get("device"))
    device = torch.device(device_str)

    labels = cfg.get("labels", ["shoplifting", "no_shoplifting"])
    model = create_videomae_model(cfg["model"]["name"], labels)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    else:
        model = model.from_pretrained(str(checkpoint_path))
    model.to(device)
    model.eval()

    transform = build_videomae_transform(cfg["processor"]["name"], cfg["video"].get("img_size"))

    targets: List[Path] = []
    if args.video:
        targets.append(Path(args.video))
    if args.directory:
        targets.extend(Path(args.directory).glob("**/*"))

    results: Dict[str, Dict[str, float]] = {}
    for video_path in targets:
        if not video_path.is_file():
            continue
        try:
            probs = _run_inference(model, transform, labels, video_path, device, cfg)
            results[str(video_path)] = probs
            best_label = max(probs, key=probs.get)
            print(f"{video_path}: {best_label} ({probs[best_label]:.3f})")
        except Exception as exc:  # pragma: no cover - runtime logging
            print(f"Failed to process {video_path}: {exc}")

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
