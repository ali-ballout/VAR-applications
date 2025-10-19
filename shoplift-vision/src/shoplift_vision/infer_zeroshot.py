"""Zero-shot inference CLI using X-CLIP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

if __package__ in {None, ""}:  # pragma: no cover - CLI convenience
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from models.xclip_zeroshot import XCLIPZeroShot  # type: ignore
    from utils.io import load_video_frames  # type: ignore
else:
    from .models.xclip_zeroshot import XCLIPZeroShot
    from .utils.io import load_video_frames


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_device(device: str | None) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _run_video(
    model: XCLIPZeroShot,
    config: Dict[str, object],
    labels: List[str],
    embeddings: Dict[str, torch.Tensor],
    video_path: Path,
) -> Dict[str, float]:
    video = load_video_frames(
        str(video_path),
        num_frames=int(config["video"]["num_frames"]),
        strategy=config["video"].get("sampling_strategy", "uniform"),
        fps=config["video"].get("fps"),
        stride=config["video"].get("stride"),
    )
    return model.predict(video.frames, embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot action recognition with X-CLIP")
    parser.add_argument("--config", type=str, required=True, help="Zero-shot config YAML")
    parser.add_argument("--video", type=str, help="Video path")
    parser.add_argument("--directory", type=str, help="Directory of videos")
    parser.add_argument(
        "--labels",
        type=str,
        help="Comma-separated labels overriding config",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--output", type=str, help="JSON output path")
    args = parser.parse_args()

    if not args.video and not args.directory:
        raise ValueError("Provide --video or --directory")

    cfg = _load_config(Path(args.config))
    device = _resolve_device(args.device)

    labels = args.labels.split(",") if args.labels else cfg.get("labels")
    if labels is None:
        raise ValueError("Labels must be provided via config or --labels")

    prompts = cfg.get("prompts")
    if prompts is None:
        raise ValueError("Zero-shot prompts missing from config")

    model = XCLIPZeroShot(cfg["model"]["name"], device=device)
    embeddings = model.encode_prompts({label: prompts[label] for label in labels})

    targets: List[Path] = []
    if args.video:
        targets.append(Path(args.video))
    if args.directory:
        targets.extend(Path(args.directory).glob("**/*"))

    results: Dict[str, Dict[str, float]] = {}
    for path in targets:
        if not path.is_file():
            continue
        try:
            probs = _run_video(model, cfg, labels, embeddings, path)
            results[str(path)] = probs
            best = max(probs, key=probs.get)
            print(f"{path}: {best} ({probs[best]:.3f})")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to process {path}: {exc}")

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
