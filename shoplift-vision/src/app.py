"""Gradio application for shoplift-vision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch
import yaml

from shoplift_vision.infer_finetuned import _load_config as load_train_config
from shoplift_vision.infer_finetuned import _resolve_device as resolve_train_device
from shoplift_vision.infer_finetuned import _run_inference as run_finetuned
from shoplift_vision.infer_zeroshot import _load_config as load_zero_config
from shoplift_vision.models.videomae_finetune import create_videomae_model
from shoplift_vision.models.xclip_zeroshot import XCLIPZeroShot
from shoplift_vision.utils.io import load_video_frames
from shoplift_vision.data.transforms import build_videomae_transform


def build_zero_shot_pipeline(config_path: Path, device: str) -> Tuple[XCLIPZeroShot, Dict[str, torch.Tensor], Dict[str, object]]:
    cfg = load_zero_config(config_path)
    labels = cfg.get("labels")
    if labels is None:
        raise ValueError("Labels missing from zero-shot config")
    prompts = cfg.get("prompts")
    if prompts is None:
        raise ValueError("Prompts missing from zero-shot config")
    model = XCLIPZeroShot(cfg["model"]["name"], device=device)
    embeddings = model.encode_prompts({label: prompts[label] for label in labels})
    return model, embeddings, cfg


def run_zero_shot(
    model: XCLIPZeroShot,
    embeddings: Dict[str, torch.Tensor],
    cfg: Dict[str, object],
    video_path: Path,
) -> Dict[str, float]:
    video = load_video_frames(
        str(video_path),
        num_frames=int(cfg["video"]["num_frames"]),
        strategy=cfg["video"].get("sampling_strategy", "uniform"),
        fps=cfg["video"].get("fps"),
        stride=cfg["video"].get("stride"),
    )
    return model.predict(video.frames, embeddings)


def build_finetuned_pipeline(
    config_path: Path,
    checkpoint: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str], Dict[str, object], callable]:
    cfg = load_train_config(config_path)
    labels = cfg.get("labels", ["shoplifting", "no_shoplifting"])
    model = create_videomae_model(cfg["model"]["name"], labels)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    transform = build_videomae_transform(cfg["processor"]["name"], cfg["video"].get("img_size"))
    return model, labels, cfg, transform


def create_interface(args: argparse.Namespace) -> gr.Blocks:
    if args.mode == "zeroshot":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, embeddings, cfg = build_zero_shot_pipeline(Path(args.config), device)

        def predict(video: str) -> Dict[str, object]:
            if video is None:
                return {"label": "", "probabilities": {}}
            probs = run_zero_shot(model, embeddings, cfg, Path(video))
            best = max(probs, key=probs.get)
            return {"label": best, "probabilities": probs}

    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for finetuned mode")
        cfg = load_train_config(Path(args.config))
        device_str = resolve_train_device(args.device or cfg.get("device"))
        device = torch.device(device_str)
        model, labels, cfg, transform = build_finetuned_pipeline(
            Path(args.config), Path(args.checkpoint), device
        )

        def predict(video: str) -> Dict[str, object]:
            if video is None:
                return {"label": "", "probabilities": {}}
            probs = run_finetuned(
                model,
                transform,
                labels,
                Path(video),
                device,
                cfg,
            )
            best = max(probs, key=probs.get)
            return {"label": best, "probabilities": probs}

    with gr.Blocks() as demo:
        gr.Markdown("## shoplift-vision Demo")
        with gr.Row():
            video_input = gr.Video(sources=["upload", "webcam"], label="Input video")
        output_label = gr.Label(label="Predicted label")
        output_json = gr.JSON(label="Probabilities")

        def _on_submit(video: str) -> Tuple[str, Dict[str, float]]:
            result = predict(video)
            return result["label"], result["probabilities"]

        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(_on_submit, inputs=video_input, outputs=[output_label, output_json])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for shoplift-vision")
    parser.add_argument("--mode", choices=["zeroshot", "finetuned"], default="zeroshot")
    parser.add_argument("--config", type=str, default="configs/zero_shot.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to finetuned checkpoint")
    parser.add_argument("--device", type=str, help="Device override")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = create_interface(args)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
