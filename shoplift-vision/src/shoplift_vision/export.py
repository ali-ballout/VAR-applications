"""Model export utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from .models.videomae_finetune import create_videomae_model


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _prepare_model(cfg: Dict[str, object], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    labels: List[str] = cfg.get("labels", ["shoplifting", "no_shoplifting"])
    model = create_videomae_model(cfg["model"]["name"], labels)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


class VideoMAEWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


def export_torchscript(model: torch.nn.Module, dummy: torch.Tensor, output: Path) -> None:
    wrapper = VideoMAEWrapper(model)
    traced = torch.jit.trace(wrapper, dummy)
    traced.save(str(output))


def export_onnx(model: torch.nn.Module, dummy: torch.Tensor, output: Path) -> None:
    wrapper = VideoMAEWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        export_params=True,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export finetuned models")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--format", choices=["torchscript", "onnx"], required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _prepare_model(cfg, Path(args.checkpoint), device)

    num_frames = int(cfg["video"]["num_frames"])
    img_size = int(cfg["video"].get("img_size", 224))
    dummy = torch.zeros(1, num_frames, 3, img_size, img_size, device=device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "torchscript":
        export_torchscript(model, dummy, output_path)
    else:
        export_onnx(model, dummy, output_path)
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    main()
