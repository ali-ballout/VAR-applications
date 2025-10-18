from pathlib import Path

import cv2
import numpy as np
import torch

from shoplift_vision.infer_finetuned import _resolve_device, _run_inference


class DummyModel(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.nn.Module:  # type: ignore[override]
        batch = pixel_values.shape[0]
        logits = torch.tensor([[2.0, 1.0]], dtype=torch.float32).repeat(batch, 1)
        return type("Output", (), {"logits": logits})


def create_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (32, 32),
    )
    for _ in range(4):
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def dummy_transform(frames):
    tensor = torch.stack([frame.permute(2, 0, 1) for frame in frames])
    return {"pixel_values": tensor}


def test_run_inference_returns_probabilities(tmp_path):
    video_path = tmp_path / "video.mp4"
    create_video(video_path)
    model = DummyModel()
    cfg = {
        "video": {
            "num_frames": 4,
            "sampling_strategy": "uniform",
        }
    }
    probs = _run_inference(model, dummy_transform, ["a", "b"], video_path, torch.device("cpu"), cfg)
    assert set(probs.keys()) == {"a", "b"}
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_resolve_device_cpu_fallback():
    device = _resolve_device("cuda")
    assert device in {"cuda", "cpu"}
