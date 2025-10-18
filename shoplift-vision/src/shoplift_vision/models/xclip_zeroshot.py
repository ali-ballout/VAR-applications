"""Zero-shot inference using X-CLIP."""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoModel, AutoProcessor


class XCLIPZeroShot:
    """Wrapper for zero-shot video classification with X-CLIP."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_prompts(self, prompts: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for label, texts in prompts.items():
                inputs = self.processor(
                    text=texts,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                encoded[label] = features.mean(dim=0)
        return encoded

    def predict(self, frames: torch.Tensor, label_embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        arrays = [frame.detach().cpu().numpy() for frame in frames]
        with torch.no_grad():
            inputs = self.processor(videos=arrays, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            video_features = self.model.get_video_features(**inputs)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        scores = []
        labels = list(label_embeddings.keys())
        for label in labels:
            text_feat = label_embeddings[label]
            score = torch.matmul(video_features, text_feat.to(self.device))
            scores.append(score.squeeze().item())
        logits = torch.tensor(scores)
        probs = torch.softmax(logits, dim=0)
        return {label: float(prob) for label, prob in zip(labels, probs)}


__all__ = ["XCLIPZeroShot"]
