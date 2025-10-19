"""Zero-shot inference using X-CLIP."""

from __future__ import annotations

import hashlib
import warnings
from typing import Dict, List

import torch
from transformers import AutoModel, AutoProcessor


class XCLIPZeroShot:
    """Wrapper for zero-shot video classification with X-CLIP."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._fallback = False
        self._embed_dim = 32
        self._fallback_proj = self._init_fallback_proj()
        if model_name in {"dummy", "debug-dummy"}:
            self._fallback = True
            self.model = None
            self.processor = None
            return
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model.eval()
        except OSError as exc:  # pragma: no cover - requires offline execution
            warnings.warn(
                f"Falling back to lightweight zero-shot stub for '{model_name}': {exc}",
                RuntimeWarning,
            )
            self._fallback = True
            self.model = None
            self.processor = None

    def _init_fallback_proj(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        return torch.randn(3, self._embed_dim, generator=generator)

    def encode_prompts(self, prompts: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        if self._fallback:
            for label, texts in prompts.items():
                joined = " ".join(texts)
                digest = hashlib.sha256((label + joined).encode("utf-8")).hexdigest()
                seed = int(digest[:8], 16)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                vec = torch.randn(self._embed_dim, generator=generator)
                encoded[label] = vec / vec.norm().clamp(min=1e-6)
            return encoded
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
        if self._fallback:
            rgb_mean = frames.float().mean(dim=(0, 1, 2))
            video_vec = rgb_mean @ self._fallback_proj
            norm = video_vec.norm().clamp(min=1e-6)
            video_features = (video_vec / norm).unsqueeze(0)
        else:
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
            feature = text_feat.to(self.device) if not self._fallback else text_feat
            score = torch.matmul(video_features, feature)
            scores.append(score.squeeze().item())
        logits = torch.tensor(scores)
        probs = torch.softmax(logits, dim=0)
        return {label: float(prob) for label, prob in zip(labels, probs)}


__all__ = ["XCLIPZeroShot"]
