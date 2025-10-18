"""Command-line entry point for fine-tuning VideoMAE."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml
from transformers import Trainer, TrainingArguments, set_seed

from .data.dataset import DatasetConfig, VideoClassificationDataset, collate_fn
from .data.transforms import build_videomae_transform
from .models.videomae_finetune import create_videomae_model
from .utils.metrics import compute_classification_metrics, format_confusion_matrix


def _merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "base_config" in config:
        base_path = (path.parent / config["base_config"]).resolve()
        base = _load_config(base_path)
        config = _merge_dict(base, {k: v for k, v in config.items() if k != "base_config"})
    return config


def _parse_override(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = dict(config)
    for override in overrides:
        key, _, value = override.partition("=")
        if not key:
            continue
        parts = key.split(".")
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_override(value)
    return result


def _ensure_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE for shoplifting detection")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values using key=value (supports dotted paths)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    if args.override:
        cfg = _apply_overrides(cfg, args.override)

    device = _ensure_device(cfg.get("device", "cuda"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    labels = cfg.get("labels", ["shoplifting", "no_shoplifting"])
    data_root = Path(cfg["paths"]["data_dir"]).resolve()
    runs_root = Path(cfg["paths"].get("runs_dir", "runs")).resolve()
    experiment_name = cfg.get("experiment_name", "videomae_run")
    run_dir = runs_root / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)

    transform = build_videomae_transform(cfg["processor"]["name"], cfg["video"].get("img_size"))

    dataset_kwargs = dict(
        labels=labels,
        num_frames=int(cfg["video"]["num_frames"]),
        sampling_strategy=cfg["video"].get("sampling_strategy", "uniform"),
        fps=cfg["video"].get("fps"),
        stride=cfg["video"].get("stride"),
    )

    train_dataset = VideoClassificationDataset(
        DatasetConfig(root=data_root, split="train", **dataset_kwargs),
        transform=transform,
    )
    val_dataset = VideoClassificationDataset(
        DatasetConfig(root=data_root, split="val", **dataset_kwargs),
        transform=transform,
    )

    model = create_videomae_model(cfg["model"]["name"], labels)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=int(cfg["train"]["batch_size"]),
        per_device_eval_batch_size=int(cfg["train"].get("eval_batch_size", cfg["train"]["batch_size"])),
        num_train_epochs=int(cfg["train"]["epochs"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        warmup_ratio=float(cfg["train"].get("warmup_ratio", 0.0)),
        evaluation_strategy=cfg["train"].get("evaluation_strategy", "epoch"),
        save_strategy=cfg["train"].get("save_strategy", "epoch"),
        logging_steps=int(cfg["train"].get("logging_steps", 50)),
        gradient_accumulation_steps=int(cfg["train"].get("gradient_accumulation_steps", 1)),
        load_best_model_at_end=True,
        metric_for_best_model=cfg["train"].get("metric_for_best_model", "f1"),
        greater_is_better=cfg["train"].get("greater_is_better", True),
        fp16=bool(cfg["train"].get("fp16", False) and device != "cpu"),
        dataloader_num_workers=int(cfg.get("num_workers", 4)),
        report_to=["none"],
    )

    def compute_metrics_wrapper(eval_pred: Any) -> Dict[str, float]:
        logits, labels_array = eval_pred
        preds = logits.argmax(axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        metrics = compute_classification_metrics(labels_array, preds, probs, labels)
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_wrapper,
        data_collator=collate_fn,
    )

    trainer.train()

    best_model_path = trainer.state.best_model_checkpoint or str(run_dir)
    trainer.save_model(best_model_path)

    eval_metrics = trainer.evaluate()
    prediction_output = trainer.predict(val_dataset)
    logits = prediction_output.predictions
    labels_array = prediction_output.label_ids
    preds = logits.argmax(axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    metrics_full = compute_classification_metrics(labels_array, preds, probs, labels)
    cm_str = format_confusion_matrix(metrics_full["confusion_matrix"], labels)
    print("Validation metrics:", json.dumps(eval_metrics, indent=2))
    print("Confusion matrix:\n" + cm_str)

    best_state_path = Path(best_model_path) / "best.pt"
    torch.save(model.state_dict(), best_state_path)
    print(f"Best model weights saved to {best_state_path}")


if __name__ == "__main__":
    main()
