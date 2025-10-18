"""Metric helpers for classification."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
) -> Dict[str, float | np.ndarray]:
    """Compute standard classification metrics."""

    metrics: Dict[str, float | np.ndarray] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=None if labels is None else list(range(len(labels))))
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr")
            )
        except ValueError:
            pass
    return metrics


def format_confusion_matrix(cm: np.ndarray, labels: List[str]) -> str:
    """Return a human-readable confusion matrix string."""

    header = "\t".join([" "] + labels)
    rows = [header]
    for idx, label in enumerate(labels):
        values = "\t".join(str(int(x)) for x in cm[idx])
        rows.append(f"{label}\t{values}")
    return "\n".join(rows)


__all__ = ["compute_classification_metrics", "format_confusion_matrix"]
