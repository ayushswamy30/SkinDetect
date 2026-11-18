"""Classification metrics. Balanced accuracy (mean per-class recall) is the ISIC 2018 Task 3 ranking metric."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from skinlesion.data import CLASSES


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Metrics from integer labels (N,) and class probabilities (N, C)."""
    labels = np.asarray(labels)
    preds = probs.argmax(1)
    k = list(range(len(CLASSES)))
    out = {
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, labels=k, average="macro", zero_division=0),
    }
    try:
        out["macro_auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=k)
    except ValueError:  # a class is absent from `labels`
        out["macro_auc"] = float("nan")
    for c, r in zip(CLASSES, recall_score(labels, preds, labels=k, average=None, zero_division=0)):
        out[f"recall_{c}"] = r
    return {name: float(v) for name, v in out.items()}


def format_metrics(m: dict[str, float]) -> str:
    head = f"bal_acc {m['balanced_acc']:.4f}  acc {m['acc']:.4f}  f1 {m['macro_f1']:.4f}  auc {m['macro_auc']:.4f}"
    recalls = "  ".join(f"{c} {m[f'recall_{c}']:.2f}" for c in CLASSES)
    return f"{head}\n    recall: {recalls}"
