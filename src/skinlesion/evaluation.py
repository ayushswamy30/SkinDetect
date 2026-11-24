"""Finding finished runs, test-time augmentation, fold ensembles and evaluation reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from skinlesion.data import CLASSES, MALIGNANT
from skinlesion.engine import autocast, build_model

PROB_COLUMNS = [f"p_{c}" for c in CLASSES]
MALIGNANT_IDX = [CLASSES.index(c) for c in sorted(MALIGNANT)]


def find_runs(runs_dir: Path, name: str) -> dict[int, Path]:
    """The latest finished run (one with summary.json) of config `name` for each fold."""
    found: dict[int, Path] = {}
    for run_dir in sorted(runs_dir.glob(f"{name}_fold*")):  # timestamps sort chronologically
        if not (run_dir / "summary.json").exists():
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        if cfg["name"] == name:
            found[cfg["data"]["val_fold"]] = run_dir
    return dict(sorted(found.items()))


def load_oof(runs: dict[int, Path]) -> pd.DataFrame:
    """Out-of-fold predictions: every fold's saved validation probabilities, stacked, with a `fold` column."""
    parts = []
    for fold, run_dir in runs.items():
        df = pd.read_csv(run_dir / "val_preds.csv")
        df.insert(1, "fold", fold)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_model(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """The best checkpoint of a run, on `device` in eval mode, and the run's config."""
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    m = ckpt["config"]["model"]
    model = build_model(m["name"], pretrained=False, drop_rate=m["drop_rate"], drop_path_rate=m["drop_path_rate"])
    model.load_state_dict(ckpt["model"])
    return model.to(device, memory_format=torch.channels_last).eval(), ckpt["config"]


def dihedral_views(images: torch.Tensor, n_views: int) -> list[torch.Tensor]:
    """Up to 8 rotations/flips of a batch (N, C, H, W). A lesion has no canonical orientation, so all are valid."""
    if n_views not in (1, 2, 4, 8):
        raise ValueError("n_views must be 1, 2, 4 or 8")
    views = []
    for flip in (False, True):
        base = images.flip(-1) if flip else images
        for k in range(4):
            views.append(torch.rot90(base, k, dims=(-2, -1)))
    # Order so that small view counts are the most different: identity, hflip, rot180, vflip, then the 90° turns
    order = [0, 4, 2, 6, 1, 5, 3, 7]
    return [views[i] for i in order[:n_views]]


@torch.no_grad()
def predict_tta(models: list[nn.Module], loader: DataLoader, device: torch.device, n_views: int = 1,
                amp: bool = True, desc: str = "predict") -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities averaged over every model and every TTA view; returns (probs (N, C), labels (N,))."""
    probs, labels = [], []
    for images, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        total = torch.zeros(len(images), len(CLASSES), device=device)
        for view in dihedral_views(images, n_views):
            view = view.contiguous(memory_format=torch.channels_last)
            for model in models:
                with autocast(device, amp):
                    total += model(view).float().softmax(1)
        probs.append((total / (n_views * len(models))).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def malignancy_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Malignant (akiec, bcc, mel) vs benign, from the 7-class prediction.

    Sensitivity counts a malignant lesion as caught when the predicted class is any malignant class,
    which is what matters for a referral decision even if the exact malignant type is wrong.
    """
    is_malignant = np.isin(labels, MALIGNANT_IDX)
    pred_malignant = np.isin(probs.argmax(1), MALIGNANT_IDX)
    p_malignant = probs[:, MALIGNANT_IDX].sum(1)
    return {
        "malignant_sensitivity": float(pred_malignant[is_malignant].mean()),
        "malignant_specificity": float((~pred_malignant[~is_malignant]).mean()),
        "malignant_auc": float(roc_auc_score(is_malignant, p_malignant)),
    }


def per_class_report(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """Precision, recall, F1, one-vs-rest AUC and support for every class."""
    k = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(labels, probs.argmax(1), labels=k,
                                                                     zero_division=0)
    auc = [roc_auc_score(labels == i, probs[:, i]) if 0 < (labels == i).sum() < len(labels) else np.nan for i in k]
    return pd.DataFrame({"class": CLASSES, "precision": precision, "recall": recall, "f1": f1, "auc": auc,
                         "support": support})


def plot_confusion(labels: np.ndarray, probs: np.ndarray, path: Path, title: str) -> None:
    """Confusion matrix normalised by true class (rows sum to 1), with the raw counts printed underneath."""
    counts = confusion_matrix(labels, probs.argmax(1), labels=list(range(len(CLASSES))))
    rates = counts / counts.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{rates[i, j]:.2f}\n({counts[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if rates[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities as softmax(log(p) / T). T > 1 softens over-confident predictions; argmax is unchanged.

    log(p) equals the logits up to a per-row constant, so this is exact for a single model and a close
    approximation for probabilities averaged over TTA views or an ensemble.
    """
    logits = np.log(np.clip(probs, 1e-12, None)) / temperature
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def nll(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(labels: np.ndarray, probs: np.ndarray) -> float:
    """The temperature that minimises the negative log-likelihood of `labels`."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda log_t: nll(labels, apply_temperature(probs, float(np.exp(log_t)))),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: the gap between confidence and accuracy, averaged over equal-width confidence bins."""
    conf, correct = probs.max(1), probs.argmax(1) == labels
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def calibration_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {"nll": nll(labels, probs), "ece": expected_calibration_error(labels, probs)}


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2))
