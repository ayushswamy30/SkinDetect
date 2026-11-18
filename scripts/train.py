"""Train a classifier on one cross-validation fold of HAM10000.

Each run writes to outputs/runs/<name>_fold<k>_<timestamp>/:
  config.yaml     the resolved config
  history.csv     per-epoch training loss and validation metrics
  best.pt         weights with the best validation balanced accuracy (plus config and metrics)
  val_preds.csv   the best model's class probabilities for every validation image
  summary.json    best epoch and its metrics

Usage:
    uv run python scripts/train.py --config configs/baseline.yaml
    uv run python scripts/train.py --config configs/baseline.yaml --set optim.lr=1e-4 data.val_fold=1
"""

import argparse
import json
import math
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from skinlesion.data import CLASSES, OUTPUTS_DIR, ROOT
from skinlesion.dataset import build_loaders, class_weights
from skinlesion.engine import build_model, predict, train_one_epoch
from skinlesion.metrics import compute_metrics, format_metrics


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[leaf] = yaml.safe_load(value)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int, final_ratio: float = 0.01):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases and normalisation parameters."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d, m, o = cfg["data"], cfg["model"], cfg["optim"]

    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["train"]["amp"]

    run_dir = OUTPUTS_DIR / "runs" / f"{cfg['name']}_fold{d['val_fold']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Run: {run_dir.relative_to(ROOT)}  device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}")

    train_loader, val_loader = build_loaders(d["val_fold"], d["img_size"], d["batch_size"], d["num_workers"],
                                             d["sampler_power"], cfg["seed"], d["cache_short_side"])
    print(f"Train {len(train_loader.dataset):,} images, val {len(val_loader.dataset):,} images")

    model = build_model(m["name"], m["pretrained"], m["drop_rate"], m["drop_path_rate"])
    model = model.to(device, memory_format=torch.channels_last)

    weights = None
    if cfg["loss"]["class_weight_power"] > 0:
        weights = class_weights(train_loader.dataset.labels, cfg["loss"]["class_weight_power"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["loss"]["label_smoothing"])

    optimizer = torch.optim.AdamW(param_groups(model, o["weight_decay"]), lr=o["lr"])
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(optimizer, o["warmup_epochs"] * steps_per_epoch, o["epochs"] * steps_per_epoch)

    history, best = [], None
    for epoch in range(1, o["epochs"] + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, amp,
                                     desc=f"epoch {epoch}/{o['epochs']}")
        probs, labels = predict(model, val_loader, device, amp, desc="val")
        metrics = compute_metrics(labels, probs)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(np.log(probs + 1e-12)),
                                                     torch.from_numpy(labels), reduction="mean"))
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics, "seconds": elapsed})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = best is None or metrics["balanced_acc"] > best["balanced_acc"]
        if improved:
            best = {"epoch": epoch, **metrics}
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics}, run_dir / "best.pt")
            preds = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
            preds.insert(0, "label", labels)
            preds.insert(0, "image_id", val_loader.dataset.image_ids)
            preds.to_csv(run_dir / "val_preds.csv", index=False)
        print(f"epoch {epoch:>2}/{o['epochs']}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"{elapsed:.0f}s{'  *best*' if improved else ''}\n    {format_metrics(metrics)}")

    summary = {"run": run_dir.name, "best_epoch": best["epoch"], "best": best,
               "minutes": sum(h["seconds"] for h in history) / 60}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest epoch {best['epoch']}:\n    {format_metrics(best)}\nSaved to {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
