"""Sanity-check the data pipeline before training.

Prints split sizes, class weights and the class mix produced by the balanced sampler, times the
training loader, and saves augmented and validation image grids to outputs/preview/.

Usage:
    uv run python scripts/preview_batch.py [--fold 0] [--img-size 224] [--workers 4]
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from skinlesion.data import CLASSES, OUTPUTS_DIR
from skinlesion.dataset import balanced_sampler, build_loaders, class_counts, class_weights, denormalize

OUT_DIR = OUTPUTS_DIR / "preview"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"


def save_grid(images: torch.Tensor, labels: torch.Tensor, title: str, name: str, cols: int = 8) -> None:
    images = denormalize(images).permute(0, 2, 3, 1).numpy()
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.75), facecolor=SURFACE)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=9, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved outputs/preview/{name}")


def sampler_mix(labels: np.ndarray) -> None:
    print("\nClass share (%) of one training epoch, by sampler power")
    print(f"  {'power':<14}" + "".join(f"{c:>7}" for c in CLASSES))
    for power in (0.0, 0.5, 1.0):
        if power == 0:
            drawn = labels
        else:
            drawn = labels[np.fromiter(balanced_sampler(labels, power), dtype=np.int64)]
        share = 100 * class_counts(drawn) / len(drawn)
        name = "0 (shuffle)" if power == 0 else str(power)
        print(f"  {name:<14}" + "".join(f"{s:>7.1f}" for s in share))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-batches", type=int, default=30)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args.fold, args.img_size, args.batch_size, args.workers)
    train_labels = train_loader.dataset.labels
    val_labels = val_loader.dataset.labels
    print(f"Fold {args.fold}: {len(train_labels):,} train / {len(val_labels):,} val images")
    print(f"  {'':<14}" + "".join(f"{c:>7}" for c in CLASSES))
    print(f"  {'train count':<14}" + "".join(f"{n:>7}" for n in class_counts(train_labels)))
    print(f"  {'val count':<14}" + "".join(f"{n:>7}" for n in class_counts(val_labels)))
    for power in (0.5, 1.0):
        print(f"  {f'weight p={power}':<14}" + "".join(f"{w:>7.2f}" for w in class_weights(train_labels, power)))

    sampler_mix(train_labels)

    print("\nBatches")
    images, labels = next(iter(train_loader))
    assert images.shape == (args.batch_size, 3, args.img_size, args.img_size), images.shape
    assert images.dtype == torch.float32 and torch.isfinite(images).all()
    print(f"  train batch {tuple(images.shape)}, mean {images.mean():.3f}, std {images.std():.3f}")
    save_grid(images, labels, f"Augmented training batch (fold {args.fold})", "train_batch.png")

    images, labels = next(iter(val_loader))
    assert images.shape[1:] == (3, args.img_size, args.img_size), images.shape
    print(f"  val batch {tuple(images.shape)}, mean {images.mean():.3f}, std {images.std():.3f}")
    save_grid(images, labels, f"Validation batch, no augmentation (fold {args.fold})", "val_batch.png")

    it = iter(train_loader)
    next(it)  # exclude worker start-up
    start = time.perf_counter()
    n = 0
    for _ in range(args.time_batches):
        images, _ = next(it)
        n += len(images)
    rate = n / (time.perf_counter() - start)
    print(f"\nTrain loader: {rate:.0f} images/s with {args.workers} workers "
          f"(~{len(train_labels) / rate:.0f} s per epoch if data loading is the bottleneck)")


if __name__ == "__main__":
    main()
