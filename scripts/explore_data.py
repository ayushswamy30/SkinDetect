"""Exploratory analysis of HAM10000: prints key statistics and saves figures to outputs/eda/.

Usage:
    uv run python scripts/explore_data.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from skinlesion.data import CLASS_NAMES, CLASSES, OUTPUTS_DIR, load_metadata

OUT_DIR = OUTPUTS_DIR / "eda"

# Reference chart palette (light mode); see the dataviz guidelines this follows.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "font.size": 10,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved outputs/eda/{name}")


def class_distribution(df: pd.DataFrame) -> None:
    counts = df.dx.value_counts().reindex(CLASSES).sort_values()
    lesions = df.groupby("dx").lesion_id.nunique()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(counts.index, counts.values, height=0.55, color=SERIES[0])
    for i, (c, n) in enumerate(counts.items()):
        ax.text(n + 60, i, f"{n:,} images · {lesions[c]:,} lesions · {100 * n / len(df):.1f}%",
                va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(counts)), [f"{c}  {CLASS_NAMES[c]}" for c in counts.index])
    ax.set_xlim(0, counts.max() * 1.55)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Images")
    ax.set_title("HAM10000 class distribution: nevi are two thirds of the data")
    save(fig, "class_distribution.png")


def images_per_lesion(df: pd.DataFrame) -> None:
    per = df.groupby("lesion_id").size().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(per.index.astype(str), per.values, width=0.55, color=SERIES[0])
    for x, n in zip(per.index.astype(str), per.values):
        ax.text(x, n + 80, f"{n:,}", ha="center", color=INK_2, fontsize=9)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Images of the same lesion")
    ax.set_ylabel("Lesions")
    multi = (df.groupby("lesion_id").size() > 1).sum()
    ax.set_title(f"{multi:,} of {df.lesion_id.nunique():,} lesions have more than one image")
    save(fig, "images_per_lesion.png")


def age_by_class(df: pd.DataFrame) -> None:
    data = [df.loc[df.dx == c, "age"].dropna().values for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.boxplot(data, tick_labels=CLASSES, widths=0.45, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=SERIES[0] + "26", edgecolor=SERIES[0], linewidth=1.5),
               medianprops=dict(color=SERIES[0], linewidth=2),
               whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel("Age (years)")
    ax.set_title("Age by diagnosis: median ranges from 45 (nv) to 70 (akiec, bcc)")
    save(fig, "age_by_class.png")


def diagnosis_method(df: pd.DataFrame) -> None:
    order = ["histo", "follow_up", "consensus", "confocal"]
    labels = {"histo": "Histopathology", "follow_up": "Follow-up", "consensus": "Expert consensus",
              "confocal": "Confocal microscopy"}
    share = pd.crosstab(df.dx, df.dx_type, normalize="index").reindex(index=CLASSES, columns=order).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    left = np.zeros(len(CLASSES))
    for color, method in zip(SERIES, order):
        # The surface-coloured edge draws the 2px gap between stacked segments
        ax.barh(CLASSES, share[method], left=left, height=0.55, color=color, label=labels[method],
                edgecolor=SURFACE, linewidth=2)
        left += share[method].values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of images")
    ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, handlelength=1, columnspacing=1.2)
    ax.set_title("How each diagnosis was confirmed", pad=28)
    save(fig, "diagnosis_method.png")


def sample_grid(df: pd.DataFrame, per_class: int = 6, seed: int = 0) -> None:
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.6, len(CLASSES) * 1.3))
    for r, c in enumerate(CLASSES):
        rows = df[df.dx == c].drop_duplicates("lesion_id").sample(per_class, random_state=seed)
        for j, path in enumerate(rows.path):
            ax = axes[r, j]
            ax.imshow(Image.open(path))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(c, rotation=0, ha="right", va="center", color=INK, fontsize=10)
    fig.suptitle("Random examples per class (one image per lesion)", x=0.02, ha="left", color=INK,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "sample_grid.png")


def print_stats(df: pd.DataFrame) -> None:
    print(f"Images: {len(df):,}   Lesions: {df.lesion_id.nunique():,}")
    print(f"Imbalance ratio (largest / smallest class): {df.dx.value_counts().max() / df.dx.value_counts().min():.0f}:1")
    print(f"Missing age: {df.age.isna().sum()}   sex unknown: {(df.sex == 'unknown').sum()}   "
          f"localization unknown: {(df.localization == 'unknown').sum()}")
    print("\nMedian age by class:")
    print(df.groupby("dx").age.median().reindex(CLASSES).to_string())
    print("\nTop body locations:")
    print(df.localization.value_counts().head(8).to_string())
    print("\nSource dataset:")
    print(df.dataset.value_counts().to_string())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    print_stats(df)
    print("\nFigures:")
    class_distribution(df)
    images_per_lesion(df)
    age_by_class(df)
    diagnosis_method(df)
    sample_grid(df)


if __name__ == "__main__":
    main()
