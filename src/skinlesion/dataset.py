"""PyTorch datasets, augmentations and class-imbalance helpers for HAM10000."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from skinlesion.data import CLASSES, FOLDS_CSV, TEST_CSV, TEST_IMAGE_DIR, TRAIN_IMAGE_DIR, resized_image_dir

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SkinLesionDataset(Dataset):
    """Reads (image, label) pairs from a split dataframe with `image_id` and `label` columns."""

    def __init__(self, df: pd.DataFrame, image_dir, transform=None):
        self.image_ids = df.image_id.tolist()
        self.labels = np.array(df.label, dtype=np.int64)  # a writable copy, not a read-only pandas view
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_dir / f"{self.image_ids[idx]}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[idx])


def build_transforms(img_size: int, train: bool) -> v2.Compose:
    """Augmentations for dermoscopy: a lesion has no canonical orientation, so any flip or rotation is valid.

    Evaluation resizes the short side and centre-crops, which keeps the 4:3 images undistorted.
    """
    to_tensor = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return v2.Compose([v2.ToImage(), v2.Resize(img_size, antialias=True), v2.CenterCrop(img_size), *to_tensor])
    return v2.Compose([
        v2.ToImage(),
        # Rotation leaves black corners. Many dermoscopy images already have a dark vignette,
        # so that's acceptable, and the random crop often removes them anyway.
        v2.RandomRotation(degrees=180, interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.75, 4 / 3), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        # No hue jitter: it is the slowest CPU augmentation and lesion colour carries diagnostic signal
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        *to_tensor,
    ])


def load_folds() -> pd.DataFrame:
    return pd.read_csv(FOLDS_CSV)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def split_fold(folds: pd.DataFrame, val_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on every fold except `val_fold`, validate on `val_fold`."""
    if val_fold not in set(folds.fold):
        raise ValueError(f"fold {val_fold} not in {sorted(set(folds.fold))}")
    train = folds[folds.fold != val_fold].reset_index(drop=True)
    val = folds[folds.fold == val_fold].reset_index(drop=True)
    return train, val


def class_counts(labels: Sequence[int]) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=len(CLASSES))


def class_weights(labels: Sequence[int], power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency loss weights, softened by `power` (1 = full inverse frequency, 0 = uniform).

    Normalised so the average weight per training sample is 1, which keeps the loss scale comparable.
    """
    counts = class_counts(labels).astype(np.float64)
    w = (counts.sum() / counts) ** power
    w /= (w * counts).sum() / counts.sum()
    return torch.tensor(w, dtype=torch.float32)


def balanced_sampler(labels: Sequence[int], power: float = 0.5, seed: int = 0) -> WeightedRandomSampler:
    """Oversample rare classes: each sample is drawn with probability proportional to count(class) ** -power.

    power=1 gives every class an equal share of each epoch, power=0 is plain shuffling.
    Epoch length stays equal to the dataset size.
    """
    labels = np.asarray(labels)
    per_class = class_counts(labels).astype(np.float64) ** -power
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(per_class[labels]), num_samples=len(labels),
                                 replacement=True, generator=generator)


def image_dir_for(base_dir, cache_short_side: int | None):
    """The cached downscaled image folder if requested, failing loudly when it hasn't been built yet."""
    path = resized_image_dir(base_dir, cache_short_side)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: uv run python scripts/resize_images.py "
                                f"--short-side {cache_short_side}")
    return path


def build_loaders(val_fold: int = 0, img_size: int = 224, batch_size: int = 32, num_workers: int = 4,
                  sampler_power: float = 0.0, seed: int = 0,
                  cache_short_side: int | None = None) -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders for one cross-validation fold.

    sampler_power > 0 switches the training loader from shuffling to class-balanced oversampling.
    cache_short_side reads the downscaled copies from scripts/resize_images.py instead of the originals.
    """
    train_df, val_df = split_fold(load_folds(), val_fold)
    image_dir = image_dir_for(TRAIN_IMAGE_DIR, cache_short_side)
    train_ds = SkinLesionDataset(train_df, image_dir, build_transforms(img_size, train=True))
    val_ds = SkinLesionDataset(val_df, image_dir, build_transforms(img_size, train=False))

    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    if sampler_power > 0:
        train_loader = DataLoader(train_ds, sampler=balanced_sampler(train_ds.labels, sampler_power, seed),
                                  drop_last=True, **common)
    else:
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, shuffle=True, generator=generator, drop_last=True, **common)
    # Validation has no augmentation, so a few workers keep up. Each worker is a separate process
    # with its own copy of torch, so this also saves RAM.
    val_workers = min(num_workers, 2)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=val_workers,
                            pin_memory=pin, persistent_workers=val_workers > 0)
    return train_loader, val_loader


def build_test_loader(img_size: int = 224, batch_size: int = 64, num_workers: int = 4,
                      cache_short_side: int | None = None) -> DataLoader:
    """The held-out ISIC 2018 Task 3 test set. Only for the final evaluation."""
    ds = SkinLesionDataset(load_test(), image_dir_for(TEST_IMAGE_DIR, cache_short_side),
                           build_transforms(img_size, train=False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalisation for display; returns values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(-1, 1, 1)
    return (images * std + mean).clamp(0, 1)
