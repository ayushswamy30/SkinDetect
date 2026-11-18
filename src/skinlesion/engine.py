"""Model construction, the training step loop and batched prediction."""

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from skinlesion.data import CLASSES


def build_model(name: str, pretrained: bool = True, drop_rate: float = 0.0, drop_path_rate: float = 0.0) -> nn.Module:
    """Any timm backbone with a fresh 7-class head."""
    return timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES), drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate)


def autocast(device: torch.device, enabled: bool):
    # bfloat16 has float32's range, so unlike float16 it needs no gradient scaler
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, device: torch.device, amp: bool = True, desc: str = "train") -> float:
    """One pass over `loader`; the scheduler is stepped per batch. Returns the mean training loss."""
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for images, labels in bar:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with autocast(device, amp):
            loss = criterion(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(labels)
        n += len(labels)
        bar.set_postfix(loss=f"{total / n:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True,
            desc: str = "eval") -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities (N, C) and labels (N,) for every sample in `loader`, in loader order."""
    model.eval()
    probs, labels = [], []
    for images, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with autocast(device, amp):
            logits = model(images)
        probs.append(logits.float().softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()
