"""Inference on new images: a single-file model bundle, an ensemble predictor with TTA, and Grad-CAM heatmaps.

A bundle (written by scripts/export_model.py) holds every fold's weights in float16 plus everything needed to
reproduce evaluation preprocessing, so the app and CLI need neither the training runs nor the dataset.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from skinlesion.data import CLASSES, ROOT
from skinlesion.dataset import build_transforms
from skinlesion.engine import autocast, build_model
from skinlesion.evaluation import dihedral_views
from skinlesion.serving import Prediction, calibrated_prediction, overlay_heatmap  # noqa: F401 (re-exported)

DEFAULT_BUNDLE = ROOT / "models" / "skinlesion.pt"


def save_bundle(path: Path, models: list[nn.Module], model_cfg: dict, data_cfg: dict, temperature: float,
                metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": 1,
        "classes": CLASSES,
        "model": {"name": model_cfg["name"], "drop_rate": model_cfg["drop_rate"],
                  "drop_path_rate": model_cfg["drop_path_rate"]},
        "img_size": data_cfg["img_size"],
        "cache_short_side": data_cfg["cache_short_side"],
        "temperature": temperature,
        "state_dicts": [{k: v.half() if v.is_floating_point() else v for k, v in m.state_dict().items()}
                        for m in models],
        "metadata": metadata,
    }, path)


class Predictor:
    """An ensemble of fold models with dihedral TTA and temperature scaling."""

    def __init__(self, models: list[nn.Module], img_size: int, cache_short_side: int | None = None,
                 temperature: float = 1.0, tta_views: int = 8, device: torch.device | None = None,
                 metadata: dict | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [m.float().to(self.device).eval() for m in models]
        self.img_size = img_size
        self.cache_short_side = cache_short_side
        self.temperature = temperature
        self.tta_views = tta_views
        self.metadata = metadata or {}
        self.transform = build_transforms(img_size, train=False)

    @classmethod
    def from_bundle(cls, path: Path = DEFAULT_BUNDLE, tta_views: int = 8, device: torch.device | None = None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build it with: uv run python scripts/export_model.py")
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b["classes"] != CLASSES:
            raise ValueError(f"bundle classes {b['classes']} differ from {CLASSES}")
        models = []
        for sd in b["state_dicts"]:
            m = build_model(b["model"]["name"], pretrained=False, drop_rate=b["model"]["drop_rate"],
                            drop_path_rate=b["model"]["drop_path_rate"])
            m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
            models.append(m)
        metadata = {**b["metadata"], "model_name": b["model"]["name"]}
        return cls(models, b["img_size"], b["cache_short_side"], b["temperature"], tta_views, device, metadata)

    def prepare(self, image: Image.Image) -> Image.Image:
        """Downscale like scripts/resize_images.py did for the training images, so the model sees the same scale."""
        image = image.convert("RGB")
        if self.cache_short_side is not None:
            scale = self.cache_short_side / min(image.size)
            if scale < 1:
                image = image.resize((round(image.width * scale), round(image.height * scale)),
                                     Image.Resampling.LANCZOS)
        return image

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(self.prepare(image)).unsqueeze(0).to(self.device)

    def model_input_image(self, image: Image.Image) -> np.ndarray:
        """The centre crop the model actually sees, as uint8 HxWx3, for overlaying heatmaps."""
        image = self.prepare(image)
        scale = self.img_size / min(image.size)
        w, h = round(image.width * scale), round(image.height * scale)
        image = image.resize((w, h), Image.Resampling.BILINEAR)
        left, top = (w - self.img_size) // 2, (h - self.img_size) // 2
        return np.asarray(image.crop((left, top, left + self.img_size, top + self.img_size)))

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        x = self.to_tensor(image)
        # All TTA views of the one image go through each model as a single batch
        views = torch.cat(dihedral_views(x, self.tta_views)).contiguous(memory_format=torch.channels_last)
        probs = torch.zeros(1, len(CLASSES), device=self.device)
        for model in self.models:
            with autocast(self.device, True):
                probs += model(views).float().softmax(1).sum(0, keepdim=True)
        probs = probs / (self.tta_views * len(self.models))
        # temperature scaling on the averaged probabilities, as fitted in scripts/evaluate.py
        return calibrated_prediction(probs[0].double().cpu().numpy(), self.temperature)

    def analyze(self, image: Image.Image) -> tuple[Prediction, np.ndarray, np.ndarray]:
        """Prediction, the model's input crop and the Grad-CAM of the predicted class (same API as OnnxPredictor)."""
        pred = self.predict(image)
        return pred, self.model_input_image(image), self.gradcam(image, CLASSES.index(pred.top_class))

    def gradcam(self, image: Image.Image, class_idx: int | None = None, max_models: int = 2) -> np.ndarray:
        """Grad-CAM for `class_idx` (default: the predicted class), averaged over the first `max_models` models.

        The fold models' maps are very similar, so a couple of them is enough and keeps CPU latency down.

        Returns an (img_size, img_size) map in [0, 1] aligned with model_input_image(). It shows which regions
        pushed the score up; it is a rough guide to what the network attends to, not a segmentation.
        """
        x = self.to_tensor(image)
        if class_idx is None:
            class_idx = CLASSES.index(self.predict(image).top_class)
        cams = []
        for model in self.models[:max_models]:
            with torch.enable_grad():
                feats = model.forward_features(x)  # (1, C, h, w), the last convolutional feature map
                score = model.forward_head(feats)[0, class_idx]
                # gradient w.r.t. the feature map only: parameter .grad is never touched
                (grad,) = torch.autograd.grad(score, feats)
            weights = grad.mean((2, 3), keepdim=True)
            cam = torch.relu((weights * feats).sum(1, keepdim=True)).detach()
            cam = nn.functional.interpolate(cam, size=(self.img_size, self.img_size), mode="bilinear",
                                            align_corners=False)[0, 0]
            cams.append(cam / cam.max().clamp_min(1e-8))
        cam = torch.stack(cams).mean(0)
        return (cam / cam.max().clamp_min(1e-8)).cpu().numpy()

