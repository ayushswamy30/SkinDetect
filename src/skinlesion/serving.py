"""Lightweight serving without PyTorch: an ONNX Runtime ensemble predictor and shared result helpers.

Used by the deployed website so the server needs only onnxruntime, numpy and Pillow (a few hundred MB less
than PyTorch, which matters on small free-tier hosts). scripts/export_onnx.py writes the model directory:
    models/onnx/meta.json      classes, preprocessing, temperature, metrics
    models/onnx/fold<k>.onnx   input (N, 3, H, W) -> logits (N, 7) and class activation maps (N, 7, h, w)

For a backbone whose head is global average pooling followed by a linear layer (EfficientNet, ResNet),
Grad-CAM reduces exactly to CAM: the gradient of a class score with respect to the feature map is that
class's classifier weights divided by h*w at every position. So the maps are computed inside the graph and
no gradients are needed at serving time.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from skinlesion.data import CLASS_NAMES, CLASSES, MALIGNANT

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MALIGNANT_IDX = [CLASSES.index(c) for c in sorted(MALIGNANT)]


@dataclass
class Prediction:
    probs: dict[str, float]  # calibrated class probabilities, in CLASSES order
    top_class: str
    malignant_prob: float    # P(akiec) + P(bcc) + P(mel)

    @property
    def top_name(self) -> str:
        return CLASS_NAMES[self.top_class]

    @property
    def top_is_malignant(self) -> bool:
        return self.top_class in MALIGNANT


def calibrated_prediction(mean_probs: np.ndarray, temperature: float) -> Prediction:
    """Temperature-scale averaged probabilities (softmax(log p / T)) and package them."""
    logits = np.log(np.clip(mean_probs, 1e-12, None)) / temperature
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return Prediction(probs=dict(zip(CLASSES, p.astype(float).tolist())), top_class=CLASSES[int(p.argmax())],
                      malignant_prob=float(p[MALIGNANT_IDX].sum()))


def jet(x: np.ndarray) -> np.ndarray:
    """The classic 'jet' colour map for values in [0, 1] -> RGB floats in [0, 1] (no matplotlib needed)."""
    x = np.clip(x, 0, 1)[..., None]
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.concatenate([r, g, b], axis=-1)


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap onto a uint8 HxWx3 image."""
    blended = (1 - alpha) * image / 255.0 + alpha * jet(cam)
    return (blended.clip(0, 1) * 255).astype(np.uint8)


def dihedral_views_np(x: np.ndarray, n_views: int) -> np.ndarray:
    """Same views and order as skinlesion.evaluation.dihedral_views, for a (1, C, H, W) array."""
    if n_views not in (1, 2, 4, 8):
        raise ValueError("n_views must be 1, 2, 4 or 8")
    views = []
    for flip in (False, True):
        base = x[..., ::-1] if flip else x
        for k in range(4):
            views.append(np.rot90(base, k, axes=(-2, -1)))
    order = [0, 4, 2, 6, 1, 5, 3, 7]
    return np.ascontiguousarray(np.concatenate([views[i] for i in order[:n_views]]), dtype=np.float32)


class OnnxPredictor:
    """The fold ensemble on ONNX Runtime, with the same preprocessing, TTA and calibration as Predictor."""

    def __init__(self, model_dir: Path, tta_views: int = 8, threads: int | None = None):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        if meta["classes"] != CLASSES:
            raise ValueError(f"model classes {meta['classes']} differ from {CLASSES}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.sessions = [ort.InferenceSession(str(model_dir / f), opts, providers=["CPUExecutionProvider"])
                         for f in meta["files"]]
        self.models = self.sessions  # len() = ensemble size, like Predictor.models
        self.img_size = meta["img_size"]
        self.cache_short_side = meta["cache_short_side"]
        self.temperature = meta["temperature"]
        self.tta_views = tta_views
        self.metadata = {**meta["metadata"], "model_name": meta["model_name"]}

    def prepare(self, image: Image.Image) -> Image.Image:
        """Downscale like scripts/resize_images.py did for the training images."""
        image = image.convert("RGB")
        if self.cache_short_side is not None:
            scale = self.cache_short_side / min(image.size)
            if scale < 1:
                image = image.resize((round(image.width * scale), round(image.height * scale)),
                                     Image.Resampling.LANCZOS)
        return image

    def model_input_image(self, image: Image.Image) -> np.ndarray:
        """Resize the short side and centre-crop, following torchvision's Resize + CenterCrop rounding."""
        image = self.prepare(image)
        w, h = image.size
        s = self.img_size
        nw, nh = (s, int(s * h / w)) if w <= h else (int(s * w / h), s)
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        left, top = int(round((nw - s) / 2)), int(round((nh - s) / 2))
        return np.asarray(image.crop((left, top, left + s, top + s)))

    def _forward(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        x = (self.model_input_image(image).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        views = dihedral_views_np(x.transpose(2, 0, 1)[None], self.tta_views)
        probs, cams = np.zeros(len(CLASSES)), []
        for sess in self.sessions:
            logits, cam = sess.run(None, {"image": views})
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs += (e / e.sum(1, keepdims=True)).sum(0)
            cams.append(cam[0])  # the un-rotated view
        return probs / (self.tta_views * len(self.sessions)), np.mean(cams, axis=0)

    def analyze(self, image: Image.Image) -> tuple[Prediction, np.ndarray, np.ndarray]:
        """Prediction, the model's input crop (uint8 HxWx3) and the CAM of the predicted class in [0, 1]."""
        mean_probs, cams = self._forward(image)
        pred = calibrated_prediction(mean_probs, self.temperature)
        cam = np.maximum(cams[CLASSES.index(pred.top_class)], 0)
        cam = np.asarray(Image.fromarray((cam / max(cam.max(), 1e-8) * 255).astype(np.uint8))
                         .resize((self.img_size, self.img_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        return pred, self.model_input_image(image), cam

    def predict(self, image: Image.Image) -> Prediction:
        return self.analyze(image)[0]
