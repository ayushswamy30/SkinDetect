"""Convert the model bundle to ONNX for PyTorch-free serving (see skinlesion.serving.OnnxPredictor).

Each fold becomes one .onnx file whose graph outputs the logits and the class activation maps for all seven
classes. That equals Grad-CAM exactly when the head is global average pooling + a linear layer, which is
checked before exporting. Then the ONNX ensemble is compared with the PyTorch one on a few images.

Usage:
    uv run python scripts/export_model.py --name baseline     # first: models/skinlesion.pt
    uv run python scripts/export_onnx.py                      # -> models/onnx/
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from skinlesion.data import CLASSES, ROOT
from skinlesion.inference import DEFAULT_BUNDLE, Predictor
from skinlesion.serving import OnnxPredictor


class LogitsAndCams(nn.Module):
    """Wraps a timm CNN: image -> (logits, per-class activation maps W_k . F)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        pool = getattr(model, "global_pool", None)
        head = model.get_classifier()
        if not isinstance(head, nn.Linear) or getattr(pool, "pool_type", None) != "avg":
            raise ValueError("CAM == Grad-CAM needs an average-pool + linear head; this model has "
                             f"{type(pool).__name__}/{type(head).__name__}")
        self.model = model
        self.register_buffer("weight", head.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.model.forward_features(x)
        logits = self.model.forward_head(feats)
        cams = torch.einsum("kc,nchw->nkhw", self.weight, feats)
        return logits, cams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "onnx")
    parser.add_argument("--check-images", type=Path, nargs="*",
                        default=sorted((ROOT / "web" / "static" / "assets" / "lesions").glob("*.jpg")))
    args = parser.parse_args()

    torch_pred = Predictor.from_bundle(args.bundle, tta_views=8, device=torch.device("cpu"))
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    s = torch_pred.img_size
    files = []
    for k, model in enumerate(torch_pred.models):
        name = f"fold{k}.onnx"
        torch.onnx.export(LogitsAndCams(model).eval(), (torch.zeros(1, 3, s, s),), args.out / name,
                          input_names=["image"], output_names=["logits", "cams"],
                          dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}, "cams": {0: "n"}},
                          opset_version=17, dynamo=False)
        files.append(name)
    meta = {
        "classes": CLASSES,
        "files": files,
        "model_name": torch_pred.metadata["model_name"],
        "img_size": s,
        "cache_short_side": torch_pred.cache_short_side,
        "temperature": torch_pred.temperature,
        "metadata": {k: v for k, v in torch_pred.metadata.items() if k != "model_name"},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Parity check against the PyTorch predictor
    onnx_pred = OnnxPredictor(args.out, tta_views=8)
    worst_p, worst_cam = 0.0, 1.0
    for path in args.check_images:
        img = Image.open(path)
        a, b = torch_pred.predict(img), onnx_pred.analyze(img)
        diff = max(abs(a.probs[c] - b[0].probs[c]) for c in CLASSES)
        cam_t = torch_pred.gradcam(img, CLASSES.index(a.top_class), max_models=len(torch_pred.models))
        corr = float(np.corrcoef(cam_t.ravel(), b[2].ravel())[0, 1])
        worst_p, worst_cam = max(worst_p, diff), min(worst_cam, corr)
        print(f"{path.name:<12} torch {a.top_class:<5} onnx {b[0].top_class:<5} max |dp| {diff:.4f}  cam corr {corr:.3f}")
    size = sum(f.stat().st_size for f in args.out.iterdir()) / 2**20
    print(f"\nSaved {len(files)} models to {args.out} ({size:.0f} MB). "
          f"Worst probability gap {worst_p:.4f}, worst heatmap correlation {worst_cam:.3f}")


if __name__ == "__main__":
    main()
