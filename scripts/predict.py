"""Classify dermoscopy images with the exported model bundle, optionally saving Grad-CAM overlays.

Research use only: this is not a medical device and must not be used for diagnosis.

Usage:
    uv run python scripts/predict.py path/to/lesion.jpg [more.jpg ...]
    uv run python scripts/predict.py images/*.jpg --cam outputs/cam --tta 8
"""

import argparse
from pathlib import Path

from PIL import Image

from skinlesion.data import CLASSES
from skinlesion.inference import DEFAULT_BUNDLE, Predictor, overlay_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--tta", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--cam", type=Path, help="directory for Grad-CAM overlays of the predicted class")
    args = parser.parse_args()

    predictor = Predictor.from_bundle(args.bundle, tta_views=args.tta)
    if args.cam:
        args.cam.mkdir(parents=True, exist_ok=True)
    print(f"{'image':<28} {'prediction':<8} {'conf':>5} {'P(malig.)':>9}   " + " ".join(f"{c:>6}" for c in CLASSES))
    for path in args.images:
        image = Image.open(path)
        pred = predictor.predict(image)
        print(f"{path.name:<28} {pred.top_class:<8} {pred.probs[pred.top_class]:>5.2f} {pred.malignant_prob:>9.2f}   "
              + " ".join(f"{pred.probs[c]:>6.3f}" for c in CLASSES))
        if args.cam:
            crop = predictor.model_input_image(image)
            cam = predictor.gradcam(image, CLASSES.index(pred.top_class))
            Image.fromarray(overlay_heatmap(crop, cam)).save(args.cam / f"{path.stem}_cam_{pred.top_class}.png")
    if args.cam:
        print(f"Grad-CAM overlays saved to {args.cam}")


if __name__ == "__main__":
    main()
