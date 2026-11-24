"""Bundle a cross-validated config's fold models into one file for inference (default models/skinlesion.pt).

Takes the latest finished run of each fold and the temperature from scripts/evaluate.py, so run that first.
The weights are stored in float16 (about 8 MB per EfficientNet-B0 fold).

Usage:
    uv run python scripts/evaluate.py --name baseline --tta 8
    uv run python scripts/export_model.py --name baseline
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from skinlesion.data import OUTPUTS_DIR, ROOT
from skinlesion.evaluation import find_runs, load_model
from skinlesion.inference import DEFAULT_BUNDLE, save_bundle


def oof_summary(eval_dir: Path, tta_views: int) -> dict:
    """Per-class report and confusion matrix of the out-of-fold predictions, for the website's charts."""
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    from skinlesion.data import CLASSES
    from skinlesion.evaluation import PROB_COLUMNS, per_class_report

    path = eval_dir / (f"oof_tta{tta_views}_preds.csv" if tta_views > 1 else "oof_preds.csv")
    preds = pd.read_csv(path)
    labels, probs = preds.label.to_numpy(), preds[PROB_COLUMNS].to_numpy()
    report = per_class_report(labels, probs).round(4)
    return {"per_class": report.to_dict(orient="records"),
            "confusion": confusion_matrix(labels, probs.argmax(1), labels=list(range(len(CLASSES)))).tolist(),
            "n_images": int(len(preds)), "n_folds": int(preds.fold.nunique())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--temperature", type=float,
                        help="override the fitted temperature. The fit uses single-model out-of-fold predictions; "
                             "averaging the fold ensemble already removes much of that over-confidence")
    args = parser.parse_args()

    runs = find_runs(OUTPUTS_DIR / "runs", args.name)
    if not runs:
        raise SystemExit(f"no finished runs named {args.name!r}")
    eval_dir = OUTPUTS_DIR / "eval" / args.name
    calibration_path = eval_dir / "calibration.json"
    if not calibration_path.exists():
        raise SystemExit(f"{calibration_path.relative_to(ROOT)} missing; run scripts/evaluate.py --name {args.name}")
    calibration = json.loads(calibration_path.read_text())
    metrics = json.loads((eval_dir / "metrics.json").read_text())

    models, cfg = [], None
    for run_dir in runs.values():
        model, cfg = load_model(run_dir, torch.device("cpu"))
        models.append(model)

    metadata = {
        "name": args.name,
        "folds": {f: r.name for f, r in runs.items()},
        "exported": datetime.now().isoformat(timespec="seconds"),
        "oof": metrics.get("oof_tta", metrics["oof"]),
        "test": metrics.get("test"),
        "calibration": calibration,
        **oof_summary(eval_dir, calibration["tta_views"]),
    }
    temperature = calibration["temperature"] if args.temperature is None else args.temperature
    metadata["temperature_source"] = "fitted on out-of-fold predictions" if args.temperature is None else "set manually"
    save_bundle(args.out, models, cfg["model"], cfg["data"], temperature, metadata)
    size = args.out.stat().st_size / 2**20
    print(f"Saved {len(models)} x {cfg['model']['name']} to {args.out} ({size:.1f} MB), "
          f"temperature {temperature:.3f}")


if __name__ == "__main__":
    main()
