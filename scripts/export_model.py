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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
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
    }
    save_bundle(args.out, models, cfg["model"], cfg["data"], calibration["temperature"], metadata)
    size = args.out.stat().st_size / 2**20
    print(f"Saved {len(models)} x {cfg['model']['name']} to {args.out} ({size:.1f} MB), "
          f"temperature {calibration['temperature']:.3f}")


if __name__ == "__main__":
    main()
