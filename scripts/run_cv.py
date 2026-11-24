"""Train every cross-validation fold of a config, one after another.

A fold is skipped when a finished run (one with summary.json) of the same config name already exists for it,
so an interrupted sweep can simply be restarted. Extra --set overrides are passed through to train.py;
use a different `name=` override for a variant so its runs aren't mistaken for the original's.

Usage:
    uv run python scripts/run_cv.py --config configs/baseline.yaml
    uv run python scripts/run_cv.py --config configs/baseline.yaml --folds 1 2 --set name=lr1e-4 optim.lr=1e-4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from skinlesion.data import OUTPUTS_DIR
from skinlesion.evaluation import find_runs

TRAIN = Path(__file__).with_name("train.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides passed to train.py")
    args = parser.parse_args()

    with open(args.config) as f:
        name = yaml.safe_load(f)["name"]
    for item in args.set:
        if item.startswith("name="):
            name = item.partition("=")[2]

    done = find_runs(OUTPUTS_DIR / "runs", name)
    for fold in args.folds:
        if fold in done:
            print(f"fold {fold}: already done ({done[fold].name}), skipping", flush=True)
            continue
        print(f"fold {fold}: training", flush=True)
        cmd = [sys.executable, str(TRAIN), "--config", args.config, "--set", *args.set, f"data.val_fold={fold}"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
