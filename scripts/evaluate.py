"""Evaluate a cross-validated config: per-fold scores, pooled out-of-fold metrics, TTA and the test set.

Reads the latest finished run of each fold (see scripts/run_cv.py) and writes to outputs/eval/<name>/:
  metrics.json            per-fold, mean/std and pooled out-of-fold metrics (plus TTA and test, if run)
  per_class_oof.csv       precision / recall / F1 / AUC / support per class
  confusion_oof.png       out-of-fold confusion matrix
  oof_preds.csv           out-of-fold class probabilities (TTA ones when --tta > 1)
  calibration.json        temperature fitted on the out-of-fold probabilities (used by scripts/export_model.py)
  test_*                  the same for the ISIC 2018 test set, with --test

Caveat: each fold's checkpoint was chosen by its validation balanced accuracy, so out-of-fold numbers are
slightly optimistic. The held-out test set is the unbiased estimate; keep it for comparing finished models,
not for tuning.

Usage:
    uv run python scripts/evaluate.py --name baseline
    uv run python scripts/evaluate.py --name baseline --tta 8
    uv run python scripts/evaluate.py --name baseline --tta 8 --test
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skinlesion.data import CLASSES, OUTPUTS_DIR, ROOT, TRAIN_IMAGE_DIR
from skinlesion.dataset import SkinLesionDataset, build_test_loader, build_transforms, image_dir_for, load_folds
from skinlesion.evaluation import (PROB_COLUMNS, apply_temperature, calibration_metrics, find_runs, fit_temperature,
                                   load_model, load_oof, malignancy_metrics, per_class_report, plot_confusion,
                                   predict_tta, save_json)
from skinlesion.metrics import compute_metrics, format_metrics

HEADLINE = ["balanced_acc", "acc", "macro_f1", "macro_auc"]


def all_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {**compute_metrics(labels, probs), **malignancy_metrics(labels, probs)}


def preds_frame(image_ids, labels, probs, **extra) -> pd.DataFrame:
    df = pd.DataFrame(probs, columns=PROB_COLUMNS)
    for i, (key, value) in enumerate({"image_id": image_ids, **extra, "label": labels}.items()):
        df.insert(i, key, value)
    return df


def report(title: str, m: dict[str, float]) -> None:
    print(f"\n{title}\n    {format_metrics(m)}\n    malignant: sensitivity {m['malignant_sensitivity']:.3f}  "
          f"specificity {m['malignant_specificity']:.3f}  auc {m['malignant_auc']:.4f}")


def write_outputs(out_dir, prefix: str, title: str, preds: pd.DataFrame) -> dict[str, float]:
    labels, probs = preds.label.to_numpy(), preds[PROB_COLUMNS].to_numpy()
    preds.to_csv(out_dir / f"{prefix}_preds.csv", index=False)
    per_class_report(labels, probs).to_csv(out_dir / f"per_class_{prefix}.csv", index=False, float_format="%.4f")
    plot_confusion(labels, probs, out_dir / f"confusion_{prefix}.png", title)
    return all_metrics(labels, probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="config name, e.g. baseline")
    parser.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8], help="dihedral TTA views")
    parser.add_argument("--test", action="store_true", help="also evaluate the fold ensemble on the ISIC 2018 test set")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    runs = find_runs(OUTPUTS_DIR / "runs", args.name)
    if not runs:
        raise SystemExit(f"no finished runs named {args.name!r} in outputs/runs")
    missing = sorted(set(range(5)) - set(runs))
    if missing:
        print(f"WARNING: folds {missing} have no finished run; results cover folds {list(runs)} only")
    out_dir = OUTPUTS_DIR / "eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {"name": args.name, "runs": {f: r.name for f, r in runs.items()}, "tta_views": args.tta}

    # Per fold and pooled out-of-fold, from the predictions saved during training (no TTA)
    oof = load_oof(runs)
    per_fold = {f: all_metrics(g.label.to_numpy(), g[PROB_COLUMNS].to_numpy()) for f, g in oof.groupby("fold")}
    table = pd.DataFrame(per_fold).T[HEADLINE + ["malignant_sensitivity"]]
    print("Per fold (saved validation predictions):")
    print(table.to_string(float_format="%.4f"))
    if len(table) > 1:
        print(f"mean  {'  '.join(f'{k} {table[k].mean():.4f}±{table[k].std():.4f}' for k in HEADLINE)}")
    results["per_fold"] = per_fold
    results["fold_mean"] = table.mean().to_dict()
    results["fold_std"] = table.std().to_dict() if len(table) > 1 else None
    results["oof"] = write_outputs(out_dir, "oof", f"{args.name}: out-of-fold ({len(oof):,} images)", oof)
    report(f"Pooled out-of-fold ({len(oof):,} images)", results["oof"])

    models = {}
    if args.tta > 1 or args.test:
        for fold, run_dir in runs.items():
            models[fold], cfg = load_model(run_dir, device)
        d = cfg["data"]

    if args.tta > 1:
        folds = load_folds()
        image_dir = image_dir_for(TRAIN_IMAGE_DIR, d["cache_short_side"])
        parts = []
        for fold, model in models.items():
            val_df = folds[folds.fold == fold].reset_index(drop=True)
            ds = SkinLesionDataset(val_df, image_dir, build_transforms(d["img_size"], train=False))
            loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                pin_memory=device.type == "cuda")
            probs, labels = predict_tta([model], loader, device, args.tta, desc=f"fold {fold} tta{args.tta}")
            parts.append(preds_frame(ds.image_ids, labels, probs, fold=fold))
        oof_tta = pd.concat(parts, ignore_index=True)
        results["oof_tta"] = write_outputs(out_dir, f"oof_tta{args.tta}",
                                           f"{args.name}: out-of-fold, TTA x{args.tta}", oof_tta)
        report(f"Pooled out-of-fold with TTA x{args.tta}", results["oof_tta"])
        oof = oof_tta

    # Temperature scaling, fitted on the out-of-fold predictions (with TTA when it was run, to match inference)
    oof_labels, oof_probs = oof.label.to_numpy(), oof[PROB_COLUMNS].to_numpy()
    temperature = fit_temperature(oof_labels, oof_probs)
    calibration = {"temperature": temperature, "tta_views": args.tta,
                   "oof_before": calibration_metrics(oof_labels, oof_probs),
                   "oof_after": calibration_metrics(oof_labels, apply_temperature(oof_probs, temperature))}
    print(f"\nCalibration: temperature {temperature:.3f}  out-of-fold NLL {calibration['oof_before']['nll']:.4f} -> "
          f"{calibration['oof_after']['nll']:.4f}  ECE {calibration['oof_before']['ece']:.4f} -> "
          f"{calibration['oof_after']['ece']:.4f}")

    if args.test:
        loader = build_test_loader(d["img_size"], args.batch_size, args.num_workers, d["cache_short_side"])
        probs, labels = predict_tta(list(models.values()), loader, device, args.tta, desc="test")
        calibration["test_before"] = calibration_metrics(labels, probs)
        probs = apply_temperature(probs, temperature)
        calibration["test_after"] = calibration_metrics(labels, probs)
        test = preds_frame(loader.dataset.image_ids, labels, probs)
        title = f"{args.name}: ISIC 2018 test, {len(models)}-fold ensemble, TTA x{args.tta}"
        results["test"] = write_outputs(out_dir, "test", title, test)
        report(f"ISIC 2018 test ({len(test):,} images), {len(models)}-model ensemble, TTA x{args.tta}",
               results["test"])

    results["calibration"] = calibration
    save_json(calibration, out_dir / "calibration.json")
    save_json(results, out_dir / "metrics.json")
    print(f"\nSaved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
