"""Build the lesion-grouped, class-stratified cross-validation folds for HAM10000.

Split design:
  * All 10,015 HAM10000 images go into K folds (default 5) with StratifiedGroupKFold.
    Every image of a lesion lands in the same fold, so no lesion is ever seen in
    both training and validation. Class proportions are kept close to equal per fold.
  * Single experiments train on folds != 0 and validate on fold 0. Final models
    can use all K folds (cross-validation / ensembling).
  * The official ISIC 2018 Task 3 test set is the held-out test set. It is never
    used for training or model selection.

Writes data/processed/ham10000_folds.csv and data/processed/isic2018_test.csv.

Usage:
    uv run python scripts/make_splits.py [--folds 5] [--seed 42]
"""

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from skinlesion.data import CLASSES, FOLDS_CSV, PROCESSED_DIR, TEST_CSV, load_metadata, load_test_metadata

COLUMNS = ["image_id", "lesion_id", "dx", "label", "dx_type", "age", "sex", "localization", "dataset"]


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=df.index)
    for k, (_, val_idx) in enumerate(sgkf.split(df, y=df.label, groups=df.lesion_id)):
        folds.iloc[val_idx] = k
    return folds


def check(train: pd.DataFrame, test: pd.DataFrame, n_folds: int) -> None:
    assert (train.fold >= 0).all(), "some images were not assigned a fold"
    assert train.groupby("lesion_id").fold.nunique().max() == 1, "a lesion spans multiple folds"
    assert set(train.fold) == set(range(n_folds))
    for k in range(n_folds):
        assert set(train[train.fold == k].dx) == set(CLASSES), f"fold {k} is missing a class"
    assert not set(train.image_id) & set(test.image_id), "image overlap between HAM10000 and test"
    assert not set(train.lesion_id) & set(test.lesion_id), "lesion overlap between HAM10000 and test"


def summarize(train: pd.DataFrame, test: pd.DataFrame) -> None:
    pct = pd.crosstab(train.fold, train.dx, normalize="index").mul(100).round(1)
    pct["images"] = train.fold.value_counts().sort_index()
    pct["lesions"] = train.groupby("fold").lesion_id.nunique()
    overall = train.dx.value_counts(normalize=True).mul(100).round(1)
    print("Class % per fold (overall: " + ", ".join(f"{c} {overall[c]}" for c in CLASSES) + ")")
    print(pct.to_string())
    print("\nLesions per class per fold")
    print(pd.crosstab(train.fold, train.dx, values=train.lesion_id, aggfunc="nunique").to_string())
    print(f"\nTest (ISIC 2018 Task 3): {len(test)} images, {test.lesion_id.nunique()} lesions")
    print(test.dx.value_counts().reindex(CLASSES).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = load_metadata()
    train["fold"] = assign_folds(train, args.folds, args.seed)
    test = load_test_metadata()

    check(train, test, args.folds)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train[COLUMNS + ["fold"]].sort_values(["fold", "image_id"]).to_csv(FOLDS_CSV, index=False)
    test[COLUMNS].sort_values("image_id").to_csv(TEST_CSV, index=False)
    print(f"Wrote {FOLDS_CSV.relative_to(PROCESSED_DIR.parents[1])} and {TEST_CSV.relative_to(PROCESSED_DIR.parents[1])}\n")
    summarize(train, test)


if __name__ == "__main__":
    main()
