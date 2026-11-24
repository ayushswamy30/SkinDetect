"""Unit tests for metrics, class balancing, TTA, calibration and run discovery. No dataset or GPU needed."""

import numpy as np
import pytest
import torch
import yaml

from skinlesion.data import CLASSES
from skinlesion.dataset import class_weights
from skinlesion.evaluation import (apply_temperature, dihedral_views, expected_calibration_error, find_runs,
                                   fit_temperature, malignancy_metrics, per_class_report)
from skinlesion.metrics import compute_metrics

K = len(CLASSES)


def one_hot_probs(labels, eps=0.0):
    p = np.full((len(labels), K), eps / (K - 1))
    p[np.arange(len(labels)), labels] = 1 - eps
    return p


def test_perfect_predictions_score_one():
    labels = np.arange(K).repeat(3)
    m = compute_metrics(labels, one_hot_probs(labels, eps=0.1))
    assert m["balanced_acc"] == pytest.approx(1.0)
    assert m["macro_auc"] == pytest.approx(1.0)
    assert all(m[f"recall_{c}"] == 1.0 for c in CLASSES)


def test_balanced_accuracy_ignores_class_frequency():
    # 90 nv all correct, 10 mel all wrong -> accuracy 0.9 but balanced accuracy over the two present classes 0.5
    nv, mel = CLASSES.index("nv"), CLASSES.index("mel")
    labels = np.array([nv] * 90 + [mel] * 10)
    m = compute_metrics(labels, one_hot_probs(np.array([nv] * 100)))
    assert m["acc"] == pytest.approx(0.9)
    assert m["balanced_acc"] == pytest.approx(0.5)


def test_class_weights_average_one_per_sample():
    labels = np.array([0] * 10 + [5] * 90 + [6] * 5 + [1, 2, 3, 4])
    for power in (0.0, 0.5, 1.0):
        w = class_weights(labels, power).numpy()
        counts = np.bincount(labels, minlength=K)
        assert (w * counts).sum() / counts.sum() == pytest.approx(1.0, rel=1e-5)
    w = class_weights(labels, 1.0).numpy()
    assert w[6] > w[0] > w[5]  # rarer class -> larger weight
    assert np.allclose(class_weights(labels, 0.0).numpy(), 1.0)


def test_dihedral_views_are_distinct_and_validated():
    x = torch.arange(9.0).view(1, 1, 3, 3)
    views = dihedral_views(x, 8)
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8
    assert torch.equal(views[0], x)
    assert len(dihedral_views(x, 2)) == 2
    with pytest.raises(ValueError):
        dihedral_views(x, 3)


def test_temperature_scaling_keeps_argmax_and_identity():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(K), size=50)
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-9)
    q = apply_temperature(p, 2.5)
    assert np.allclose(q.sum(1), 1)
    assert (q.argmax(1) == p.argmax(1)).all()
    assert q.max(1).mean() < p.max(1).mean()  # T > 1 softens


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, size=(4000, K))
    true_t = 2.0
    calibrated = np.exp(logits / true_t)
    calibrated /= calibrated.sum(1, keepdims=True)
    labels = np.array([rng.choice(K, p=row) for row in calibrated])
    overconfident = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    t = fit_temperature(labels, overconfident)
    assert t == pytest.approx(true_t, rel=0.1)
    assert expected_calibration_error(labels, apply_temperature(overconfident, t)) < \
        expected_calibration_error(labels, overconfident)


def test_malignancy_metrics_counts_any_malignant_class_as_caught():
    mel, bcc, nv = CLASSES.index("mel"), CLASSES.index("bcc"), CLASSES.index("nv")
    labels = np.array([mel, mel, nv, nv])
    preds = np.array([bcc, nv, nv, mel])  # mel->bcc still caught; one nv falsely flagged
    m = malignancy_metrics(labels, one_hot_probs(preds, eps=0.01))
    assert m["malignant_sensitivity"] == pytest.approx(0.5)
    assert m["malignant_specificity"] == pytest.approx(0.5)


def test_per_class_report_shape():
    labels = np.arange(K).repeat(4)
    df = per_class_report(labels, one_hot_probs(labels, eps=0.2))
    assert list(df["class"]) == CLASSES
    assert (df.support == 4).all()


def test_find_runs_picks_latest_finished_run_per_fold(tmp_path):
    def make(name, fold, stamp, finished=True, cfg_name=None):
        d = tmp_path / f"{name}_fold{fold}_{stamp}"
        d.mkdir()
        (d / "config.yaml").write_text(yaml.safe_dump({"name": cfg_name or name, "data": {"val_fold": fold}}))
        if finished:
            (d / "summary.json").write_text("{}")
        return d

    make("base", 0, "20260101-000000")
    newer = make("base", 0, "20260102-000000")
    make("base", 0, "20260103-000000", finished=False)  # interrupted: ignored
    f1 = make("base", 1, "20260101-000000")
    make("base_large", 2, "20260101-000000")  # different config whose name shares the prefix
    assert find_runs(tmp_path, "base") == {0: newer, 1: f1}
