"""Tests for slice discovery: a planted weak region must rank first with a
negative delta and be significant; a clean model must surface no high-rank
false positives. Also covers binning, metric resolution, and BH correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthetic
from ml_xray.slices import (
    Slice,
    SliceFinder,
    benjamini_hochberg,
    bin_feature,
    resolve_metric,
)


def test_weak_region_ranks_first():
    planted = synthetic.weak_region_predictions()
    finder = SliceFinder(metric="accuracy", min_support=50, max_depth=2).fit(
        planted.X, planted.y_true, planted.y_pred
    )
    report = finder.report()
    assert report.slices, "expected at least one underperforming slice"

    top = report.slices[0]
    # The planted weak predicate must be the (or part of the) top slice.
    assert top.predicate.get("region") == "EU"
    assert top.delta < 0  # underperforms the baseline (higher-is-better metric)
    assert top.p_value < 0.05
    assert top.metric_value < top.baseline


def test_clean_model_no_high_rank_false_positive():
    planted = synthetic.clean_predictions()
    finder = SliceFinder(metric="accuracy", min_support=50).fit(
        planted.X, planted.y_true, planted.y_pred
    )
    report = finder.report()
    # A uniformly-erroring model should not produce strong, significant slices.
    strong = [s for s in report.slices if abs(s.delta) > 0.1]
    assert strong == []


def test_slice_report_ranking_is_descending():
    planted = synthetic.weak_region_predictions()
    report = (
        SliceFinder(metric="accuracy", min_support=50)
        .fit(planted.X, planted.y_true, planted.y_pred)
        .report()
    )
    harms = [abs(s.delta) * np.log(max(s.support, 2)) for s in report.slices]
    assert harms == sorted(harms, reverse=True)


def test_top_k_respected():
    planted = synthetic.weak_region_predictions()
    report = (
        SliceFinder(metric="accuracy", min_support=30, top_k=3)
        .fit(planted.X, planted.y_true, planted.y_pred)
        .report()
    )
    assert len(report.slices) <= 3


def test_regression_metric_auto():
    rng = np.random.default_rng(0)
    n = 1500
    group = rng.choice(["a", "b"], size=n)
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.1, n)
    # Group 'b' has large errors -> underperforms on MSE (lower is better).
    bad = group == "b"
    y_pred[bad] = y_true[bad] + rng.normal(0, 3, bad.sum())
    X = pd.DataFrame({"group": group})
    report = (
        SliceFinder(metric="auto", min_support=50)
        .fit(X, y_true, y_pred, task="regression")
        .report()
    )
    assert report.metric == "mse"
    assert report.slices
    assert report.slices[0].predicate.get("group") == "b"
    assert report.slices[0].delta > 0  # higher MSE than baseline


def test_bin_feature_quantile_labels():
    s = pd.Series(np.arange(100, dtype=float))
    binned = bin_feature(s, strategy="quantile", n_bins=4)
    assert binned.nunique() == 4
    assert binned.isna().sum() == 0


def test_resolve_metric_directions():
    assert resolve_metric("accuracy", None).greater_is_better is True
    assert resolve_metric("mse", None).greater_is_better is False
    assert resolve_metric("auto", "classification").name == "accuracy"
    assert resolve_metric("auto", "regression").name == "mse"
    assert resolve_metric("roc_auc", None).needs_proba is True
    assert resolve_metric("log_loss", None).greater_is_better is False


def _proba_predictions(seed=0):
    rng = np.random.default_rng(seed)
    n = 3000
    region = rng.choice(["EU", "US", "APAC"], size=n)
    y = rng.integers(0, 2, n)
    proba = np.where(y == 1, rng.uniform(0.6, 0.95, n), rng.uniform(0.05, 0.4, n))
    eu = region == "EU"
    proba[eu] = rng.uniform(0.4, 0.6, eu.sum())  # uninformative in the weak region
    y_pred = (proba > 0.5).astype(int)
    return pd.DataFrame({"region": region}), y, y_pred, proba


def test_roc_auc_metric_finds_weak_region():
    X, y, y_pred, proba = _proba_predictions()
    report = SliceFinder(metric="roc_auc", min_support=50).fit(X, y, y_pred, proba).report()
    assert report.metric == "roc_auc"
    assert report.slices[0].predicate.get("region") == "EU"
    assert report.slices[0].delta < 0  # lower AUC than baseline


def test_log_loss_metric_direction():
    X, y, y_pred, proba = _proba_predictions()
    report = SliceFinder(metric="log_loss", min_support=50).fit(X, y, y_pred, proba).report()
    assert report.metric == "log_loss"
    assert report.slices[0].predicate.get("region") == "EU"
    assert report.slices[0].delta > 0  # higher loss than baseline


def test_roc_auc_requires_proba():
    X, y, y_pred, _ = _proba_predictions()
    finder = SliceFinder(metric="roc_auc", min_support=50)
    with pytest.raises(ValueError):
        finder.fit(X, y, y_pred)  # no y_proba supplied


def test_benjamini_hochberg_monotone():
    p = np.array([0.001, 0.02, 0.04, 0.5, 0.9])
    mask = benjamini_hochberg(p, alpha=0.05)
    # The two smallest should survive; the largest should not.
    assert mask[0]
    assert not mask[-1]
    assert benjamini_hochberg(np.array([]), 0.05).size == 0


def test_slice_dataclass_describe():
    s = Slice({"region": "EU", "tenure": "<3mo"}, 1204, 0.42, 0.79, -0.37, 0.001)
    assert s.describe() == "region=EU & tenure=<3mo"


def _numeric_weak_region(seed=0):
    rng = np.random.default_rng(seed)
    n = 3000
    tenure = rng.uniform(0, 24, n)
    y = rng.integers(0, 2, n)
    y_pred = y.copy()
    weak = tenure < 6  # spans ~2 quantile bins
    flip = weak & (rng.uniform(size=n) < 0.55)
    y_pred[flip] = 1 - y_pred[flip]
    return pd.DataFrame({"tenure": tenure}), y, y_pred


def test_numeric_range_predicate_captures_weak_region():
    X, y, y_pred = _numeric_weak_region()
    report = SliceFinder(metric="accuracy", min_support=50).fit(X, y, y_pred).report()
    top = report.slices[0]
    assert "tenure" in top.predicate
    # A contiguous-range predicate reads like "< x" or "[lo, hi)".
    label = top.predicate["tenure"]
    assert label.startswith("<") or label.startswith("[") or label.startswith(">=")
    assert top.delta < 0


def test_numeric_ranges_off_uses_single_bins():

    X, y, y_pred = _numeric_weak_region()
    report = (
        SliceFinder(metric="accuracy", min_support=50, numeric_ranges=False)
        .fit(X, y, y_pred)
        .report()
    )
    assert report.slices
    # Single-bin labels are full intervals, not open-ended "< x" ranges.
    assert not report.slices[0].predicate["tenure"].startswith("<")


def test_numeric_range_items_are_contiguous_and_labeled():
    s = pd.Series(np.arange(100, dtype=float))
    items = numeric_range_items_helper(s)
    labels = [lbl for lbl, _ in items]
    assert any(lbl.startswith("< ") for lbl in labels)
    assert any(lbl.startswith(">= ") for lbl in labels)
    # No item selects every row (the whole-column range is excluded).
    assert all(mask.sum() < len(s) for _, mask in items)


def numeric_range_items_helper(s):
    from ml_xray.slices.binning import numeric_range_items

    return numeric_range_items(s, n_bins=4, min_support=1)
