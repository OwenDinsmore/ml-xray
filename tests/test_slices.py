"""Tests for slice discovery: a planted weak region must rank first with a
negative delta and be significant; a clean model must surface no high-rank
false positives. Also covers binning, metric resolution, and BH correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
