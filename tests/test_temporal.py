"""Tests for the temporal-leakage check: train rows dated after the test split
(future leakage) and a feature that is a monotonic proxy for the time column."""

from __future__ import annotations

import numpy as np
import pandas as pd

import synthetic
from ml_xray.lint import Severity
from ml_xray.lint.base import LintContext
from ml_xray.lint.checks.temporal import TemporalLeakageCheck


def _findings(planted):
    ctx = LintContext(
        df=planted.df, target=planted.target, split=planted.split, time=planted.meta["time"]
    )
    return TemporalLeakageCheck().run(ctx)


def test_train_test_time_overlap_identifies_rows():
    planted = synthetic.with_temporal_leakage(leak=40)
    overlap = [f for f in _findings(planted) if f.detail.get("kind") == "train_test_time_overlap"]
    assert overlap, "expected future-in-train leakage to be flagged"
    assert overlap[0].severity is Severity.ERROR
    assert set(planted.rows).issubset(set(overlap[0].rows))


def test_time_proxy_feature_flagged():
    planted = synthetic.with_temporal_leakage()
    proxy = [f for f in _findings(planted) if f.detail.get("kind") == "time_proxy_feature"]
    assert any(f.column == "proxy" for f in proxy)
    assert proxy[0].detail["abs_rho"] > 0.99


def test_no_time_column_is_noop():
    planted = synthetic.clean_classification()
    ctx = LintContext(df=planted.df, target=planted.target, time=None)
    assert TemporalLeakageCheck().run(ctx) == []


def test_proper_temporal_split_is_clean():
    # Train strictly before test, no proxy feature.
    n = 600
    ts = np.arange(n, dtype=float)
    split = np.where(ts < 400, "train", "test").astype(object)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "ts": ts, "y": rng.integers(0, 2, n)})
    ctx = LintContext(df=df, target="y", split=pd.Series(split), time="ts")
    overlap = [
        f
        for f in TemporalLeakageCheck().run(ctx)
        if f.detail.get("kind") == "train_test_time_overlap"
    ]
    assert overlap == []


def test_datetime_time_column_supported():
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    split = np.where(np.arange(n) < 200, "train", "test").astype(object)
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "ts": dates, "y": np.zeros(n, int)})
    # Move one train row into the future.
    df.loc[10, "ts"] = dates[-1]
    ctx = LintContext(df=df, target="y", split=pd.Series(split), time="ts")
    findings = TemporalLeakageCheck().run(ctx)
    assert any(f.detail.get("kind") == "train_test_time_overlap" for f in findings)


def test_time_col_override_on_constructor():
    planted = synthetic.with_temporal_leakage()
    ctx = LintContext(df=planted.df, target=planted.target, split=planted.split)  # no ctx.time
    findings = TemporalLeakageCheck(time_col="ts").run(ctx)
    assert any(f.detail.get("kind") == "train_test_time_overlap" for f in findings)
