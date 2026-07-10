"""Tests for the outliers check: planted numeric outliers, a high-null column, a
constant column, and out-of-declared-range values must all be flagged."""

from __future__ import annotations

import numpy as np
import pandas as pd

import synthetic
from ml_xray.lint import Severity
from ml_xray.lint.base import LintContext
from ml_xray.lint.checks.outliers import OutliersCheck


def test_numeric_outliers_identify_rows():
    planted = synthetic.with_outliers_and_nulls()
    ctx = LintContext(df=planted.df, target=planted.target)
    findings = OutliersCheck().run(ctx)
    outliers = [f for f in findings if f.detail.get("kind") == "numeric_outlier"]
    assert any(f.column == "val" for f in outliers)
    val_finding = next(f for f in outliers if f.column == "val")
    assert set(planted.rows).issubset(set(val_finding.rows))


def test_high_null_column_flagged():
    planted = synthetic.with_outliers_and_nulls()
    ctx = LintContext(df=planted.df, target=planted.target)
    findings = OutliersCheck().run(ctx)
    high_null = [f for f in findings if f.detail.get("kind") == "high_null"]
    assert any(f.column == "mostly_null" for f in high_null)


def test_constant_column_flagged():
    planted = synthetic.with_outliers_and_nulls()
    ctx = LintContext(df=planted.df, target=planted.target)
    findings = OutliersCheck().run(ctx)
    constant = [f for f in findings if f.detail.get("kind") == "constant"]
    assert any(f.column == "constant" for f in constant)


def test_out_of_declared_range():
    df = pd.DataFrame({"age": [25, 30, -5, 250, 40], "y": [0, 1, 0, 1, 0]})
    ctx = LintContext(df=df, target="y")
    findings = OutliersCheck(ranges={"age": (0, 120)}).run(ctx)
    oob = [f for f in findings if f.detail.get("kind") == "out_of_range"]
    assert oob and oob[0].severity is Severity.ERROR
    assert set(oob[0].rows) == {2, 3}


def test_clean_numeric_has_no_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500), "y": rng.integers(0, 2, 500)})
    ctx = LintContext(df=df, target="y")
    findings = OutliersCheck().run(ctx)
    outliers = [f for f in findings if f.detail.get("kind") == "numeric_outlier"]
    # A clean Gaussian yields at most a negligible number of robust-z outliers.
    total = sum(f.detail["n_outliers"] for f in outliers)
    assert total <= 5
