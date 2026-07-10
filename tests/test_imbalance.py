"""Tests for the imbalance check: a tiny minority class and single-value columns
must be flagged; a balanced target must not be."""

from __future__ import annotations

import synthetic
from ml_xray.lint import Severity
from ml_xray.lint.base import LintContext
from ml_xray.lint.checks.imbalance import ImbalanceCheck


def test_class_imbalance_flagged():
    planted = synthetic.with_imbalance(n=1000, minority=5)
    ctx = LintContext(df=planted.df, target=planted.target)
    findings = ImbalanceCheck().run(ctx)
    imbalance = [f for f in findings if f.detail.get("kind") == "class_imbalance"]
    assert imbalance, "expected class imbalance to be flagged"
    assert imbalance[0].severity is Severity.ERROR
    assert imbalance[0].detail["ratio"] >= 100


def test_single_value_column_flagged():
    planted = synthetic.with_outliers_and_nulls()
    ctx = LintContext(df=planted.df, target=planted.target)
    findings = ImbalanceCheck().run(ctx)
    single = [f for f in findings if f.detail.get("kind") == "single_value"]
    assert any(f.column == "constant" for f in single)


def test_balanced_target_not_flagged():
    planted = synthetic.clean_classification(n=600)
    ctx = LintContext(df=planted.df, target=planted.target)
    findings = ImbalanceCheck().run(ctx)
    imbalance = [f for f in findings if f.detail.get("kind") == "class_imbalance"]
    assert imbalance == []
