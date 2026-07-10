"""Tests for the leakage check: correlated leak, deterministic leak, cross-split
overlap. The clean dataset must not trigger a false positive."""

from __future__ import annotations

import synthetic
from ml_xray.lint import Severity
from ml_xray.lint.base import LintContext
from ml_xray.lint.checks.leakage import LeakageCheck


def _leakage_findings(planted):
    ctx = LintContext(df=planted.df, target=planted.target, split=planted.split)
    return LeakageCheck().run(ctx)


def test_numeric_leak_flagged_as_error():
    planted = synthetic.with_leaked_column()
    findings = _leakage_findings(planted)
    leak = [f for f in findings if f.column == "leak"]
    assert leak, "expected the leaked column to be flagged"
    assert leak[0].severity is Severity.ERROR
    assert leak[0].detail["abs_correlation"] > 0.99


def test_deterministic_leak_flagged():
    planted = synthetic.with_deterministic_leak()
    findings = _leakage_findings(planted)
    leak = [f for f in findings if f.column == "leak_cat"]
    assert leak, "expected the deterministic predictor to be flagged"
    assert leak[0].severity is Severity.ERROR
    assert leak[0].detail["kind"] == "deterministic_predictor"


def test_cross_split_overlap_identifies_rows():
    planted = synthetic.with_train_test_overlap(dup=40)
    findings = _leakage_findings(planted)
    overlap = [f for f in findings if f.detail.get("kind") == "cross_split_overlap"]
    assert overlap, "expected cross-split overlap to be flagged"
    finding = overlap[0]
    assert finding.severity is Severity.ERROR
    # Every planted duplicate row must be caught (recall == 1 on planted rows).
    planted_rows = set(planted.rows)
    assert planted_rows.issubset(set(finding.rows))


def test_clean_dataset_has_no_leakage():
    planted = synthetic.clean_classification()
    findings = _leakage_findings(planted)
    assert findings == []
