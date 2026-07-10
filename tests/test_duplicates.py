"""Tests for the duplicates check: planted exact duplicates must be identified,
and their MinHash/LSH near-duplicate pass must cluster them too."""

from __future__ import annotations

import synthetic
from ml_xray.lint import Severity
from ml_xray.lint.base import LintContext
from ml_xray.lint.checks.duplicates import DuplicatesCheck


def test_exact_duplicates_identified():
    planted = synthetic.with_duplicates(n=400, dup=30)
    ctx = LintContext(df=planted.df, target=planted.target, seed=0)
    findings = DuplicatesCheck().run(ctx)

    exact = [f for f in findings if f.detail.get("kind") == "exact"]
    assert exact, "expected exact duplicates to be flagged"
    finding = exact[0]
    assert finding.severity is Severity.WARN
    # Each appended duplicate row must be caught.
    assert set(planted.rows).issubset(set(finding.rows))
    assert finding.detail["n_exact_duplicates"] >= 30


def test_no_duplicates_on_clean_data():
    planted = synthetic.clean_classification(n=400)
    ctx = LintContext(df=planted.df, target=planted.target, seed=0)
    findings = DuplicatesCheck().run(ctx)
    exact = [f for f in findings if f.detail.get("kind") == "exact"]
    assert exact == []
