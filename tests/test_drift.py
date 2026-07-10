"""Tests for the drift check: a shifted test-split feature must be flagged while
a stable feature is not."""

from __future__ import annotations

import numpy as np

import synthetic
from ml_xray.lint import Severity
from ml_xray.lint.base import LintContext
from ml_xray.lint.checks.drift import DriftCheck, psi


def test_drifted_column_flagged_not_stable():
    planted = synthetic.with_drift()
    ctx = LintContext(df=planted.df, target=planted.target, split=planted.split)
    findings = DriftCheck().run(ctx)

    flagged = {f.column for f in findings}
    assert "drifted" in flagged
    assert "stable" not in flagged

    drifted = next(f for f in findings if f.column == "drifted")
    assert drifted.severity is Severity.ERROR
    assert drifted.detail["psi"] > 0.25
    assert drifted.detail["ks_pvalue"] < 0.05


def test_no_drift_without_split():
    planted = synthetic.with_drift()
    ctx = LintContext(df=planted.df, target=planted.target, split=None)
    assert DriftCheck().run(ctx) == []


def test_psi_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert psi(x, x) < 1e-6


def test_psi_positive_for_shifted_samples():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(3, 1, 1000)
    assert psi(a, b) > 0.25
