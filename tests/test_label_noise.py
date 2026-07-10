"""Tests for the label-noise check: planted label flips must be recovered with
good precision/recall, and a clean dataset must not raise many false positives."""

from __future__ import annotations

import synthetic
from ml_xray.lint.base import LintContext
from ml_xray.lint.checks.label_noise import LabelNoiseCheck


def test_flipped_labels_recovered():
    planted = synthetic.with_label_noise(n=800, flip=40, seed=5)
    ctx = LintContext(df=planted.df, target=planted.target, seed=0)
    findings = LabelNoiseCheck().run(ctx)
    assert findings, "expected label-noise findings"

    detected = set(findings[0].rows)
    planted_rows = set(planted.rows)

    tp = len(detected & planted_rows)
    recall = tp / len(planted_rows)
    precision = tp / max(len(detected), 1)
    # On two well-separated blobs, flips are highly detectable.
    assert recall >= 0.7, f"recall too low: {recall:.2f}"
    assert precision >= 0.6, f"precision too low: {precision:.2f}"


def test_clean_dataset_few_false_positives():
    planted = synthetic.with_label_noise(n=800, flip=0, seed=9)
    ctx = LintContext(df=planted.df, target=planted.target, seed=0)
    findings = LabelNoiseCheck().run(ctx)
    n_flagged = len(findings[0].rows) if findings else 0
    # With no planted noise, false positives should be a tiny fraction.
    assert n_flagged <= 0.02 * len(planted.df)


def test_regression_target_skipped():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    ctx = LintContext(df=df, target="y", task="regression")
    assert LabelNoiseCheck().run(ctx) == []
