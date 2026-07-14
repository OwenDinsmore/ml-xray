"""Tests for baseline snapshots and report diffing: JSON round-trip, new /
resolved / persisting / escalated classification, and the new-findings gate."""

from __future__ import annotations

import synthetic
from ml_xray import Linter, Severity, diff_reports
from ml_xray.lint.linter import LintReport


def _report(planted, **kw):
    return Linter(target="y").run(planted.df, **kw)


def test_json_roundtrip_preserves_findings(tmp_path):
    report = _report(synthetic.with_train_test_overlap(dup=40), split=None)
    path = tmp_path / "base.json"
    report.to_json(str(path))
    loaded = LintReport.from_json(str(path))
    assert len(loaded) == len(report)
    assert {f.fingerprint() for f in loaded} == {f.fingerprint() for f in report}
    assert {f.severity for f in loaded} == {f.severity for f in report}


def test_self_diff_is_empty():
    report = _report(synthetic.with_leaked_column())
    diff = diff_reports(report, report)
    assert not diff
    assert diff.counts()["new"] == 0
    assert diff.counts()["persisting"] == len(report)


def test_new_and_resolved_detected():
    # Baseline: a clean dataset. Current: a leaked column appears.
    baseline = _report(synthetic.clean_classification())
    current = _report(synthetic.with_leaked_column())
    diff = diff_reports(baseline, current)
    new_fps = {f.fingerprint() for f in diff.new}
    assert ("leakage", "leak", "numeric_corr") in new_fps
    assert not diff.is_clean(threshold=Severity.ERROR)  # a new ERROR exists
    assert diff.new_at_or_above(Severity.ERROR)


def test_resolved_when_problem_removed():
    baseline = _report(synthetic.with_leaked_column())
    current = _report(synthetic.clean_classification())
    diff = diff_reports(baseline, current)
    resolved_fps = {f.fingerprint() for f in diff.resolved}
    assert ("leakage", "leak", "numeric_corr") in resolved_fps


def test_escalation_detected():
    from ml_xray.lint.base import Finding

    base = LintReport(
        [Finding("drift", Severity.WARN, "x", column="c", detail={"kind": "numeric"})]
    )
    cur = LintReport(
        [Finding("drift", Severity.ERROR, "x", column="c", detail={"kind": "numeric"})]
    )
    diff = diff_reports(base, cur)
    assert len(diff.escalated) == 1
    assert diff.escalated[0][0].severity is Severity.WARN
    assert diff.escalated[0][1].severity is Severity.ERROR
    assert not diff.new and not diff.resolved


def test_gate_only_fails_on_new_not_preexisting():
    # Pre-existing ERROR debt in both -> not "new"; a fresh dataset adds no new ERROR.
    baseline = _report(synthetic.with_leaked_column())
    current = _report(synthetic.with_leaked_column())
    diff = diff_reports(baseline, current)
    assert diff.is_clean(threshold=Severity.ERROR)  # same leak, nothing new
