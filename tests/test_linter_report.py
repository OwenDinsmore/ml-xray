"""Tests for the Linter orchestrator, LintReport, and HTML/JSON rendering."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import synthetic
from ml_xray import Linter, Severity
from ml_xray.lint.base import default_checks, get_registry


def test_registry_has_all_builtin_checks():
    names = set(get_registry())
    assert {"leakage", "drift", "label_noise", "duplicates", "imbalance", "outliers"} <= names
    assert len(default_checks()) == len(names)


def test_report_bool_false_on_error():
    planted = synthetic.with_leaked_column()
    report = Linter(target="y").run(planted.df)
    assert report.counts()["ERROR"] >= 1
    assert bool(report) is False  # CI gate: any ERROR -> falsy


def test_report_bool_true_when_clean():
    planted = synthetic.clean_classification()
    report = Linter(target="y").run(planted.df)
    assert bool(report) is True


def test_worst_orders_by_severity():
    planted = synthetic.with_leaked_column()
    report = Linter(target="y").run(planted.df)
    worst = report.worst(5)
    severities = [f.severity for f in worst]
    assert severities == sorted(severities, reverse=True)


def test_run_with_split_detects_drift_and_overlap():
    planted = synthetic.with_train_test_overlap(dup=30)
    df = planted.df
    report = Linter(target="y").run(df, split=planted.split)
    checks_fired = {f.check for f in report}
    assert "leakage" in checks_fired


def test_to_dict_is_json_serializable():
    planted = synthetic.with_leaked_column()
    report = Linter(target="y").run(planted.df)
    d = report.to_dict()
    # Round-trips through JSON without error.
    text = json.dumps(d, default=str)
    assert "findings" in json.loads(text)


def test_to_html_writes_self_contained_file(tmp_path):
    planted = synthetic.with_leaked_column()
    report = Linter(target="y").run(planted.df)
    out = tmp_path / "report.html"
    report.to_html(str(out))
    html = out.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "ml-xray" in html
    # No external asset requests allowed.
    assert "http://" not in html
    assert "https://" not in html


def test_missing_target_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        Linter(target="nope").run(df)


def test_split_length_mismatch_raises():
    planted = synthetic.clean_classification(n=100)
    with pytest.raises(ValueError):
        Linter(target="y").run(planted.df, split=pd.Series(["train"] * 5))


def test_severity_from_name_roundtrip():
    assert Severity.from_name("error") is Severity.ERROR
    assert Severity.from_name("WARN") is Severity.WARN
    with pytest.raises(ValueError):
        Severity.from_name("bogus")


def test_reproducible_reports():
    planted = synthetic.with_label_noise(n=600, flip=30)
    r1 = Linter(target="y").run(planted.df, seed=0)
    r2 = Linter(target="y").run(planted.df, seed=0)
    assert r1.to_dict()["counts"] == r2.to_dict()["counts"]
