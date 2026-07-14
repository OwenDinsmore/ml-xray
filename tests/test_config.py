"""Tests for LintConfig: check selection, disable, severity remapping/ignore,
check_args passthrough, and TOML loading (file + pyproject + discovery)."""

from __future__ import annotations

import pytest

import synthetic
from ml_xray import LintConfig, Linter, Severity


def test_check_selection_and_disable():
    cfg = LintConfig(checks=["leakage", "duplicates", "outliers"], disable=["outliers"])
    names = [c.name for c in cfg.build_checks()]
    assert names == ["duplicates", "leakage"]


def test_unknown_check_raises():
    with pytest.raises(ValueError):
        LintConfig(checks=["nope"]).build_checks()


def test_severity_override_downgrade_makes_gate_pass():
    planted = synthetic.with_train_test_overlap(dup=40)
    # The train/test overlap is normally ERROR -> gate fails.
    strict = Linter(target="y").run(planted.df, split=planted.split)
    assert bool(strict) is False

    cfg = LintConfig(severity={"leakage.cross_split_overlap": "warn"})
    lenient = Linter(target="y", config=cfg).run(planted.df, split=planted.split)
    assert bool(lenient) is True  # downgraded to WARN
    overlap = [f for f in lenient if f.detail.get("kind") == "cross_split_overlap"]
    assert overlap and overlap[0].severity is Severity.WARN


def test_severity_ignore_drops_findings():
    planted = synthetic.with_imbalance()
    cfg = LintConfig(checks=["imbalance"], severity={"imbalance.class_imbalance": "ignore"})
    report = Linter(target="y", config=cfg).run(planted.df)
    kinds = {f.detail.get("kind") for f in report}
    assert "class_imbalance" not in kinds


def test_kind_rule_beats_bare_check_rule():
    planted = synthetic.with_duplicates(n=400, dup=30)
    cfg = LintConfig(
        checks=["duplicates"],
        severity={"duplicates": "error", "duplicates.near": "ignore"},
    )
    report = Linter(target="y", config=cfg).run(planted.df)
    kinds = {f.detail.get("kind"): f.severity for f in report}
    assert "near" not in kinds  # ignored by the more specific rule
    assert kinds["exact"] is Severity.ERROR  # bare rule applied


def test_check_args_passthrough():
    import pandas as pd

    df = pd.DataFrame({"age": [25, -5, 250, 30], "y": [0, 1, 0, 1]})
    cfg = LintConfig(checks=["outliers"], check_args={"outliers": {"ranges": {"age": (0, 120)}}})
    report = Linter(target="y", config=cfg).run(df)
    oob = [f for f in report if f.detail.get("kind") == "out_of_range"]
    assert oob and set(oob[0].rows) == {1, 2}


def test_from_toml_file(tmp_path):
    (tmp_path / "ml-xray.toml").write_text(
        'checks = ["leakage", "drift"]\ndisable = ["drift"]\nseed = 9\n'
        '[severity]\nleakage = "warn"\n'
    )
    cfg = LintConfig.from_toml(tmp_path / "ml-xray.toml")
    assert cfg.checks == ["leakage", "drift"]
    assert cfg.disable == ["drift"]
    assert cfg.seed == 9
    assert cfg.severity == {"leakage": "warn"}
    assert [c.name for c in cfg.build_checks()] == ["leakage"]


def test_from_pyproject_table(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ml-xray]\nchecks = ["imbalance"]\n[tool.ml-xray.severity]\nimbalance = "info"\n'
    )
    cfg = LintConfig.from_toml(tmp_path / "pyproject.toml")
    assert cfg.checks == ["imbalance"]
    assert cfg.severity == {"imbalance": "info"}


def test_discover_finds_ml_xray_toml(tmp_path):
    (tmp_path / "ml-xray.toml").write_text('checks = ["leakage"]\n')
    sub = tmp_path / "nested" / "deeper"
    sub.mkdir(parents=True)
    cfg = LintConfig.discover(sub)
    assert cfg.checks == ["leakage"]


def test_discover_returns_default_when_absent(tmp_path):
    cfg = LintConfig.discover(tmp_path)
    assert cfg.checks is None
    assert len(cfg.build_checks()) == 7
