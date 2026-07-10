"""Tests for the ``ml-xray`` CLI: the lint gate exit codes and HTML/JSON output,
plus the not-yet-implemented subcommands."""

from __future__ import annotations

import synthetic
from ml_xray.cli import main


def _write_csv(planted, path, with_split=False):
    df = planted.df.copy()
    if with_split and planted.split is not None:
        df["split"] = planted.split.to_numpy()
    df.to_csv(path, index=False)
    return path


def test_lint_gate_fails_on_error(tmp_path, capsys):
    planted = synthetic.with_leaked_column()
    csv = _write_csv(planted, tmp_path / "data.csv")
    code = main(["lint", str(csv), "--target", "y", "--fail-on", "error"])
    assert code == 1  # gate fails because a leak is ERROR
    assert "gate failed" in capsys.readouterr().err


def test_lint_gate_passes_on_clean(tmp_path):
    planted = synthetic.clean_classification()
    csv = _write_csv(planted, tmp_path / "clean.csv")
    code = main(["lint", str(csv), "--target", "y", "--fail-on", "error"])
    assert code == 0


def test_lint_writes_html_and_json(tmp_path):
    planted = synthetic.with_leaked_column()
    csv = _write_csv(planted, tmp_path / "data.csv")
    html = tmp_path / "out.html"
    js = tmp_path / "out.json"
    code = main(["lint", str(csv), "--target", "y", "--html", str(html), "--json", str(js)])
    assert code == 0  # no --fail-on, so exit stays 0
    assert html.exists() and js.exists()
    assert "ml-xray" in html.read_text(encoding="utf-8")


def test_lint_with_split_column(tmp_path):
    planted = synthetic.with_drift()
    csv = _write_csv(planted, tmp_path / "drift.csv", with_split=True)
    code = main(["lint", str(csv), "--target", "y", "--split", "split"])
    assert code == 0


def test_unknown_split_column_errors(tmp_path, capsys):
    planted = synthetic.clean_classification()
    csv = _write_csv(planted, tmp_path / "data.csv")
    code = main(["lint", str(csv), "--target", "y", "--split", "missing"])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_slices_not_implemented(capsys):
    assert main(["slices", "preds.csv"]) == 3
    assert "not implemented" in capsys.readouterr().err


def test_embed_diff_not_implemented(capsys):
    assert main(["embed-diff", "a.npy", "b.npy"]) == 3
    assert "not implemented" in capsys.readouterr().err
