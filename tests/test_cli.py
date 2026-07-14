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


def test_lint_with_config_downgrades_gate(tmp_path):
    planted = synthetic.with_leaked_column()
    csv = _write_csv(planted, tmp_path / "data.csv")
    (tmp_path / "ml-xray.toml").write_text('[severity]\nleakage = "warn"\n')
    # Without config the leak is ERROR -> gate fails; with config it's WARN -> passes.
    assert main(["lint", str(csv), "--target", "y", "--fail-on", "error"]) == 1
    code = main(
        [
            "lint",
            str(csv),
            "--target",
            "y",
            "--fail-on",
            "error",
            "--config",
            str(tmp_path / "ml-xray.toml"),
        ]
    )
    assert code == 0


def test_lint_save_and_use_baseline(tmp_path, capsys):
    # Save a baseline on a clean dataset.
    clean = _write_csv(synthetic.clean_classification(), tmp_path / "clean.csv")
    base = tmp_path / "base.json"
    assert main(["lint", str(clean), "--target", "y", "--save-baseline", str(base)]) == 0
    assert base.exists()

    # A leaked dataset diffed against the clean baseline -> a NEW error -> gate fails.
    leaked = _write_csv(synthetic.with_leaked_column(), tmp_path / "leaked.csv")
    code = main(
        [
            "lint",
            str(leaked),
            "--target",
            "y",
            "--baseline",
            str(base),
            "--fail-on-new",
            "error",
        ]
    )
    out = capsys.readouterr().out
    assert "vs baseline:" in out
    assert code == 1


def test_fail_on_new_requires_baseline(tmp_path, capsys):
    csv = _write_csv(synthetic.clean_classification(), tmp_path / "data.csv")
    code = main(["lint", str(csv), "--target", "y", "--fail-on-new", "error"])
    assert code == 2
    assert "requires --baseline" in capsys.readouterr().err


def test_lint_baseline_no_new_findings_passes(tmp_path):
    leaked = _write_csv(synthetic.with_leaked_column(), tmp_path / "leaked.csv")
    base = tmp_path / "base.json"
    main(["lint", str(leaked), "--target", "y", "--save-baseline", str(base)])
    # Same dataset vs its own baseline: pre-existing debt, nothing new -> passes.
    code = main(
        [
            "lint",
            str(leaked),
            "--target",
            "y",
            "--baseline",
            str(base),
            "--fail-on-new",
            "error",
        ]
    )
    assert code == 0


def _write_preds(path):
    planted = synthetic.weak_region_predictions()
    df = planted.X.copy()
    df["y_true"] = planted.y_true
    df["y_pred"] = planted.y_pred
    df.to_csv(path, index=False)
    return path


def test_slices_command(tmp_path, capsys):
    preds = _write_preds(tmp_path / "preds.csv")
    html = tmp_path / "slices.html"
    code = main(
        ["slices", str(preds), "--metric", "accuracy", "--min-support", "50", "--html", str(html)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "underperforming slices" in out
    assert "region=EU" in out
    assert html.exists()


def test_slices_missing_columns_errors(tmp_path, capsys):
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    code = main(["slices", str(tmp_path / "bad.csv")])
    assert code == 2
    assert "y_true" in capsys.readouterr().err


def test_embed_diff_command(tmp_path, capsys):
    import numpy as np

    planted = synthetic.partially_perturbed_spaces(n=150, moved=15)
    np.save(tmp_path / "a.npy", planted.emb_a)
    np.save(tmp_path / "b.npy", planted.emb_b)
    html = tmp_path / "embed.html"
    code = main(
        [
            "embed-diff",
            str(tmp_path / "a.npy"),
            str(tmp_path / "b.npy"),
            "-k",
            "8",
            "--html",
            str(html),
        ]
    )
    assert code == 0
    assert "neighbor_overlap" in capsys.readouterr().out
    assert html.exists()


def test_report_combines_sections(tmp_path):
    import numpy as np

    lint_csv = _write_csv(synthetic.with_leaked_column(), tmp_path / "data.csv")
    preds = _write_preds(tmp_path / "preds.csv")
    planted = synthetic.partially_perturbed_spaces(n=120, moved=12)
    np.save(tmp_path / "a.npy", planted.emb_a)
    np.save(tmp_path / "b.npy", planted.emb_b)
    out = tmp_path / "combined.html"
    code = main(
        [
            "report",
            "--lint",
            str(lint_csv),
            "--target",
            "y",
            "--slices",
            str(preds),
            "--embed",
            str(tmp_path / "a.npy"),
            str(tmp_path / "b.npy"),
            "--html",
            str(out),
        ]
    )
    assert code == 0
    html = out.read_text(encoding="utf-8")
    assert "Lint findings" in html
    assert "Underperforming slices" in html
    assert "Embedding diff" in html


def test_report_requires_a_section(tmp_path, capsys):
    code = main(["report", "--html", str(tmp_path / "x.html")])
    assert code == 2
    assert "at least one" in capsys.readouterr().err
