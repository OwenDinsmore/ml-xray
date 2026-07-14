"""Tests for interactive plotly reports: figures build, HTML embeds a
self-contained plotly div, and everything degrades gracefully without plotly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthetic
from ml_xray import EmbeddingDiff, SliceFinder


def _slice_report():
    rng = np.random.default_rng(0)
    n = 2000
    region = rng.choice(["EU", "US", "APAC"], size=n)
    y = rng.integers(0, 2, n)
    y_pred = y.copy()
    eu = region == "EU"
    y_pred[eu & (rng.uniform(size=n) < 0.5)] ^= 1
    return (
        SliceFinder(metric="accuracy", min_support=50)
        .fit(pd.DataFrame({"region": region}), y, y_pred)
        .report()
    )


def test_slice_plotly_figure_builds():
    pytest.importorskip("plotly")
    fig = _slice_report().plot(interactive=True)
    assert type(fig).__module__.startswith("plotly")


def test_embed_plotly_figure_builds():
    pytest.importorskip("plotly")
    planted = synthetic.partially_perturbed_spaces(n=200, moved=20)
    report = EmbeddingDiff(k=10).fit(planted.emb_a, planted.emb_b).report()
    fig = report.plot_projection(interactive=True)
    assert type(fig).__module__.startswith("plotly")


def test_slice_interactive_html_is_self_contained(tmp_path):
    pytest.importorskip("plotly")
    from ml_xray.report import slice_report_html

    html = slice_report_html(_slice_report(), interactive=True)
    assert "plotly" in html.lower()
    # Plotly.js is inlined (no external <script src>/<img src>/<link href>).
    import re

    assert not re.search(r'(?:src|href)="https?://[^"]+\.(?:js|css|png|svg)"', html)
    out = tmp_path / "slice.html"
    out.write_text(html, encoding="utf-8")
    assert out.stat().st_size > 100_000  # plotly.js inlined


def test_embed_interactive_html_embeds_plotly(tmp_path):
    pytest.importorskip("plotly")
    from ml_xray.report import embed_report_html

    planted = synthetic.partially_perturbed_spaces(n=150, moved=15)
    report = EmbeddingDiff(k=8).fit(planted.emb_a, planted.emb_b).report()
    html = embed_report_html(report, interactive=True)
    assert "plotly" in html.lower()


def test_interactive_degrades_without_plotly(monkeypatch):
    # Simulate plotly being unavailable: the report still renders (no chart).
    import ml_xray.report as report_mod

    monkeypatch.setattr(report_mod, "_plotly", lambda: None)
    html = report_mod.slice_report_html(_slice_report(), interactive=True)
    assert "Underperforming slices" in html  # table still present
    assert "plotly" not in html.lower()
