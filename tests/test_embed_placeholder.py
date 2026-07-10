"""Placeholder test for the Phase 3 embedding-diff scaffold.

When Phase 3 is implemented, replace this with the real test: identical spaces
(overlap ~= 1), fully shuffled spaces (overlap ~= 0), and a controlled partial
perturbation (overlap in a known band), asserting metrics land in expected
ranges and ``movers`` surfaces the perturbed points.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_xray.embed import EmbedDiffReport, EmbeddingDiff


def test_report_default_fields():
    report = EmbedDiffReport()
    assert np.isnan(report.neighbor_overlap)
    assert report.movers == []


def test_embedding_diff_not_implemented():
    a = np.random.default_rng(0).normal(size=(10, 4))
    b = np.random.default_rng(1).normal(size=(10, 4))
    diff = EmbeddingDiff(k=3)
    with pytest.raises(NotImplementedError):
        diff.fit(a, b)


def test_report_render_not_implemented():
    with pytest.raises(NotImplementedError):
        EmbedDiffReport().to_html("out.html")
