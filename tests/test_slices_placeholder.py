"""Placeholder test for the Phase 2 slice-discovery scaffold.

When Phase 2 is implemented, replace this with the real test: synthetic data
with a planted weak region, asserting that slice ranks in the top results with
the correct predicate and a negative delta, and that a clean model surfaces no
high-rank false positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_xray.slices import Slice, SliceFinder, SliceReport


def test_slice_dataclass_shape():
    s = Slice(
        predicate={"region": "EU"},
        support=1204,
        metric_value=0.42,
        baseline=0.79,
        delta=-0.37,
        p_value=0.001,
    )
    assert s.support == 1204
    assert s.delta < 0


def test_slice_finder_not_implemented():
    X = pd.DataFrame({"region": ["EU", "US"]})
    finder = SliceFinder(max_depth=2, top_k=5)
    with pytest.raises(NotImplementedError):
        finder.fit(X, y_true=np.array([0, 1]), y_pred=np.array([0, 0]))


def test_slice_report_render_not_implemented():
    with pytest.raises(NotImplementedError):
        SliceReport().to_html("out.html")
