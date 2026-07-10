"""``ml_xray.slices`` -- post-training error analysis / slice discovery.

Public API: :class:`SliceFinder`, :class:`SliceReport`, :class:`Slice`. Helper
modules :mod:`~ml_xray.slices.binning` and :mod:`~ml_xray.slices.metrics` expose
the discretization strategies and the metric / significance machinery.
"""

from __future__ import annotations

from .binning import bin_feature, discretize
from .finder import Slice, SliceFinder, SliceReport
from .metrics import Metric, benjamini_hochberg, resolve_metric, slice_pvalue

__all__ = [
    "SliceFinder",
    "SliceReport",
    "Slice",
    "bin_feature",
    "discretize",
    "Metric",
    "resolve_metric",
    "slice_pvalue",
    "benjamini_hochberg",
]
