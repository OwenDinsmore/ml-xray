"""ml-xray: framework-agnostic ML model & dataset analysis.

The public surface is organized by training-lifecycle stage:

- :mod:`ml_xray.lint` -- pre-training dataset QA (leakage, drift, label noise,
  duplicates, imbalance, outliers).
- :mod:`ml_xray.slices` -- post-training error analysis / slice discovery.
- :mod:`ml_xray.embed` -- post-training embedding diff.

All analysis entry points accept arrays / DataFrames and never a model object.
"""

from __future__ import annotations

from .baseline import ReportDiff, diff_reports
from .config import LintConfig
from .embed import EmbedDiffReport, EmbeddingDiff
from .lint import Check, Finding, Linter, LintReport, Severity
from .slices import Slice, SliceFinder, SliceReport

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # lint
    "Linter",
    "LintReport",
    "Finding",
    "Severity",
    "Check",
    "LintConfig",
    # baseline / regression tracking
    "diff_reports",
    "ReportDiff",
    # slices
    "SliceFinder",
    "SliceReport",
    "Slice",
    # embed
    "EmbeddingDiff",
    "EmbedDiffReport",
]
