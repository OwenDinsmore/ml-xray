"""ml-xray: framework-agnostic ML model & dataset analysis.

The public surface is organized by training-lifecycle stage:

- :mod:`ml_xray.lint` -- pre-training dataset QA (leakage, drift, label noise,
  duplicates, imbalance, outliers). *Implemented.*
- :mod:`ml_xray.slices` -- post-training error analysis / slice discovery.
  *Scaffolded (Phase 2).*
- :mod:`ml_xray.embed` -- post-training embedding diff. *Scaffolded (Phase 3).*

All analysis entry points accept arrays / DataFrames and never a model object.
"""

from __future__ import annotations

from .lint import Check, Finding, Linter, LintReport, Severity

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Linter",
    "LintReport",
    "Finding",
    "Severity",
    "Check",
]
