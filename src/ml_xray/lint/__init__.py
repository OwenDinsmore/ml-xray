"""``ml_xray.lint`` -- pre-training dataset QA.

Public API:

- :class:`Linter` -- orchestrates checks over a DataFrame.
- :class:`LintReport` -- structured findings, with ``to_html`` / ``to_dict`` /
  ``worst`` and a CI-friendly ``__bool__``.
- :class:`Finding`, :class:`Severity`, :class:`Check` -- building blocks.
- The built-in checks (``LeakageCheck``, ``DriftCheck``, ...).

Importing this package registers all built-in checks.
"""

from __future__ import annotations

from .base import (
    Check,
    Finding,
    LintContext,
    Severity,
    default_checks,
    get_registry,
    register_check,
)

# Import for side effect: registers every built-in check.
from .checks import (
    DriftCheck,
    DuplicatesCheck,
    ImbalanceCheck,
    LabelNoiseCheck,
    LeakageCheck,
    OutliersCheck,
)
from .linter import Linter, LintReport

__all__ = [
    "Linter",
    "LintReport",
    "Finding",
    "Severity",
    "Check",
    "LintContext",
    "register_check",
    "get_registry",
    "default_checks",
    "LeakageCheck",
    "DriftCheck",
    "LabelNoiseCheck",
    "DuplicatesCheck",
    "ImbalanceCheck",
    "OutliersCheck",
]
