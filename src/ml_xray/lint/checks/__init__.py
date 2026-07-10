"""Built-in dataset checks.

Importing this package registers every check with the shared registry (see
:func:`ml_xray.lint.base.register_check`), so
:func:`ml_xray.lint.base.default_checks` can enumerate them.
"""

from __future__ import annotations

from .drift import DriftCheck
from .duplicates import DuplicatesCheck
from .imbalance import ImbalanceCheck
from .label_noise import LabelNoiseCheck
from .leakage import LeakageCheck
from .outliers import OutliersCheck

__all__ = [
    "LeakageCheck",
    "DriftCheck",
    "LabelNoiseCheck",
    "DuplicatesCheck",
    "ImbalanceCheck",
    "OutliersCheck",
]
