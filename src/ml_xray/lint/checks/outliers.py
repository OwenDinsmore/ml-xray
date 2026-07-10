"""Numeric outlier, missingness, and constant-column checks.

- **numeric outliers** via a robust z-score (median / MAD) with an IQR fallback;
- **high-null-fraction columns** whose missing rate exceeds a threshold;
- **constant columns** with a single non-null value;
- **out-of-declared-range** values, when per-column ``(low, high)`` bounds are
  supplied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._util import numeric_columns
from ..base import Check, Finding, LintContext, Severity, register_check

__all__ = ["OutliersCheck"]

_ROBUST_Z = 3.5
_NULL_WARN = 0.5
_NULL_ERROR = 0.9
_OUTLIER_WARN_FRACTION = 0.01
# Iglewicz-Hoaglin modified z-score constant: 0.6745 * (x - median) / MAD is
# ~N(0, 1) for normal data, so |z| > 3.5 is the standard outlier rule.
_MAD_SCALE = 0.6745


@register_check
class OutliersCheck(Check):
    """Flag numeric outliers, high-null columns, and constant columns."""

    name = "outliers"
    description = "Numeric outliers (robust z / IQR), high-null columns, and constant columns."

    def __init__(self, ranges: dict[str, tuple[float, float]] | None = None) -> None:
        """
        Parameters
        ----------
        ranges : dict, optional
            Optional ``column -> (low, high)`` inclusive declared bounds. Values
            outside a declared range are reported as ``ERROR``.
        """
        self.ranges = ranges or {}

    def run(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._null_and_constant(ctx))
        findings.extend(self._numeric_outliers(ctx))
        findings.extend(self._out_of_range(ctx))
        return findings

    def _null_and_constant(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        n = len(ctx.df)
        if n == 0:
            return findings
        for col in ctx.feature_columns:
            s = ctx.df[col]
            null_frac = float(s.isna().mean())
            if null_frac >= _NULL_ERROR:
                sev = Severity.ERROR
            elif null_frac >= _NULL_WARN:
                sev = Severity.WARN
            else:
                sev = None
            if sev is not None:
                findings.append(
                    Finding(
                        check=self.name,
                        severity=sev,
                        message=f"Column '{col}' is {null_frac:.0%} null.",
                        column=col,
                        detail={"null_fraction": null_frac, "kind": "high_null"},
                    )
                )
            if s.nunique(dropna=True) <= 1 and null_frac < 1.0:
                findings.append(
                    Finding(
                        check=self.name,
                        severity=Severity.WARN,
                        message=f"Column '{col}' is constant.",
                        column=col,
                        detail={"kind": "constant"},
                    )
                )
        return findings

    def _numeric_outliers(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for col in numeric_columns(ctx.df, exclude=ctx.target):
            s = ctx.df[col]
            values = s.to_numpy(dtype=float)
            finite = np.isfinite(values)
            v = values[finite]
            if v.size < 10 or np.unique(v).size < 3:
                continue
            outlier_finite_mask = self._robust_outlier_mask(v)
            if not outlier_finite_mask.any():
                continue
            positions = np.flatnonzero(finite)[outlier_finite_mask]
            frac = positions.size / len(s)
            sev = Severity.WARN if frac >= _OUTLIER_WARN_FRACTION else Severity.INFO
            findings.append(
                Finding(
                    check=self.name,
                    severity=sev,
                    message=(
                        f"Column '{col}' has {positions.size} numeric outlier(s) "
                        f"({frac:.1%}) by robust z-score."
                    ),
                    column=col,
                    detail={
                        "n_outliers": int(positions.size),
                        "fraction": frac,
                        "threshold_z": _ROBUST_Z,
                        "kind": "numeric_outlier",
                    },
                    rows=[int(i) for i in positions],
                )
            )
        return findings

    @staticmethod
    def _robust_outlier_mask(v: np.ndarray) -> np.ndarray:
        median = np.median(v)
        mad = np.median(np.abs(v - median))
        if mad > 0:
            z = _MAD_SCALE * (v - median) / mad
            return np.abs(z) > _ROBUST_Z
        # Degenerate MAD (heavy ties): fall back to the IQR rule.
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return np.zeros_like(v, dtype=bool)
        return (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)

    def _out_of_range(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for col, (low, high) in self.ranges.items():
            if col not in ctx.df.columns:
                continue
            s = ctx.df[col]
            values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            oob = np.isfinite(values) & ((values < low) | (values > high))
            positions = np.flatnonzero(oob)
            if positions.size == 0:
                continue
            findings.append(
                Finding(
                    check=self.name,
                    severity=Severity.ERROR,
                    message=(
                        f"Column '{col}' has {positions.size} value(s) outside the "
                        f"declared range [{low}, {high}]."
                    ),
                    column=col,
                    detail={
                        "n_out_of_range": int(positions.size),
                        "low": low,
                        "high": high,
                        "kind": "out_of_range",
                    },
                    rows=[int(i) for i in positions],
                )
            )
        return findings
