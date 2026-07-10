"""Train-vs-test distribution drift.

Given a ``split`` label, this check compares the reference split (``"train"`` if
present, else the largest split) against every other split value, column by
column:

- numeric columns: Population Stability Index (PSI) plus a two-sample
  Kolmogorov-Smirnov test;
- categorical columns: PSI over category frequencies plus the Jensen-Shannon
  divergence between the two distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from .._util import is_numeric
from ..base import Check, Finding, LintContext, Severity, register_check

__all__ = ["DriftCheck", "psi"]

# Standard PSI thresholds: <0.1 stable, 0.1-0.25 moderate shift, >0.25 large.
_PSI_WARN = 0.1
_PSI_ERROR = 0.25
_KS_ALPHA = 0.05
_JS_WARN = 0.1
_JS_ERROR = 0.2
_EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index between a reference and a comparison sample.

    Parameters
    ----------
    expected : numpy.ndarray
        Reference (e.g. train) numeric values.
    actual : numpy.ndarray
        Comparison (e.g. test) numeric values.
    bins : int
        Number of quantile bins built from ``expected``.

    Returns
    -------
    float
        The PSI. ``0`` means identical distributions; larger is more shifted.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_frac = e_counts / max(e_counts.sum(), 1)
    a_frac = a_counts / max(a_counts.sum(), 1)
    e_frac = np.clip(e_frac, _EPS, None)
    a_frac = np.clip(a_frac, _EPS, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def _categorical_psi(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Return (PSI, Jensen-Shannon divergence) over category frequencies."""
    e_counts = expected.value_counts()
    a_counts = actual.value_counts()
    levels = e_counts.index.union(a_counts.index)
    e_frac = (e_counts.reindex(levels).fillna(0) / max(len(expected), 1)).to_numpy()
    a_frac = (a_counts.reindex(levels).fillna(0) / max(len(actual), 1)).to_numpy()
    e_frac = np.clip(e_frac, _EPS, None)
    a_frac = np.clip(a_frac, _EPS, None)
    psi_val = float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))
    js = float(jensenshannon(e_frac, a_frac, base=2) ** 2)
    if not np.isfinite(js):
        js = 0.0
    return psi_val, js


@register_check
class DriftCheck(Check):
    """Flag feature columns whose distribution shifts across splits."""

    name = "drift"
    description = (
        "Train-vs-test distribution drift via PSI, KS test, and Jensen-Shannon divergence."
    )

    def run(self, ctx: LintContext) -> list[Finding]:
        split = ctx.split
        if split is None or split.nunique(dropna=True) < 2:
            return []

        split = split.reset_index(drop=True)
        df = ctx.df.reset_index(drop=True)
        ref_value = self._reference_split(split)
        ref_mask = (split == ref_value).to_numpy()

        findings: list[Finding] = []
        for other in sorted(v for v in split.dropna().unique() if v != ref_value):
            other_mask = (split == other).to_numpy()
            for col in ctx.feature_columns:
                series = df[col]
                ref = series[ref_mask]
                cur = series[other_mask]
                if is_numeric(series):
                    finding = self._numeric_drift(col, ref, cur, ref_value, other)
                else:
                    finding = self._categorical_drift(col, ref, cur, ref_value, other)
                if finding is not None:
                    findings.append(finding)
        return findings

    @staticmethod
    def _reference_split(split: pd.Series) -> object:
        values = set(split.dropna().unique())
        for candidate in ("train", "TRAIN", "Train"):
            if candidate in values:
                return candidate
        return split.value_counts().idxmax()

    def _numeric_drift(
        self, col: str, ref: pd.Series, cur: pd.Series, ref_value: object, other: object
    ) -> Finding | None:
        ref_v = ref.dropna().astype(float).to_numpy()
        cur_v = cur.dropna().astype(float).to_numpy()
        if ref_v.size < 5 or cur_v.size < 5:
            return None
        psi_val = psi(ref_v, cur_v)
        ks = ks_2samp(ref_v, cur_v)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        sev = self._severity_from_psi(psi_val, ks_p)
        if sev is None:
            return None
        return Finding(
            check=self.name,
            severity=sev,
            message=(
                f"Numeric column '{col}' drifts between split '{ref_value}' and "
                f"'{other}' (PSI={psi_val:.3f}, KS p={ks_p:.3g})."
            ),
            column=col,
            detail={
                "psi": psi_val,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "reference_split": ref_value,
                "comparison_split": other,
                "kind": "numeric",
            },
        )

    def _categorical_drift(
        self, col: str, ref: pd.Series, cur: pd.Series, ref_value: object, other: object
    ) -> Finding | None:
        ref_v = ref.dropna().astype("object")
        cur_v = cur.dropna().astype("object")
        if len(ref_v) < 5 or len(cur_v) < 5:
            return None
        psi_val, js = _categorical_psi(ref_v, cur_v)
        sev = self._severity_from_js(psi_val, js)
        if sev is None:
            return None
        return Finding(
            check=self.name,
            severity=sev,
            message=(
                f"Categorical column '{col}' drifts between split '{ref_value}' and "
                f"'{other}' (JSD={js:.3f}, PSI={psi_val:.3f})."
            ),
            column=col,
            detail={
                "psi": psi_val,
                "jensen_shannon": js,
                "reference_split": ref_value,
                "comparison_split": other,
                "kind": "categorical",
            },
        )

    @staticmethod
    def _severity_from_psi(psi_val: float, ks_p: float) -> Severity | None:
        if psi_val >= _PSI_ERROR:
            return Severity.ERROR
        if psi_val >= _PSI_WARN or ks_p < _KS_ALPHA:
            return Severity.WARN
        return None

    @staticmethod
    def _severity_from_js(psi_val: float, js: float) -> Severity | None:
        if psi_val >= _PSI_ERROR or js >= _JS_ERROR:
            return Severity.ERROR
        if psi_val >= _PSI_WARN or js >= _JS_WARN:
            return Severity.WARN
        return None
