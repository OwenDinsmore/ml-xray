"""Temporal-leakage checks.

When a dataset has a time/ordering column, two leakage patterns matter:

- **train/test time overlap** -- in a proper temporal split all training rows
  precede all evaluation rows; training rows that occur at or after the start of
  a later split leak future information into training;
- **time-proxy features** -- a feature that is (nearly) monotonic with time acts
  as a surrogate timestamp/row-id and can leak ordering into the model.

The time column is supplied via ``Linter.run(..., time="ts")`` or an explicit
``TemporalLeakageCheck(time_col=...)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..base import Check, Finding, LintContext, Severity, register_check

__all__ = ["TemporalLeakageCheck"]

_PROXY_ABS_RHO = 0.999
_PROXY_WARN_RHO = 0.98


@register_check
class TemporalLeakageCheck(Check):
    """Detect train/test time overlap and time-proxy features."""

    name = "temporal_leakage"
    description = "Train/test time overlap and features that are proxies for the time column."

    def __init__(self, time_col: str | None = None) -> None:
        """
        Parameters
        ----------
        time_col : str, optional
            Time column name. Overrides ``ctx.time`` when given; otherwise the
            column passed to ``Linter.run(time=...)`` is used.
        """
        self.time_col = time_col

    def run(self, ctx: LintContext) -> list[Finding]:
        time_col = self.time_col or ctx.time
        if time_col is None or time_col not in ctx.df.columns:
            return []
        time = self._as_numeric_time(ctx.df[time_col])
        if time.notna().sum() < 3:
            return []

        findings: list[Finding] = []
        findings.extend(self._split_overlap(ctx, time, time_col))
        findings.extend(self._time_proxy_features(ctx, time, time_col))
        return findings

    @staticmethod
    def _as_numeric_time(s: pd.Series) -> pd.Series:
        """Coerce a time column to a numeric ordering (datetimes -> int64 ns)."""
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s.astype(float)
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().any():
            return parsed.astype("int64").astype(float).where(parsed.notna())
        return pd.to_numeric(s, errors="coerce")

    def _split_overlap(self, ctx: LintContext, time: pd.Series, time_col: str) -> list[Finding]:
        split = ctx.split
        if split is None or split.nunique(dropna=True) < 2:
            return []
        split = split.reset_index(drop=True)
        time = time.reset_index(drop=True)

        ref = self._reference_split(split)
        train_mask = (split == ref).to_numpy()
        train_time = time[train_mask]
        if train_time.notna().sum() == 0:
            return []

        findings: list[Finding] = []
        for other in sorted(v for v in split.dropna().unique() if v != ref):
            other_time = time[(split == other).to_numpy()]
            other_start = other_time.min()
            if pd.isna(other_start):
                continue
            # Training rows at or after the later split's start leak the future.
            leaking = np.flatnonzero(train_mask & (time >= other_start).to_numpy())
            if leaking.size == 0:
                continue
            frac = leaking.size / max(train_mask.sum(), 1)
            sev = Severity.ERROR if frac >= 0.02 else Severity.WARN
            findings.append(
                Finding(
                    check=self.name,
                    severity=sev,
                    message=(
                        f"{leaking.size} '{ref}' rows occur at/after the start of split "
                        f"'{other}' in '{time_col}' - future leaks into training."
                    ),
                    column=time_col,
                    detail={
                        "n_leaking_rows": int(leaking.size),
                        "fraction_of_train": float(frac),
                        "reference_split": ref,
                        "comparison_split": other,
                        "kind": "train_test_time_overlap",
                    },
                    rows=[int(i) for i in leaking],
                )
            )
        return findings

    def _time_proxy_features(
        self, ctx: LintContext, time: pd.Series, time_col: str
    ) -> list[Finding]:
        findings: list[Finding] = []
        time = time.reset_index(drop=True)
        for col in ctx.feature_columns:
            series = ctx.df[col].reset_index(drop=True)
            if not (
                pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            ):
                continue
            mask = series.notna() & time.notna()
            if mask.sum() < 5 or series[mask].nunique() < 3:
                continue
            rho = spearmanr(series[mask], time[mask]).correlation
            if rho is None or not np.isfinite(rho):
                continue
            arho = abs(float(rho))
            if arho >= _PROXY_ABS_RHO:
                sev = Severity.ERROR
            elif arho >= _PROXY_WARN_RHO:
                sev = Severity.WARN
            else:
                continue
            findings.append(
                Finding(
                    check=self.name,
                    severity=sev,
                    message=(
                        f"Feature '{col}' is {arho:.4f} rank-correlated with time "
                        f"'{time_col}' - a time proxy that can leak ordering."
                    ),
                    column=col,
                    detail={
                        "spearman_rho": float(rho),
                        "abs_rho": arho,
                        "kind": "time_proxy_feature",
                    },
                )
            )
        return findings

    @staticmethod
    def _reference_split(split: pd.Series) -> object:
        values = set(split.dropna().unique())
        for candidate in ("train", "TRAIN", "Train"):
            if candidate in values:
                return candidate
        return split.value_counts().idxmax()
