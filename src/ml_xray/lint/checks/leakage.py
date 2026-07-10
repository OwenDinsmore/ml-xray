"""Target-leakage and train/test-overlap checks.

Three failure modes are covered:

- a feature whose values are almost perfectly correlated with a numeric target;
- a feature that *deterministically* predicts a categorical target (each feature
  value maps to a single label) -- the classic "leaked id / post-outcome"
  column;
- rows duplicated *across* a train/test split, which silently leaks test labels
  into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._util import infer_task, is_numeric
from ..base import Check, Finding, LintContext, Severity, register_check

__all__ = ["LeakageCheck"]

# Above this |corr| a numeric feature is treated as a near-copy of the target.
_CORR_ERROR = 0.999
_CORR_WARN = 0.95
# A deterministic categorical predictor must cover at least this many rows to be
# worth flagging, and must not simply be a per-row identifier.
_MIN_GROUP_ROWS = 20


@register_check
class LeakageCheck(Check):
    """Detect target leakage and train/test row overlap."""

    name = "leakage"
    description = "Target leakage: near-perfect feature/target relationships and cross-split rows."

    def run(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._cross_split_overlap(ctx))

        y = ctx.target_series
        if y is None:
            return findings

        task = ctx.task or infer_task(y)
        for col in ctx.feature_columns:
            feat = ctx.df[col]
            if is_numeric(feat) and (task == "regression" or is_numeric(y)):
                findings.extend(self._numeric_corr(col, feat, y))
            else:
                findings.extend(self._deterministic_predictor(col, feat, y))
        return findings

    def _numeric_corr(self, col: str, feat: pd.Series, y: pd.Series) -> list[Finding]:
        mask = feat.notna() & y.notna()
        if mask.sum() < 3:
            return []
        x = feat[mask].astype(float)
        t = y[mask].astype(float)
        if x.nunique() < 2 or t.nunique() < 2:
            return []
        corr = float(np.corrcoef(x, t)[0, 1])
        if not np.isfinite(corr):
            return []
        acorr = abs(corr)
        if acorr >= _CORR_ERROR:
            sev = Severity.ERROR
        elif acorr >= _CORR_WARN:
            sev = Severity.WARN
        else:
            return []
        return [
            Finding(
                check=self.name,
                severity=sev,
                message=(
                    f"Feature '{col}' is {acorr:.4f}-correlated with target "
                    f"'{y.name}' - likely leakage."
                ),
                column=col,
                detail={"correlation": corr, "abs_correlation": acorr, "kind": "numeric_corr"},
            )
        ]

    def _deterministic_predictor(self, col: str, feat: pd.Series, y: pd.Series) -> list[Finding]:
        mask = feat.notna() & y.notna()
        if mask.sum() < _MIN_GROUP_ROWS:
            return []
        x = feat[mask].astype("object")
        t = y[mask].astype("object")
        n_unique_x = x.nunique()
        # A near-per-row identifier trivially "predicts" anything; not leakage.
        if n_unique_x >= 0.9 * len(x):
            return []
        # Purity: does each feature value map to a single target label?
        grouped = t.groupby(x, observed=True)
        purity = grouped.apply(lambda g: g.value_counts(normalize=True).iloc[0])
        weights = grouped.size()
        weighted_purity = float(np.average(purity, weights=weights))
        if weighted_purity >= 0.999:
            sev = Severity.ERROR
        elif weighted_purity >= 0.98:
            sev = Severity.WARN
        else:
            return []
        return [
            Finding(
                check=self.name,
                severity=sev,
                message=(
                    f"Feature '{col}' deterministically predicts target '{y.name}' "
                    f"(purity {weighted_purity:.4f}) - likely leakage."
                ),
                column=col,
                detail={
                    "weighted_purity": weighted_purity,
                    "n_unique_values": int(n_unique_x),
                    "kind": "deterministic_predictor",
                },
            )
        ]

    def _cross_split_overlap(self, ctx: LintContext) -> list[Finding]:
        split = ctx.split
        if split is None or split.nunique(dropna=True) < 2:
            return []
        # Compare feature rows only (ignoring the split column itself).
        feat_cols = [c for c in ctx.feature_columns]
        if not feat_cols:
            feat_cols = list(ctx.df.columns)
        sub = ctx.df[feat_cols].reset_index(drop=True)
        split_reset = split.reset_index(drop=True)

        key = pd.util.hash_pandas_object(sub, index=False)
        # Rows sharing an identical feature signature but living in >1 split value.
        groups = pd.DataFrame({"key": key.to_numpy(), "split": split_reset.to_numpy()})
        n_splits_per_key = groups.groupby("key")["split"].nunique()
        leaked_keys = set(n_splits_per_key[n_splits_per_key > 1].index)
        if not leaked_keys:
            return []
        offending = groups.index[groups["key"].isin(leaked_keys)].tolist()
        return [
            Finding(
                check=self.name,
                severity=Severity.ERROR,
                message=(
                    f"{len(offending)} rows appear in more than one split value - "
                    f"train/test overlap leaks labels."
                ),
                column=None,
                detail={
                    "n_overlapping_rows": len(offending),
                    "n_shared_signatures": len(leaked_keys),
                    "kind": "cross_split_overlap",
                },
                rows=[int(i) for i in offending],
            )
        ]
