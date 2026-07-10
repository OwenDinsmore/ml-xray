"""Class imbalance and rare-category checks.

Covers three related issues:

- **class imbalance** of a classification target (majority-to-minority ratio);
- **rare categorical levels** that appear in too few rows to learn from;
- **single-value columns**, which carry no signal.
"""

from __future__ import annotations

from .._util import categorical_columns, infer_task
from ..base import Check, Finding, LintContext, Severity, register_check

__all__ = ["ImbalanceCheck"]

_RATIO_WARN = 10.0
_RATIO_ERROR = 100.0
_RARE_FRACTION = 0.01
_RARE_MIN_COUNT = 10


@register_check
class ImbalanceCheck(Check):
    """Flag class imbalance, rare categorical levels, and single-value columns."""

    name = "imbalance"
    description = "Class imbalance ratio, rare categorical levels, and single-value columns."

    def run(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._class_imbalance(ctx))
        findings.extend(self._rare_levels(ctx))
        findings.extend(self._single_value_columns(ctx))
        return findings

    def _class_imbalance(self, ctx: LintContext) -> list[Finding]:
        y = ctx.target_series
        if y is None:
            return []
        if (ctx.task or infer_task(y)) != "classification":
            return []
        counts = y.dropna().value_counts()
        if len(counts) < 2:
            return []
        majority, minority = int(counts.iloc[0]), int(counts.iloc[-1])
        ratio = majority / max(minority, 1)
        if ratio >= _RATIO_ERROR:
            sev = Severity.ERROR
        elif ratio >= _RATIO_WARN:
            sev = Severity.WARN
        else:
            return []
        return [
            Finding(
                check=self.name,
                severity=sev,
                message=(
                    f"Target '{y.name}' is imbalanced: majority/minority ratio "
                    f"{ratio:.1f} (min class '{counts.index[-1]}' has {minority} rows)."
                ),
                column=y.name,
                detail={
                    "ratio": ratio,
                    "majority_count": majority,
                    "minority_count": minority,
                    "class_counts": {str(k): int(v) for k, v in counts.items()},
                    "kind": "class_imbalance",
                },
            )
        ]

    def _rare_levels(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        n = len(ctx.df)
        threshold = max(_RARE_MIN_COUNT, int(_RARE_FRACTION * n))
        for col in categorical_columns(ctx.df, exclude=ctx.target):
            counts = ctx.df[col].value_counts(dropna=True)
            rare = counts[counts < threshold]
            if len(rare) == 0 or len(counts) < 2:
                continue
            findings.append(
                Finding(
                    check=self.name,
                    severity=Severity.INFO,
                    message=(
                        f"Column '{col}' has {len(rare)} rare level(s) "
                        f"(< {threshold} rows each)."
                    ),
                    column=col,
                    detail={
                        "n_rare_levels": int(len(rare)),
                        "threshold": threshold,
                        "rare_levels": {str(k): int(v) for k, v in rare.head(25).items()},
                        "kind": "rare_levels",
                    },
                )
            )
        return findings

    def _single_value_columns(self, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for col in ctx.feature_columns:
            if ctx.df[col].nunique(dropna=True) <= 1:
                findings.append(
                    Finding(
                        check=self.name,
                        severity=Severity.WARN,
                        message=f"Column '{col}' has a single value - no signal.",
                        column=col,
                        detail={
                            "n_unique": int(ctx.df[col].nunique(dropna=True)),
                            "kind": "single_value",
                        },
                    )
                )
        return findings
