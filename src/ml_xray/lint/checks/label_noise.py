"""Mislabel-candidate detection.

The core implementation trains an out-of-fold classifier and flags rows that are
*confidently wrong*: the model's predicted class disagrees with the given label
and the predicted probability of the given label is low. When the ``[noise]``
extra (``cleanlab``) is installed, the out-of-fold probabilities are handed to
its stronger confident-learning routine (``find_label_issues``).

Only classification targets are supported; regression targets are skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..._optional import optional_import
from .._util import design_matrix, infer_task
from ..base import Check, Finding, LintContext, Severity, register_check

__all__ = ["LabelNoiseCheck"]

# Flag a row when P(given label) falls below this and another class wins.
_MARGIN_THRESHOLD = 0.2
_MIN_ROWS = 40


@register_check
class LabelNoiseCheck(Check):
    """Detect likely-mislabeled rows via out-of-fold predictions."""

    name = "label_noise"
    description = "Mislabel candidates via out-of-fold confident-learning / low-margin flagging."

    def run(self, ctx: LintContext) -> list[Finding]:
        y = ctx.target_series
        if y is None:
            return []
        task = ctx.task or infer_task(y)
        if task != "classification":
            return []

        mask = y.notna()
        if int(mask.sum()) < _MIN_ROWS:
            return []

        features = ctx.df.loc[mask, ctx.feature_columns]
        labels = y[mask].astype("object")
        if labels.nunique() < 2 or labels.value_counts().min() < 3:
            return []

        X, names = design_matrix(features)
        if X.shape[1] == 0:
            return []

        classes = np.array(sorted(labels.unique(), key=str))
        y_codes = pd.Categorical(labels, categories=classes).codes
        positions = np.flatnonzero(mask.to_numpy())

        proba = self._oof_proba(X, y_codes, len(classes), ctx.seed)
        if proba is None:
            return []

        issue_local, backend = self._find_issues(proba, y_codes)
        offending = positions[issue_local].tolist()
        if not offending:
            return []

        given_proba = proba[np.arange(len(y_codes)), y_codes]
        frac = len(offending) / len(y_codes)
        # The margin heuristic has an inherent low false-positive rate near any
        # decision boundary, so only a substantial fraction escalates to ERROR.
        sev = Severity.ERROR if frac >= 0.05 else Severity.WARN
        worst = positions[issue_local][np.argsort(given_proba[issue_local])[:10]]
        return [
            Finding(
                check=self.name,
                severity=sev,
                message=(
                    f"{len(offending)} rows ({frac:.1%}) look mislabeled: the model "
                    f"disagrees with the given label for column '{y.name}'."
                ),
                column=y.name,
                detail={
                    "n_candidates": len(offending),
                    "fraction": frac,
                    "backend": backend,
                    "margin_threshold": _MARGIN_THRESHOLD,
                    "example_rows": [int(i) for i in worst],
                    "n_features": len(names),
                },
                rows=[int(i) for i in offending],
            )
        ]

    def _find_issues(self, proba: np.ndarray, y_codes: np.ndarray) -> tuple[np.ndarray, str]:
        """Return (boolean issue mask, backend name)."""
        cleanlab = optional_import("cleanlab")
        if cleanlab is not None:  # pragma: no cover - exercised only with [noise] extra
            try:
                from cleanlab.filter import find_label_issues

                mask = find_label_issues(
                    labels=y_codes,
                    pred_probs=proba,
                    return_indices_ranked_by="self_confidence",
                )
                issue = np.zeros(len(y_codes), dtype=bool)
                issue[np.asarray(mask, dtype=int)] = True
                return issue, "cleanlab"
            except Exception:
                pass
        given = proba[np.arange(len(y_codes)), y_codes]
        pred = proba.argmax(axis=1)
        return (pred != y_codes) & (given < _MARGIN_THRESHOLD), "out_of_fold"

    @staticmethod
    def _oof_proba(
        X: np.ndarray, y_codes: np.ndarray, n_classes: int, seed: int
    ) -> np.ndarray | None:
        n_splits = int(min(5, np.bincount(y_codes).min()))
        if n_splits < 2:
            return None
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = HistGradientBoostingClassifier(random_state=seed, max_iter=100)
        try:
            proba = cross_val_predict(model, X, y_codes, cv=cv, method="predict_proba")
        except Exception:  # pragma: no cover - defensive against degenerate folds
            return None
        proba = np.asarray(proba, dtype=float)
        if proba.shape[1] != n_classes:
            full = np.zeros((proba.shape[0], n_classes), dtype=float)
            full[:, : proba.shape[1]] = proba
            proba = full
        return proba
