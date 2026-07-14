"""Per-slice metrics and significance testing.

Provides:

- :func:`resolve_metric` -- turn a metric name (or ``"auto"``) into a scoring
  callable that knows whether higher is better;
- :func:`slice_pvalue` -- test whether a slice's per-row loss differs from the
  rest of the data (two-proportion z-test for classification, Welch's t-test for
  regression);
- :func:`benjamini_hochberg` -- FDR correction across the many slices tested,
  which is what keeps slice discovery from surfacing noise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, ttest_ind
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

__all__ = ["Metric", "resolve_metric", "slice_pvalue", "benjamini_hochberg"]


@dataclass
class Metric:
    """A named scoring function with an optimization direction.

    Attributes
    ----------
    name : str
        Metric name (``"accuracy"``, ``"f1"``, ``"mse"``, ``"mae"``,
        ``"roc_auc"``, ``"log_loss"``, ``"custom"``).
    fn : callable
        ``(y_true, y_pred, y_proba) -> float``.
    greater_is_better : bool
        Whether a larger score means a better model on that slice.
    needs_proba : bool
        Whether the metric requires ``y_proba``.
    """

    name: str
    fn: Callable[..., float]
    greater_is_better: bool
    needs_proba: bool = False

    def __call__(self, y_true, y_pred, y_proba=None) -> float:
        if self.needs_proba and y_proba is None:
            raise ValueError(f"metric {self.name!r} requires y_proba")
        return float(self.fn(y_true, y_pred, y_proba))


def _f1(y_true, y_pred, _y_proba=None) -> float:
    average = "binary" if len(np.unique(y_true)) <= 2 else "macro"
    try:
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except ValueError:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _roc_auc(y_true, _y_pred, y_proba) -> float:
    # Undefined when a slice contains a single class -> NaN, which the finder skips.
    try:
        proba = np.asarray(y_proba, dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba = proba[:, 1]
        if proba.ndim == 1:
            return float(roc_auc_score(y_true, proba))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except ValueError:
        return float("nan")


def _log_loss(y_true, _y_pred, y_proba) -> float:
    try:
        return float(log_loss(y_true, np.asarray(y_proba, dtype=float)))
    except ValueError:
        return float("nan")


_BUILTINS: dict[str, Metric] = {
    "accuracy": Metric("accuracy", lambda yt, yp, pp=None: accuracy_score(yt, yp), True),
    "f1": Metric("f1", _f1, True),
    "mse": Metric("mse", lambda yt, yp, pp=None: mean_squared_error(yt, yp), False),
    "mae": Metric("mae", lambda yt, yp, pp=None: mean_absolute_error(yt, yp), False),
    "roc_auc": Metric("roc_auc", _roc_auc, True, needs_proba=True),
    "log_loss": Metric("log_loss", _log_loss, False, needs_proba=True),
}


def resolve_metric(metric: str | Callable, task: str | None) -> Metric:
    """Resolve a metric name (or ``"auto"``/callable) to a :class:`Metric`.

    Parameters
    ----------
    metric : str or callable
        ``"accuracy"`` / ``"f1"`` / ``"mse"`` / ``"mae"`` / ``"auto"``, or a
        callable ``(y_true, y_pred, y_proba) -> float`` (assumed higher-is-better).
    task : {"classification", "regression"}, optional
        Used to resolve ``"auto"`` (accuracy for classification, MSE otherwise).

    Returns
    -------
    Metric
        The resolved metric.

    Raises
    ------
    ValueError
        If ``metric`` is an unknown name.
    """
    if isinstance(metric, Metric):
        return metric
    if callable(metric):
        return Metric("custom", lambda yt, yp, pp=None: float(metric(yt, yp, pp)), True)
    name = metric.lower()
    if name == "auto":
        name = "accuracy" if (task or "classification") == "classification" else "mse"
    if name not in _BUILTINS:
        valid = ", ".join(sorted(_BUILTINS))
        raise ValueError(f"unknown metric {metric!r}; expected one of: {valid}, auto")
    return _BUILTINS[name]


def _two_proportion_pvalue(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided two-proportion z-test p-value for error rates."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return float(2 * norm.sf(abs(z)))


def slice_pvalue(in_slice: np.ndarray, loss: np.ndarray, task: str) -> float:
    """Significance of a slice's per-row loss vs the rest of the data.

    Parameters
    ----------
    in_slice : numpy.ndarray
        Boolean mask selecting the slice's rows.
    loss : numpy.ndarray
        Per-row loss where larger is worse: a 0/1 error indicator for
        classification, or the absolute error for regression.
    task : {"classification", "regression"}
        Chooses the test (two-proportion z-test vs Welch's t-test).

    Returns
    -------
    float
        Two-sided p-value. ``1.0`` when the test is degenerate (empty rest,
        zero variance, etc.).
    """
    rest = ~in_slice
    n1, n2 = int(in_slice.sum()), int(rest.sum())
    if n1 == 0 or n2 == 0:
        return 1.0
    if task == "classification":
        x1 = int(loss[in_slice].sum())
        x2 = int(loss[rest].sum())
        return _two_proportion_pvalue(x1, n1, x2, n2)
    a, b = loss[in_slice], loss[rest]
    if np.std(a) == 0 and np.std(b) == 0:
        return 1.0
    result = ttest_ind(a, b, equal_var=False)
    p = float(result.pvalue)
    return p if np.isfinite(p) else 1.0


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction over many p-values.

    Parameters
    ----------
    p_values : numpy.ndarray
        Raw per-hypothesis p-values.
    alpha : float
        Target false-discovery rate.

    Returns
    -------
    numpy.ndarray
        Boolean mask of hypotheses that remain significant after correction.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    mask = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        mask[order[: cutoff + 1]] = True
    return mask
