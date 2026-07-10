"""Per-slice metrics and significance testing (Phase 2 -- scaffolded).

Provides the task-metric evaluation for a slice and the multiple-comparison
correction (Benjamini-Hochberg) applied across the many slices tested, which is
what keeps slice discovery from surfacing noise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["resolve_metric", "benjamini_hochberg"]


def resolve_metric(metric: str | Callable, task: str | None) -> Callable[..., float]:
    """Resolve a metric name (or ``"auto"``) to a scoring callable.

    Parameters
    ----------
    metric : str or callable
        ``"f1"`` / ``"accuracy"`` / ``"mse"`` / ``"mae"`` / ``"auto"``, or a
        callable ``(y_true, y_pred, y_proba) -> float``.
    task : {"classification", "regression"}, optional
        Used to resolve ``"auto"``.

    Returns
    -------
    callable
        A scoring function.

    Raises
    ------
    NotImplementedError
        Always -- Phase 2.
    """
    raise NotImplementedError("TODO(phase 2): resolve metric name/callable from task.")


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction over many slice p-values.

    Parameters
    ----------
    p_values : numpy.ndarray
        Raw per-slice p-values.
    alpha : float
        Target false-discovery rate.

    Returns
    -------
    numpy.ndarray
        Boolean mask of slices that remain significant after correction.

    Raises
    ------
    NotImplementedError
        Always -- Phase 2.
    """
    raise NotImplementedError("TODO(phase 2): Benjamini-Hochberg FDR correction.")
