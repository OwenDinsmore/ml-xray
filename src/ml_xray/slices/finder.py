"""Slice discovery / error analysis (Phase 2 -- scaffolded).

Find where a trained model underperforms, e.g. "F1 is 0.42 on
``region=EU & tenure<3mo`` (n=1,204) vs 0.79 overall." Inspired by Slice Finder
and SliceLens error-deviation metrics; not a reimplementation of any single
paper.

Everything here is typed and documented but raises ``NotImplementedError`` until
Phase 2 lands.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["Slice", "SliceReport", "SliceFinder"]


@dataclass
class Slice:
    """A single data slice defined by a feature-value predicate.

    Parameters
    ----------
    predicate : dict
        The slice definition, e.g. ``{"region": "EU", "tenure_bin": "<3mo"}``.
    support : int
        Number of rows matching the predicate.
    metric_value : float
        The task metric evaluated on this slice.
    baseline : float
        The same metric evaluated on the full dataset.
    delta : float
        Signed underperformance vs baseline (``metric_value - baseline``).
    p_value : float
        Multiple-comparison-corrected significance of the deviation.
    """

    predicate: dict[str, Any]
    support: int
    metric_value: float
    baseline: float
    delta: float
    p_value: float


@dataclass
class SliceReport:
    """Ranked slices where a model underperforms (Phase 2 -- scaffolded)."""

    slices: list[Slice] = field(default_factory=list)

    def to_html(self, path: str) -> None:
        """Render the slice report to a self-contained HTML file.

        Raises
        ------
        NotImplementedError
            Always -- Phase 2.
        """
        raise NotImplementedError("TODO(phase 2): render SliceReport to HTML.")

    def plot(self):
        """Bar/heatmap of the worst slices via the selected viz backend.

        Raises
        ------
        NotImplementedError
            Always -- Phase 2.
        """
        raise NotImplementedError("TODO(phase 2): plot worst slices via _viz backend.")


class SliceFinder:
    """Discover underperforming slices via lattice search with pruning.

    Parameters
    ----------
    metric : str or callable
        ``"f1"`` / ``"accuracy"`` / ``"mse"`` / ``"mae"``, a callable
        ``(y_true, y_pred, y_proba) -> float``, or ``"auto"`` to pick from the
        task.
    max_depth : int
        Maximum number of features combined per slice predicate.
    min_support : int
        Minimum rows for a slice to be eligible.
    top_k : int
        Number of slices to return.
    significance : float
        Family-wise / FDR-corrected significance level.
    """

    def __init__(
        self,
        *,
        metric: str | Callable[..., float] = "auto",
        max_depth: int = 2,
        min_support: int = 30,
        top_k: int = 20,
        significance: float = 0.05,
    ) -> None:
        self.metric = metric
        self.max_depth = max_depth
        self.min_support = min_support
        self.top_k = top_k
        self.significance = significance

    def fit(
        self,
        X: pd.DataFrame,
        y_true: Sequence | np.ndarray,
        y_pred: Sequence | np.ndarray,
        y_proba: Sequence | np.ndarray | None = None,
        *,
        task: str | None = None,
    ) -> SliceFinder:
        """Search for underperforming slices over ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Features to slice on (discretized internally by ``binning``).
        y_true, y_pred : array-like
            Ground-truth labels and model predictions, row-aligned to ``X``.
        y_proba : array-like, optional
            Predicted probabilities/scores, when the metric needs them.
        task : {"classification", "regression"}, optional
            Learning task; inferred when omitted.

        Returns
        -------
        SliceFinder
            ``self``, fitted.

        Raises
        ------
        NotImplementedError
            Always -- Phase 2.
        """
        raise NotImplementedError(
            "TODO(phase 2): lattice search + Benjamini-Hochberg-corrected significance."
        )

    def report(self) -> SliceReport:
        """Return the ranked :class:`SliceReport` after :meth:`fit`.

        Raises
        ------
        NotImplementedError
            Always -- Phase 2.
        """
        raise NotImplementedError("TODO(phase 2): build SliceReport from fitted slices.")
