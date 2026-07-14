"""Slice discovery / error analysis.

Find where a trained model underperforms, e.g. "F1 is 0.42 on
``region=EU & tenure<3mo`` (n=1,204) vs 0.79 overall." Inspired by Slice Finder
and SliceLens error-deviation metrics; not a reimplementation of any single
paper.

Algorithm
---------
1. Discretize continuous features (quantile / tree bins); keep categoricals with
   rare-level bucketing (:mod:`ml_xray.slices.binning`).
2. Enumerate slices up to ``max_depth`` feature conjunctions with an Apriori-style
   lattice search that only extends frequent predicates and skips low-support
   branches early.
3. Score each slice with the task metric and compute ``delta`` vs the global
   baseline.
4. Test each slice for significance and apply a Benjamini-Hochberg FDR correction
   across every slice tested (:mod:`ml_xray.slices.metrics`).
5. Drop slices whose underperformance is already explained by a more general
   parent, then rank the rest by ``|underperformance| * log(support)`` and return
   ``top_k``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import log

import numpy as np
import pandas as pd

from ..lint._util import infer_task, is_numeric
from .binning import MISSING, bin_feature, bucket_rare_levels, numeric_range_items
from .metrics import Metric, benjamini_hochberg, resolve_metric, slice_pvalue

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
        Signed difference vs baseline (``metric_value - baseline``). Negative for
        a higher-is-better metric on an underperforming slice.
    p_value : float
        Multiple-comparison-corrected significance of the deviation.
    """

    predicate: dict[str, object]
    support: int
    metric_value: float
    baseline: float
    delta: float
    p_value: float

    def describe(self) -> str:
        """Return a readable ``feat=level & feat=level`` predicate string."""
        return " & ".join(f"{k}={v}" for k, v in self.predicate.items())


@dataclass
class SliceReport:
    """Ranked slices where a model underperforms.

    Attributes
    ----------
    slices : list of Slice
        Slices ranked by underperformance x log(support), worst first.
    baseline : float
        Global metric value.
    metric : str
        Name of the metric used.
    """

    slices: list[Slice] = field(default_factory=list)
    baseline: float = float("nan")
    metric: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the report."""
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "slices": [
                {
                    "predicate": s.predicate,
                    "support": s.support,
                    "metric_value": s.metric_value,
                    "baseline": s.baseline,
                    "delta": s.delta,
                    "p_value": s.p_value,
                }
                for s in self.slices
            ],
        }

    def to_html(self, path: str) -> None:
        """Render the slice report to a self-contained HTML file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        from ..report import slice_report_html

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(slice_report_html(self))

    def plot(self):
        """Bar chart of the worst slices via the selected viz backend.

        Returns
        -------
        object
            A matplotlib ``Figure`` when matplotlib is available.

        Raises
        ------
        ImportError
            If no plotting backend is installed.
        """
        from ..report import slice_bar_figure

        return slice_bar_figure(self)

    def __bool__(self) -> bool:
        return bool(self.slices)

    def __len__(self) -> int:
        return len(self.slices)

    def __iter__(self):
        return iter(self.slices)


# One depth-1 predicate item: a (column, level) pair and its row mask.
_Item = tuple[str, object, np.ndarray]


class SliceFinder:
    """Discover underperforming slices via lattice search with pruning.

    Parameters
    ----------
    metric : str or callable
        ``"f1"`` / ``"accuracy"`` / ``"mse"`` / ``"mae"``, a callable
        ``(y_true, y_pred, y_proba) -> float``, or ``"auto"`` (accuracy for
        classification, MSE for regression).
    max_depth : int
        Maximum number of features combined per slice predicate.
    min_support : int
        Minimum rows for a slice to be eligible.
    top_k : int
        Number of slices to return.
    significance : float
        Benjamini-Hochberg-corrected significance level.
    n_bins : int
        Target bins per numeric feature during discretization.
    numeric_ranges : bool
        When ``True`` (default), numeric features produce contiguous *range*
        predicates (``tenure < 3``, ``[3, 9)``, ``>= 9``) rather than only single
        quantile bins, so a weak region spanning several bins reads as one range.
    """

    def __init__(
        self,
        *,
        metric: str | Callable[..., float] = "auto",
        max_depth: int = 2,
        min_support: int = 30,
        top_k: int = 20,
        significance: float = 0.05,
        n_bins: int = 4,
        numeric_ranges: bool = True,
    ) -> None:
        self.metric = metric
        self.max_depth = max_depth
        self.min_support = min_support
        self.top_k = top_k
        self.significance = significance
        self.n_bins = n_bins
        self.numeric_ranges = numeric_ranges
        self._report: SliceReport | None = None

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
            Features to slice on (discretized internally).
        y_true, y_pred : array-like
            Ground-truth labels and model predictions, row-aligned to ``X``.
        y_proba : array-like, optional
            Predicted probabilities/scores, when the metric needs them.
        task : {"classification", "regression"}, optional
            Learning task; inferred from ``y_true`` when omitted.

        Returns
        -------
        SliceFinder
            ``self``, fitted. Call :meth:`report` for results.
        """
        X = X.reset_index(drop=True)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        proba = None if y_proba is None else np.asarray(y_proba)
        n = len(X)
        if not (len(y_true) == len(y_pred) == n):
            raise ValueError("X, y_true and y_pred must be the same length")

        task = task or infer_task(pd.Series(y_true))
        metric = resolve_metric(self.metric, task)
        baseline = metric(y_true, y_pred, proba)

        # Per-row loss (larger = worse) drives significance and tree binning.
        if task == "classification":
            loss = (y_true != y_pred).astype(float)
        else:
            loss = np.abs(y_true.astype(float) - y_pred.astype(float))

        items = self._depth1_items(X, loss)

        tested = self._lattice_search(items, n, y_true, y_pred, proba, loss, metric, baseline, task)
        self._report = self._build_report(tested, baseline, metric)
        return self

    def report(self) -> SliceReport:
        """Return the ranked :class:`SliceReport` produced by :meth:`fit`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._report is None:
            raise RuntimeError("call fit() before report()")
        return self._report

    # -- internals --------------------------------------------------------

    def _depth1_items(self, X: pd.DataFrame, loss: np.ndarray) -> list[_Item]:
        """Build depth-1 predicate items per feature.

        Numeric features become contiguous range predicates (or single quantile
        bins when ``numeric_ranges`` is off); categoricals become rare-bucketed
        level predicates.
        """
        items: list[_Item] = []
        for col in X.columns:
            series = X[col]
            if is_numeric(series):
                items.extend(self._numeric_items(col, series, loss))
            else:
                bucketed = bucket_rare_levels(series, self.min_support)
                for level, count in bucketed.value_counts().items():
                    if count < self.min_support:
                        continue
                    items.append((col, level, (bucketed == level).to_numpy()))
        return items

    def _numeric_items(self, col: str, series: pd.Series, loss: np.ndarray) -> list[_Item]:
        if self.numeric_ranges:
            ranges = numeric_range_items(series, n_bins=self.n_bins, min_support=self.min_support)
            return [(col, label, mask) for label, mask in ranges]
        binned = bin_feature(series, strategy="quantile", n_bins=self.n_bins)
        out: list[_Item] = []
        for level, count in binned.value_counts().items():
            if count < self.min_support or level == MISSING:
                continue
            out.append((col, level, (binned == level).to_numpy()))
        return out

    def _lattice_search(
        self, items, n, y_true, y_pred, proba, loss, metric, baseline, task
    ) -> dict:
        tested: dict[frozenset, dict] = {}

        def evaluate(pred_set: frozenset, mask: np.ndarray) -> None:
            support = int(mask.sum())
            mv = metric(y_true[mask], y_pred[mask], None if proba is None else proba[mask])
            # Metrics like ROC-AUC are undefined on single-class slices (NaN); a
            # NaN-metric slice can't be scored, so record it only as visited.
            if not np.isfinite(mv):
                tested[pred_set] = {"underperf": float("-inf"), "p_value": 1.0, "mask": mask}
                return
            delta = mv - baseline
            underperf = (baseline - mv) if metric.greater_is_better else (mv - baseline)
            pval = slice_pvalue(mask, loss, task)
            tested[pred_set] = {
                "predicate": dict(sorted(pred_set)),
                "mask": mask,
                "support": support,
                "metric_value": float(mv),
                "delta": float(delta),
                "underperf": float(underperf),
                "p_value": float(pval),
            }

        frontier: list[tuple[frozenset, np.ndarray]] = []
        for col, level, mask in items:
            pred = frozenset({(col, level)})
            evaluate(pred, mask)
            frontier.append((pred, mask))

        for _depth in range(2, self.max_depth + 1):
            new_frontier: list[tuple[frozenset, np.ndarray]] = []
            for pred, mask in frontier:
                used = {c for c, _ in pred}
                for col, level, imask in items:
                    if col in used:
                        continue
                    new_pred = pred | {(col, level)}
                    if new_pred in tested:
                        continue
                    new_mask = mask & imask
                    if int(new_mask.sum()) < self.min_support:
                        continue
                    evaluate(new_pred, new_mask)
                    new_frontier.append((new_pred, new_mask))
            if not new_frontier:
                break
            frontier = new_frontier
        return tested

    def _build_report(self, tested: dict, baseline: float, metric: Metric) -> SliceReport:
        if not tested:
            return SliceReport(slices=[], baseline=float(baseline), metric=metric.name)

        keys = list(tested)
        pvals = np.array([tested[k]["p_value"] for k in keys])
        significant = benjamini_hochberg(pvals, alpha=self.significance)
        sig_keys = {k for k, ok in zip(keys, significant) if ok and tested[k]["underperf"] > 0}

        kept = self._prune_by_parents(sig_keys, tested)
        ranked = sorted(
            kept,
            key=lambda k: tested[k]["underperf"] * log(max(tested[k]["support"], 2)),
            reverse=True,
        )
        slices = [
            Slice(
                predicate=tested[k]["predicate"],
                support=tested[k]["support"],
                metric_value=tested[k]["metric_value"],
                baseline=float(baseline),
                delta=tested[k]["delta"],
                p_value=tested[k]["p_value"],
            )
            for k in ranked[: self.top_k]
        ]
        return SliceReport(slices=slices, baseline=float(baseline), metric=metric.name)

    @staticmethod
    def _prune_by_parents(sig_keys: set, tested: dict) -> set:
        """Drop a slice when a more general parent is at least as underperforming.

        A depth-d slice is kept only if every strict subset predicate that was
        also flagged has strictly lower underperformance -- i.e. the extra
        condition genuinely concentrates the error further.
        """
        kept = set()
        for key in sig_keys:
            underperf = tested[key]["underperf"]
            redundant = False
            for item in key:
                parent = key - {item}
                if not parent:
                    continue
                if parent in sig_keys and tested[parent]["underperf"] >= underperf:
                    redundant = True
                    break
            if not redundant:
                kept.add(key)
        return kept
