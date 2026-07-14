"""Feature discretization strategies for slice discovery.

Continuous features are binned before enumerating slices; categoricals are kept
as-is with rare-level bucketing. Strategies: quantile bins, tree-based bins
(supervised on the error signal), or caller-supplied edges. Bin labels are
human-readable strings so slice predicates read like ``tenure=[0.0, 3.0)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ..lint._util import is_numeric

__all__ = [
    "bin_feature",
    "discretize",
    "numeric_range_items",
    "bucket_rare_levels",
    "BinStrategy",
    "MISSING",
    "OTHER",
]

BinStrategy = Literal["quantile", "tree", "custom"]

#: Label used for missing values in a binned feature.
MISSING = "__missing__"
#: Label used for pooled rare categorical levels.
OTHER = "__other__"


def _interval_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        closer = "]" if i == len(edges) - 2 else ")"
        labels.append(f"[{left:.3g}, {right:.3g}{closer}")
    return labels


def bin_feature(
    values: pd.Series,
    *,
    strategy: BinStrategy = "quantile",
    n_bins: int = 4,
    edges: Sequence[float] | None = None,
    error: np.ndarray | None = None,
    seed: int = 0,
) -> pd.Series:
    """Discretize a continuous feature into labeled string bins.

    Parameters
    ----------
    values : pandas.Series
        The continuous feature to bin.
    strategy : {"quantile", "tree", "custom"}
        Binning strategy. ``"tree"`` fits shallow splits against ``error`` and
        falls back to quantile binning when ``error`` is ``None``; ``"custom"``
        uses ``edges``.
    n_bins : int
        Target number of bins for the quantile/tree strategies.
    edges : sequence of float, optional
        Explicit bin edges for the ``"custom"`` strategy.
    error : numpy.ndarray, optional
        Per-row error signal used by the ``"tree"`` strategy.
    seed : int
        Random seed for the tree strategy.

    Returns
    -------
    pandas.Series
        Object dtype bin labels aligned to ``values`` (missing -> ``MISSING``).
    """
    s = values.reset_index(drop=True)
    finite = s[s.notna()].astype(float)
    out = pd.Series([MISSING] * len(s), index=s.index, dtype="object")
    if finite.empty:
        return out

    if strategy == "custom":
        if edges is None:
            raise ValueError("strategy='custom' requires explicit `edges`")
        cut_edges = np.asarray(sorted(edges), dtype=float)
    elif strategy == "tree" and error is not None:
        cut_edges = _tree_edges(finite.to_numpy(), np.asarray(error)[finite.index], n_bins, seed)
    else:
        quantiles = np.linspace(0, 1, n_bins + 1)
        cut_edges = np.unique(np.quantile(finite.to_numpy(), quantiles))

    if cut_edges.size < 2:
        # Degenerate (constant-ish) feature: one bucket.
        out.loc[finite.index] = f"={finite.iloc[0]:.3g}"
        return out

    cut_edges[0] = -np.inf
    cut_edges[-1] = np.inf
    labels = _interval_labels(cut_edges)
    binned = pd.cut(finite, bins=cut_edges, labels=labels, include_lowest=True, right=False)
    out.loc[finite.index] = binned.astype("object")
    # Right-most edge is inf; ensure the max value still lands in the last bin.
    out.loc[finite.index] = out.loc[finite.index].fillna(labels[-1])
    return out


def _tree_edges(values: np.ndarray, error: np.ndarray, n_bins: int, seed: int) -> np.ndarray:
    """Find split points that best separate the error signal."""
    from sklearn.tree import DecisionTreeRegressor

    tree = DecisionTreeRegressor(max_leaf_nodes=max(2, n_bins), random_state=seed)
    tree.fit(values.reshape(-1, 1), error)
    thresholds = tree.tree_.threshold[tree.tree_.feature == 0]
    edges = np.unique(np.concatenate([[values.min()], np.sort(thresholds), [values.max()]]))
    return edges


def _bucket_rare(values: pd.Series, min_count: int) -> pd.Series:
    """Pool rare categorical levels into ``OTHER`` and mark missing as ``MISSING``."""
    s = values.reset_index(drop=True).astype("object")
    counts = s.value_counts(dropna=True)
    keep = set(counts[counts >= min_count].index)
    out = s.where(s.isin(keep), OTHER)
    out = out.where(s.notna(), MISSING)
    return out.astype("object")


# Public alias for use outside this module.
bucket_rare_levels = _bucket_rare


def _fmt_range(lo: float, hi: float) -> str:
    """Human-readable label for a half-open ``[lo, hi)`` range."""
    if lo == -np.inf:
        return f"< {hi:.3g}"
    if hi == np.inf:
        return f">= {lo:.3g}"
    return f"[{lo:.3g}, {hi:.3g})"


def numeric_range_items(
    values: pd.Series,
    *,
    n_bins: int = 4,
    min_support: int = 1,
) -> list[tuple[str, np.ndarray]]:
    """Enumerate contiguous numeric-range predicates for a feature.

    Rather than only single quantile bins, this yields every *contiguous* range
    of quantile bins (e.g. ``< 3``, ``[3, 9)``, ``>= 9``), so a weak region that
    spans several adjacent bins can be reported as one readable range instead of
    fragmented pieces.

    Parameters
    ----------
    values : pandas.Series
        The numeric feature.
    n_bins : int
        Number of quantile bins whose boundaries seed the ranges.
    min_support : int
        Minimum matching rows for a range to be emitted.

    Returns
    -------
    list of (str, numpy.ndarray)
        ``(label, boolean mask)`` pairs. The all-encompassing range is omitted.
    """
    v = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    finite = v[v.notna()]
    if finite.nunique() < 2:
        return []
    edges = np.unique(np.quantile(finite.to_numpy(), np.linspace(0, 1, n_bins + 1)))
    interior = edges[1:-1]
    if interior.size == 0:
        return []
    bounds = np.concatenate([[-np.inf], interior, [np.inf]])
    arr = v.to_numpy(dtype=float)
    notna = v.notna().to_numpy()

    items: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, float]] = set()
    m = len(bounds)
    for i in range(m - 1):
        for j in range(i + 1, m):
            lo, hi = float(bounds[i]), float(bounds[j])
            if lo == -np.inf and hi == np.inf:
                continue  # the whole column is not a predicate
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            mask = notna & (arr >= lo) & (arr < hi)
            support = int(mask.sum())
            if support < min_support or support == int(notna.sum()):
                continue
            items.append((_fmt_range(lo, hi), mask))
    return items


def discretize(
    X: pd.DataFrame,
    *,
    n_bins: int = 4,
    rare_min_count: int = 30,
    error: np.ndarray | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Discretize every column of ``X`` for slice enumeration.

    Numeric columns are binned (tree-based when ``error`` is provided, else
    quantile); categorical columns keep their levels with rare-level pooling.

    Parameters
    ----------
    X : pandas.DataFrame
        Features to slice on.
    n_bins : int
        Target bins per numeric feature.
    rare_min_count : int
        Categorical levels rarer than this are pooled into ``OTHER``.
    error : numpy.ndarray, optional
        Per-row error signal enabling supervised (tree) numeric binning.
    seed : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        All-categorical frame of string bin labels, aligned to ``X`` positionally.
    """
    strategy: BinStrategy = "tree" if error is not None else "quantile"
    cols = {}
    for col in X.columns:
        s = X[col]
        if is_numeric(s):
            cols[col] = bin_feature(
                s, strategy=strategy, n_bins=n_bins, error=error, seed=seed
            ).to_numpy()
        else:
            cols[col] = _bucket_rare(s, rare_min_count).to_numpy()
    return pd.DataFrame(cols)
