"""Feature discretization strategies for slice discovery (Phase 2 -- scaffolded).

Continuous features are binned before enumerating slices; categoricals are kept
as-is with rare-level bucketing. Strategies: quantile bins, tree-based bins
(supervised on the error signal), or caller-supplied edges.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["bin_feature", "BinStrategy"]

BinStrategy = Literal["quantile", "tree", "custom"]


def bin_feature(
    values: pd.Series,
    *,
    strategy: BinStrategy = "quantile",
    n_bins: int = 4,
    edges: Sequence[float] | None = None,
    error: np.ndarray | None = None,
) -> pd.Series:
    """Discretize a continuous feature into labeled bins.

    Parameters
    ----------
    values : pandas.Series
        The continuous feature to bin.
    strategy : {"quantile", "tree", "custom"}
        Binning strategy. ``"tree"`` fits shallow splits against ``error``;
        ``"custom"`` uses ``edges``.
    n_bins : int
        Target number of bins for quantile/tree strategies.
    edges : sequence of float, optional
        Explicit bin edges for the ``"custom"`` strategy.
    error : numpy.ndarray, optional
        Per-row error signal used by the ``"tree"`` strategy.

    Returns
    -------
    pandas.Series
        Categorical bin labels aligned to ``values``.

    Raises
    ------
    NotImplementedError
        Always -- Phase 2.
    """
    raise NotImplementedError("TODO(phase 2): quantile/tree/custom feature binning.")
