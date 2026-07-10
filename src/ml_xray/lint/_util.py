"""Shared helpers for lint checks: task inference, column typing, and building a
numeric design matrix from a mixed DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "infer_task",
    "numeric_columns",
    "categorical_columns",
    "is_numeric",
    "design_matrix",
]


def is_numeric(s: pd.Series) -> bool:
    """Return ``True`` if ``s`` has a numeric (non-boolean-object) dtype."""
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def numeric_columns(df: pd.DataFrame, exclude: str | None = None) -> list[str]:
    """List numeric column names, optionally excluding one column."""
    cols = [c for c in df.columns if is_numeric(df[c])]
    if exclude is not None and exclude in cols:
        cols.remove(exclude)
    return cols


def categorical_columns(df: pd.DataFrame, exclude: str | None = None) -> list[str]:
    """List non-numeric (categorical/object/boolean) column names."""
    cols = [c for c in df.columns if not is_numeric(df[c])]
    if exclude is not None and exclude in cols:
        cols.remove(exclude)
    return cols


def infer_task(y: pd.Series) -> str:
    """Infer the learning task from a target column.

    Uses a simple heuristic: a float dtype with many distinct values is treated
    as regression; everything else (integers with few levels, strings, booleans,
    categoricals) is treated as classification.

    Parameters
    ----------
    y : pandas.Series
        The target column.

    Returns
    -------
    {"classification", "regression"}
        The inferred task.
    """
    y = y.dropna()
    n = len(y)
    n_unique = y.nunique()
    if n == 0:
        return "classification"
    if not is_numeric(y):
        return "classification"
    if pd.api.types.is_float_dtype(y):
        # Floats that only take a couple of values are really labels.
        if n_unique <= max(2, min(20, int(0.05 * n))):
            return "classification"
        return "regression"
    # Integer-like: few distinct levels -> classification, otherwise regression.
    if n_unique <= max(2, min(20, int(0.05 * n))):
        return "classification"
    return "regression"


def design_matrix(
    df: pd.DataFrame,
    *,
    exclude: str | None = None,
    max_cardinality: int = 50,
) -> tuple[np.ndarray, list[str]]:
    """Build a dense numeric matrix from mixed feature columns.

    Numeric columns are median-imputed; low-cardinality categoricals are
    one-hot encoded; high-cardinality categoricals are dropped. This is a
    deliberately small, dependency-light encoder used by model-based checks
    (e.g. label noise), not a general preprocessing pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature frame (the caller should exclude the target).
    exclude : str, optional
        A column to leave out (e.g. the target if still present).
    max_cardinality : int
        Categoricals with more distinct levels than this are dropped rather than
        one-hot encoded, to avoid exploding the feature space.

    Returns
    -------
    (numpy.ndarray, list of str)
        The ``(n_rows, n_features)`` matrix and the generated feature names.
    """
    frames: list[pd.DataFrame] = []
    names: list[str] = []

    for col in df.columns:
        if exclude is not None and col == exclude:
            continue
        s = df[col]
        if is_numeric(s):
            filled = s.astype(float)
            median = filled.median()
            filled = filled.fillna(median if pd.notna(median) else 0.0)
            frames.append(filled.to_frame(col))
            names.append(col)
        else:
            if s.nunique(dropna=True) > max_cardinality:
                continue
            dummies = pd.get_dummies(s.astype("object"), prefix=col, dummy_na=False)
            if dummies.shape[1] == 0:
                continue
            frames.append(dummies.astype(float))
            names.extend(dummies.columns.tolist())

    if not frames:
        return np.zeros((len(df), 0), dtype=float), []

    matrix = pd.concat(frames, axis=1)
    return matrix.to_numpy(dtype=float), names
