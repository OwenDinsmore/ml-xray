"""Synthetic dataset builders that *plant* known defects.

Each builder returns a DataFrame (and, where relevant, the ground-truth indices
or columns of the planted defect) so tests can assert both that the right check
fires and that it identifies the planted rows/columns. Detection precision/recall
on these planted defects is the core test signal for the lint module.

All builders take a NumPy ``Generator`` (or seed) so datasets are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _rng(seed: int | np.random.Generator) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


@dataclass
class Planted:
    """A synthetic dataset with metadata describing the planted defect."""

    df: pd.DataFrame
    target: str | None = None
    split: pd.Series | None = None
    columns: list[str] | None = None
    rows: list[int] | None = None
    meta: dict | None = None


def clean_classification(n: int = 600, seed: int = 0) -> Planted:
    """A defect-free classification dataset (balanced, learnable, no leakage/drift).

    The label is a near-deterministic function of two numeric features with only
    a thin boundary of ambiguity, so a model can fit it well and the label-noise
    check does not fire. No single feature is close enough to the target to look
    like leakage, and classes are roughly balanced.
    """
    rng = _rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.4, 0.35, 0.25])
    # Deterministic, learnable boundary split across two features: no genuine
    # label noise, yet no single feature is a near-copy of the target.
    score = x1 - x2
    y = (score > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})
    return Planted(df=df, target="y")


def with_leaked_column(n: int = 600, seed: int = 1) -> Planted:
    """Plant a numeric feature almost perfectly correlated with the target."""
    base = clean_classification(n, seed)
    df = base.df.copy()
    rng = _rng(seed + 100)
    # A near-copy of the target: leaks it almost perfectly.
    df["leak"] = df["y"].to_numpy().astype(float) + rng.normal(0, 1e-4, n)
    return Planted(df=df, target="y", columns=["leak"])


def with_deterministic_leak(n: int = 600, seed: int = 2) -> Planted:
    """Plant a categorical feature that deterministically encodes the target."""
    base = clean_classification(n, seed)
    df = base.df.copy()
    # Each label maps to a distinct token -> perfect purity, not per-row id.
    df["leak_cat"] = df["y"].map({0: "neg", 1: "pos"}).astype("object")
    return Planted(df=df, target="y", columns=["leak_cat"])


def with_train_test_overlap(n: int = 600, dup: int = 40, seed: int = 3) -> Planted:
    """Duplicate ``dup`` train rows into the test split (cross-split leakage)."""
    base = clean_classification(n, seed)
    df = base.df.reset_index(drop=True)
    split = np.array(["train"] * n, dtype=object)
    rng = _rng(seed)
    test_idx = rng.choice(n, size=n // 3, replace=False)
    split[test_idx] = "test"

    # Copy `dup` train rows and append them as test rows -> identical across split.
    train_pool = np.setdiff1d(np.arange(n), test_idx)
    copied = rng.choice(train_pool, size=dup, replace=False)
    dup_rows = df.iloc[copied].reset_index(drop=True)
    df = pd.concat([df, dup_rows], ignore_index=True)
    split = np.concatenate([split, np.array(["test"] * dup, dtype=object)])

    planted_rows = list(range(n, n + dup))
    return Planted(
        df=df,
        target="y",
        split=pd.Series(split, name="split"),
        rows=planted_rows,
        meta={"n_original": n, "dup": dup},
    )


def with_drift(n: int = 900, seed: int = 4) -> Planted:
    """Shift a numeric feature in the test split relative to train."""
    rng = _rng(seed)
    split = np.where(np.arange(n) < (2 * n) // 3, "train", "test").astype(object)
    stable = rng.normal(0, 1, n)
    drifted = rng.normal(0, 1, n)
    test_mask = split == "test"
    drifted[test_mask] += 3.0  # large mean shift only in test
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"stable": stable, "drifted": drifted, "y": y})
    return Planted(
        df=df,
        target="y",
        split=pd.Series(split, name="split"),
        columns=["drifted"],
    )


def with_label_noise(n: int = 800, flip: int = 40, seed: int = 5) -> Planted:
    """Flip ``flip`` labels in an otherwise well-separated classification set."""
    rng = _rng(seed)
    # Two well-separated Gaussian blobs -> labels are easy, so flips stand out.
    half = n // 2
    x = np.concatenate([rng.normal(-3, 0.6, half), rng.normal(3, 0.6, n - half)])
    x2 = np.concatenate([rng.normal(-3, 0.6, half), rng.normal(3, 0.6, n - half)])
    y = np.concatenate([np.zeros(half, int), np.ones(n - half, int)])
    order = rng.permutation(n)
    x, x2, y = x[order], x2[order], y[order]

    flip_idx = rng.choice(n, size=flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    df = pd.DataFrame({"f1": x, "f2": x2, "y": y_noisy})
    return Planted(df=df, target="y", rows=sorted(flip_idx.tolist()))


def with_duplicates(n: int = 400, dup: int = 30, seed: int = 6) -> Planted:
    """Append ``dup`` exact duplicate rows to a clean dataset."""
    base = clean_classification(n, seed)
    df = base.df.reset_index(drop=True)
    rng = _rng(seed)
    src = rng.choice(n, size=dup, replace=False)
    dup_rows = df.iloc[src].reset_index(drop=True)
    out = pd.concat([df, dup_rows], ignore_index=True)
    planted_rows = list(range(n, n + dup))
    return Planted(df=out, target="y", rows=planted_rows, meta={"n_original": n})


def with_imbalance(n: int = 1000, minority: int = 5, seed: int = 7) -> Planted:
    """A classification target with a tiny minority class."""
    rng = _rng(seed)
    y = np.zeros(n, int)
    y[:minority] = 1
    rng.shuffle(y)
    x1 = rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "y": y})
    return Planted(df=df, target="y", meta={"minority": minority})


def with_outliers_and_nulls(n: int = 500, seed: int = 8) -> Planted:
    """Plant extreme numeric outliers, a high-null column, and a constant column."""
    rng = _rng(seed)
    val = rng.normal(0, 1, n)
    outlier_rows = [3, 17, 42, 99, 250]
    val[outlier_rows] = 500.0  # far outside the robust-z band
    mostly_null = np.where(np.arange(n) < int(0.95 * n), np.nan, 1.0)
    constant = np.ones(n)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"val": val, "mostly_null": mostly_null, "constant": constant, "y": y})
    return Planted(
        df=df,
        target="y",
        rows=sorted(outlier_rows),
        columns=["val", "mostly_null", "constant"],
    )


@dataclass
class PlantedSlice:
    """Synthetic predictions with a planted underperforming region."""

    X: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    weak_predicate: dict
    weak_rows: np.ndarray


def weak_region_predictions(n: int = 2000, seed: int = 10) -> PlantedSlice:
    """Predictions that are accurate everywhere except a planted weak region.

    The model is near-perfect except on ``region == "EU"`` (and worst on the
    younger-tenure part of it), so slice discovery should surface that region.
    """
    rng = _rng(seed)
    region = rng.choice(["EU", "US", "APAC"], size=n, p=[0.3, 0.4, 0.3])
    tenure = rng.uniform(0, 24, n)
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()

    weak = region == "EU"
    # Corrupt ~55% of predictions inside the weak region only.
    corrupt = weak & (rng.uniform(size=n) < 0.55)
    y_pred[corrupt] = 1 - y_pred[corrupt]

    X = pd.DataFrame({"region": region, "tenure": tenure})
    return PlantedSlice(
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        weak_predicate={"region": "EU"},
        weak_rows=np.flatnonzero(weak),
    )


def clean_predictions(n: int = 2000, seed: int = 11) -> PlantedSlice:
    """Predictions with uniform error and no underperforming region."""
    rng = _rng(seed)
    region = rng.choice(["EU", "US", "APAC"], size=n)
    tenure = rng.uniform(0, 24, n)
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    # Uniform 10% error spread everywhere, correlated with no feature.
    corrupt = rng.uniform(size=n) < 0.1
    y_pred[corrupt] = 1 - y_pred[corrupt]
    X = pd.DataFrame({"region": region, "tenure": tenure})
    return PlantedSlice(
        X=X, y_true=y_true, y_pred=y_pred, weak_predicate={}, weak_rows=np.empty(0, int)
    )


@dataclass
class PlantedEmbed:
    """A pair of embedding spaces with a known relationship."""

    emb_a: np.ndarray
    emb_b: np.ndarray
    moved_ids: np.ndarray


def clustered_embedding(n: int = 300, d: int = 16, seed: int = 20) -> np.ndarray:
    """A clustered embedding matrix (well-separated Gaussian blobs)."""
    rng = _rng(seed)
    n_clusters = 6
    centers = rng.normal(0, 8, (n_clusters, d))
    assign = rng.integers(0, n_clusters, n)
    return centers[assign] + rng.normal(0, 1, (n, d))


def identical_spaces(n: int = 300, seed: int = 20) -> PlantedEmbed:
    """Two identical spaces (overlap should be ~1)."""
    a = clustered_embedding(n, seed=seed)
    return PlantedEmbed(emb_a=a, emb_b=a.copy(), moved_ids=np.empty(0, int))


def unrelated_spaces(n: int = 300, d: int = 16, seed: int = 20) -> PlantedEmbed:
    """Two independent spaces (overlap should be ~0)."""
    a = clustered_embedding(n, d, seed=seed)
    b = clustered_embedding(n, d, seed=seed + 999)
    return PlantedEmbed(emb_a=a, emb_b=b, moved_ids=np.arange(n))


def partially_perturbed_spaces(
    n: int = 300, d: int = 16, moved: int = 30, seed: int = 20
) -> PlantedEmbed:
    """Space B equals A except a known subset of points is moved far away."""
    rng = _rng(seed)
    a = clustered_embedding(n, d, seed=seed)
    b = a.copy()
    moved_ids = rng.choice(n, size=moved, replace=False)
    b[moved_ids] += rng.normal(0, 12, (moved, d))
    return PlantedEmbed(emb_a=a, emb_b=b, moved_ids=np.sort(moved_ids))


def with_temporal_leakage(n: int = 800, leak: int = 40, seed: int = 30) -> Planted:
    """Plant train rows dated after the test split plus a time-proxy feature.

    A proper temporal split has all train timestamps before all test timestamps.
    Here ``leak`` train rows are given future timestamps (overlapping test), and
    a ``proxy`` feature is made monotonic with time so it acts as a surrogate
    timestamp.
    """
    rng = _rng(seed)
    ts = np.arange(n, dtype=float)
    split = np.where(ts < (2 * n) // 3, "train", "test").astype(object)
    train_pos = np.flatnonzero(split == "train")
    leak_rows = rng.choice(train_pos, size=leak, replace=False)
    ts[leak_rows] = float(n) + 50  # dated into the future, inside the test window
    proxy = ts * 1.5 + rng.normal(0, 0.01, n)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "proxy": proxy,
            "ts": ts,
            "y": rng.integers(0, 2, n),
        }
    )
    return Planted(
        df=df,
        target="y",
        split=pd.Series(split, name="split"),
        columns=["proxy", "ts"],
        rows=sorted(int(i) for i in leak_rows),
        meta={"time": "ts"},
    )
