"""Exact and near-duplicate row detection.

Exact duplicates are found with a plain hash of feature rows. Near-duplicates
are found with a small, dependency-free MinHash + LSH implementation over each
row's set of ``column=value`` tokens: rows are shingled, hashed into a compact
signature, banded into LSH buckets, and candidate pairs are confirmed by
estimated Jaccard similarity before being reported as clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Check, Finding, LintContext, Severity, register_check

__all__ = ["DuplicatesCheck"]

_NUM_PERM = 64
_BANDS = 16  # rows-per-band = _NUM_PERM / _BANDS = 4
_JACCARD_THRESHOLD = 0.8
_MAX_MERSENNE = (1 << 61) - 1


def _row_tokens(df: pd.DataFrame) -> list[frozenset[int]]:
    """Turn each row into a set of hashed ``column=value`` tokens."""
    tokens: list[list[int]] = [[] for _ in range(len(df))]
    for col in df.columns:
        col_hash = pd.util.hash_array(df[col].astype("object").to_numpy())
        name_seed = hash(col) & 0xFFFFFFFF
        for i, h in enumerate(col_hash):
            tokens[i].append(int(h) ^ name_seed)
    return [frozenset(t) for t in tokens]


def _minhash_signatures(token_sets: list[frozenset[int]], seed: int) -> np.ndarray:
    """Compute an ``(n_rows, _NUM_PERM)`` MinHash signature matrix."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MAX_MERSENNE, size=_NUM_PERM, dtype=np.int64)
    b = rng.integers(0, _MAX_MERSENNE, size=_NUM_PERM, dtype=np.int64)
    sig = np.full((len(token_sets), _NUM_PERM), _MAX_MERSENNE, dtype=np.int64)
    for i, tokens in enumerate(token_sets):
        if not tokens:
            continue
        vals = np.fromiter((t & _MAX_MERSENNE for t in tokens), dtype=np.int64, count=len(tokens))
        # (a * x + b) mod p, one column per permutation; take the per-column min.
        hashed = (np.outer(vals, a) + b) % _MAX_MERSENNE
        sig[i] = hashed.min(axis=0)
    return sig


class _Union:
    """Tiny union-find for grouping candidate rows into clusters."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


@register_check
class DuplicatesCheck(Check):
    """Report exact duplicate rows and near-duplicate clusters."""

    name = "duplicates"
    description = "Exact duplicate rows and near-duplicate clusters via MinHash + LSH."

    def run(self, ctx: LintContext) -> list[Finding]:
        df = ctx.df[ctx.feature_columns].reset_index(drop=True)
        if df.shape[1] == 0 or len(df) < 2:
            return []

        findings: list[Finding] = []
        exact_rows = self._exact_duplicates(df)
        if exact_rows:
            findings.append(
                Finding(
                    check=self.name,
                    severity=Severity.WARN,
                    message=f"{len(exact_rows)} exact duplicate rows (beyond first occurrence).",
                    column=None,
                    detail={"n_exact_duplicates": len(exact_rows), "kind": "exact"},
                    rows=exact_rows,
                )
            )

        near_clusters = self._near_duplicate_clusters(df, ctx.seed, exclude=set(exact_rows))
        if near_clusters:
            involved = sorted({r for cluster in near_clusters for r in cluster})
            findings.append(
                Finding(
                    check=self.name,
                    severity=Severity.INFO,
                    message=(
                        f"{len(near_clusters)} near-duplicate clusters "
                        f"({len(involved)} rows, Jaccard >= {_JACCARD_THRESHOLD})."
                    ),
                    column=None,
                    detail={
                        "n_clusters": len(near_clusters),
                        "clusters": [list(map(int, c)) for c in near_clusters[:50]],
                        "kind": "near",
                    },
                    rows=involved,
                )
            )
        return findings

    @staticmethod
    def _exact_duplicates(df: pd.DataFrame) -> list[int]:
        dup_mask = df.duplicated(keep="first").to_numpy()
        return [int(i) for i in np.flatnonzero(dup_mask)]

    def _near_duplicate_clusters(
        self, df: pd.DataFrame, seed: int, *, exclude: set[int]
    ) -> list[list[int]]:
        token_sets = _row_tokens(df)
        sig = _minhash_signatures(token_sets, seed)
        rows_per_band = _NUM_PERM // _BANDS

        # LSH: rows agreeing on a whole band land in the same bucket -> candidates.
        candidates: set[tuple[int, int]] = set()
        for band in range(_BANDS):
            start = band * rows_per_band
            block = sig[:, start : start + rows_per_band]
            buckets: dict[tuple[int, ...], list[int]] = {}
            for i in range(block.shape[0]):
                buckets.setdefault(tuple(block[i].tolist()), []).append(i)
            for members in buckets.values():
                if len(members) > 1:
                    for a_idx in range(len(members)):
                        for b_idx in range(a_idx + 1, len(members)):
                            candidates.add((members[a_idx], members[b_idx]))

        uf = _Union(len(df))
        matched = False
        for i, j in candidates:
            est = float(np.mean(sig[i] == sig[j]))
            if est >= _JACCARD_THRESHOLD:
                uf.union(i, j)
                matched = True
        if not matched:
            return []

        groups: dict[int, list[int]] = {}
        for i in range(len(df)):
            groups.setdefault(uf.find(i), []).append(i)
        clusters = [sorted(g) for g in groups.values() if len(g) > 1]
        # Drop clusters that are only exact-duplicate rows already reported.
        clusters = [c for c in clusters if not set(c).issubset(exclude)]
        return sorted(clusters, key=lambda c: (-len(c), c[0]))
