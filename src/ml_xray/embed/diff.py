"""Embedding diff.

Compare two embedding spaces -- model v1 vs v2, or embeddings over time -- to see
what moved:

- **local structure**: mean Jaccard overlap of each point's k-NN set before/after;
- **per-point drift**: how much each point's neighborhood changed (``1 - Jaccard``);
- **global structure**: cluster stability via the Adjusted Rand Index between
  KMeans clusterings of A and B;
- **movers**: the ids whose neighborhoods changed most.

Spaces may differ in dimensionality; neighborhood overlap is dimension-free, so
alignment is only needed for the projection scatter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .._optional import optional_import

__all__ = ["EmbedDiffReport", "EmbeddingDiff"]

Backend = Literal["auto", "exact", "approx"]

# Above this row count, "auto" prefers the approximate backend when available.
_APPROX_THRESHOLD = 5000


def _resolve_backend(backend: Backend, n: int) -> str:
    """Pick the concrete k-NN backend, honoring availability for ``"auto"``."""
    if backend == "exact":
        return "exact"
    have_pynndescent = optional_import("pynndescent") is not None
    if backend == "approx":
        if not have_pynndescent:
            raise ImportError(
                "backend='approx' needs pynndescent (bundled with the [embeddings] "
                'extra). Install it with: pip install "ml-xray[embeddings]"'
            )
        return "approx"
    # auto: approximate only pays off on large sets, and only if available.
    return "approx" if (have_pynndescent and n > _APPROX_THRESHOLD) else "exact"


def _drop_self(idx: np.ndarray, k: int) -> np.ndarray:
    """Remove each row's own index from its neighbor list and keep ``k`` columns."""
    out = np.empty((idx.shape[0], k), dtype=int)
    for i in range(idx.shape[0]):
        row = idx[i][idx[i] != i][:k]
        if row.size < k:  # pad degenerate rows by repeating the last neighbor
            row = np.pad(row, (0, k - row.size), mode="edge") if row.size else np.full(k, i)
        out[i] = row
    return out


def _knn_indices(emb: np.ndarray, k: int, *, backend: str = "exact", seed: int = 0) -> np.ndarray:
    """Return each row's ``k`` nearest-neighbor indices (excluding self).

    Parameters
    ----------
    emb : numpy.ndarray
        Embedding matrix.
    k : int
        Neighborhood size.
    backend : {"exact", "approx"}
        ``"exact"`` uses scikit-learn; ``"approx"`` uses pynndescent (NN-descent)
        for large matrices.
    seed : int
        Random seed for the approximate index.
    """
    n = emb.shape[0]
    k_eff = min(k, n - 1)
    if backend == "approx":
        from pynndescent import NNDescent

        index = NNDescent(emb, n_neighbors=k_eff + 1, random_state=seed)
        idx, _ = index.neighbor_graph
        return _drop_self(np.asarray(idx), k_eff)

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)
    return _drop_self(idx, k_eff)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    union = len(sa | sb)
    if union == 0:
        return 1.0
    return len(sa & sb) / union


@dataclass
class EmbedDiffReport:
    """Result of comparing two embedding spaces.

    Attributes
    ----------
    neighbor_overlap : float
        Mean Jaccard overlap of k-NN sets before/after (0..1).
    per_point_drift : numpy.ndarray
        Per-point neighborhood change (``1 - Jaccard``), one value per row.
    cluster_stability : float
        Adjusted Rand Index between KMeans clusterings of A and B (-1..1).
    movers : list
        Ids whose neighborhoods changed most, worst first.
    ids : list
        Row ids in the original order.
    k : int
        Neighborhood size used.
    backend : str
        The k-NN backend actually used (``"exact"`` or ``"approx"``).
    """

    neighbor_overlap: float = float("nan")
    per_point_drift: np.ndarray = field(default_factory=lambda: np.empty(0))
    cluster_stability: float = float("nan")
    movers: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    k: int = 0
    backend: str = "exact"
    _emb_a: np.ndarray | None = field(default=None, repr=False)
    _emb_b: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary (per-point drift is summarized)."""
        drift = self.per_point_drift
        return {
            "neighbor_overlap": self.neighbor_overlap,
            "cluster_stability": self.cluster_stability,
            "k": self.k,
            "backend": self.backend,
            "n_points": int(drift.size),
            "mean_drift": float(np.mean(drift)) if drift.size else float("nan"),
            "max_drift": float(np.max(drift)) if drift.size else float("nan"),
            "movers": list(self.movers),
        }

    def plot_projection(self, seed: int = 0, *, interactive: bool = False):
        """Linked before/after 2D scatter of the two embedding spaces.

        Parameters
        ----------
        seed : int
            Seed for the projection.
        interactive : bool
            Return an interactive plotly figure (needs plotly) with per-point
            hover instead of a static matplotlib figure.

        Returns
        -------
        object
            A plotly or matplotlib ``Figure``.

        Raises
        ------
        ImportError
            If the required plotting backend is not installed.
        RuntimeError
            If the source embeddings were not retained on the report.
        """
        if interactive:
            from ..report import embed_projection_figure_plotly

            return embed_projection_figure_plotly(self, seed=seed)
        from ..report import embed_projection_figure

        return embed_projection_figure(self, seed=seed)

    def to_html(self, path: str) -> None:
        """Render the embedding-diff report to a self-contained HTML file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        from ..report import embed_report_html

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(embed_report_html(self))


class EmbeddingDiff:
    """Compare two row-aligned embedding matrices.

    Parameters
    ----------
    k : int
        Neighborhood size for k-NN overlap and drift.
    align : {"procrustes", "none"}
        Cross-space alignment used for the projection scatter. Neighborhood
        metrics are computed in each space natively and do not require alignment.
    projector : {"umap", "pca", "auto"}
        2D projector for the report scatter.
    n_clusters : int
        Number of clusters for the cluster-stability (ARI) metric.
    backend : {"auto", "exact", "approx"}
        k-NN backend. ``"exact"`` uses scikit-learn; ``"approx"`` uses
        pynndescent (from the ``[embeddings]`` extra) for large sets; ``"auto"``
        picks approximate only when it is available and ``n`` is large.
    seed : int
        Random seed for the approximate index.
    """

    def __init__(
        self,
        *,
        k: int = 10,
        align: str = "procrustes",
        projector: str = "auto",
        n_clusters: int = 8,
        backend: Backend = "auto",
        seed: int = 0,
    ) -> None:
        self.k = k
        self.align = align
        self.projector = projector
        self.n_clusters = n_clusters
        self.backend = backend
        self.seed = seed
        self._report: EmbedDiffReport | None = None

    def fit(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        *,
        ids: Sequence | None = None,
        labels: Sequence | None = None,
    ) -> EmbeddingDiff:
        """Compute the diff between two embedding spaces.

        Parameters
        ----------
        emb_a, emb_b : numpy.ndarray
            Embedding matrices of shape ``(n, d_a)`` and ``(n, d_b)``, row-aligned
            by id. Dimensions may differ.
        ids : sequence, optional
            Row identifiers (defaults to ``range(n)``), used to label ``movers``.
        labels : sequence, optional
            Unused placeholder for future label-aware diagnostics.

        Returns
        -------
        EmbeddingDiff
            ``self``, fitted. Call :meth:`report` for results.

        Raises
        ------
        ValueError
            If the two matrices have different row counts.
        """
        a = np.asarray(emb_a, dtype=float)
        b = np.asarray(emb_b, dtype=float)
        if a.shape[0] != b.shape[0]:
            raise ValueError("emb_a and emb_b must have the same number of rows")
        n = a.shape[0]
        ids_list = list(ids) if ids is not None else list(range(n))

        backend = _resolve_backend(self.backend, n)
        knn_a = _knn_indices(a, self.k, backend=backend, seed=self.seed)
        knn_b = _knn_indices(b, self.k, backend=backend, seed=self.seed)
        per_point_jaccard = np.array([_jaccard(knn_a[i], knn_b[i]) for i in range(n)])
        per_point_drift = 1.0 - per_point_jaccard
        neighbor_overlap = float(np.mean(per_point_jaccard)) if n else float("nan")

        cluster_stability = self._cluster_stability(a, b)
        n_movers = min(n, max(10, int(0.1 * n)))
        order = np.argsort(-per_point_drift)[:n_movers]
        movers = [ids_list[i] for i in order]

        self._report = EmbedDiffReport(
            neighbor_overlap=neighbor_overlap,
            per_point_drift=per_point_drift,
            cluster_stability=cluster_stability,
            movers=movers,
            ids=ids_list,
            k=min(self.k, max(n - 1, 0)),
            backend=backend,
            _emb_a=a,
            _emb_b=b,
        )
        return self

    def report(self) -> EmbedDiffReport:
        """Return the :class:`EmbedDiffReport` produced by :meth:`fit`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._report is None:
            raise RuntimeError("call fit() before report()")
        return self._report

    def _cluster_stability(self, a: np.ndarray, b: np.ndarray) -> float:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        n = a.shape[0]
        n_clusters = min(self.n_clusters, n)
        if n_clusters < 2:
            return float("nan")
        km_a = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(a)
        km_b = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(b)
        return float(adjusted_rand_score(km_a, km_b))
