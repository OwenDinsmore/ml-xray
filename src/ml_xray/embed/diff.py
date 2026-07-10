"""Embedding diff (Phase 3 -- scaffolded).

Compare two embedding spaces -- model v1 vs v2, or embeddings over time -- to see
what moved: local structure (k-NN Jaccard overlap), per-point drift, and global
structure (cluster stability via Adjusted Rand Index).

Everything here is typed and documented but raises ``NotImplementedError`` until
Phase 3 lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["EmbedDiffReport", "EmbeddingDiff"]


@dataclass
class EmbedDiffReport:
    """Result of comparing two embedding spaces (Phase 3 -- scaffolded).

    Attributes
    ----------
    neighbor_overlap : float
        Mean Jaccard overlap of k-NN sets before/after (0..1).
    per_point_drift : numpy.ndarray
        How much each point's neighborhood changed.
    cluster_stability : float
        Adjusted Rand Index between clusterings of A and B.
    movers : list
        Ids whose neighborhoods changed the most.
    """

    neighbor_overlap: float = float("nan")
    per_point_drift: np.ndarray = field(default_factory=lambda: np.empty(0))
    cluster_stability: float = float("nan")
    movers: list = field(default_factory=list)

    def plot_projection(self):
        """Linked before/after 2D scatter via the viz backend.

        Raises
        ------
        NotImplementedError
            Always -- Phase 3.
        """
        raise NotImplementedError("TODO(phase 3): linked before/after projection scatter.")

    def to_html(self, path: str) -> None:
        """Render the embedding-diff report to a self-contained HTML file.

        Raises
        ------
        NotImplementedError
            Always -- Phase 3.
        """
        raise NotImplementedError("TODO(phase 3): render EmbedDiffReport to HTML.")


class EmbeddingDiff:
    """Compare two row-aligned embedding matrices.

    Parameters
    ----------
    k : int
        Neighborhood size for k-NN overlap and drift.
    align : str
        Cross-space alignment strategy (``"procrustes"`` or ``"none"``).
    projector : str
        2D projector for the report (``"umap"``, ``"pca"``, or ``"auto"``).
    """

    def __init__(
        self,
        *,
        k: int = 10,
        align: str = "procrustes",
        projector: str = "auto",
    ) -> None:
        self.k = k
        self.align = align
        self.projector = projector

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
            Embedding matrices of shape ``(n, d_a)`` and ``(n, d_b)``,
            row-aligned by id. Dimensions may differ.
        ids : sequence, optional
            Row identifiers, used to label ``movers``.
        labels : sequence, optional
            Optional ground-truth labels for cluster-stability context.

        Returns
        -------
        EmbeddingDiff
            ``self``, fitted.

        Raises
        ------
        NotImplementedError
            Always -- Phase 3.
        """
        raise NotImplementedError(
            "TODO(phase 3): align spaces, compute k-NN overlap, drift, and ARI."
        )

    def report(self) -> EmbedDiffReport:
        """Return the :class:`EmbedDiffReport` after :meth:`fit`.

        Raises
        ------
        NotImplementedError
            Always -- Phase 3.
        """
        raise NotImplementedError("TODO(phase 3): build EmbedDiffReport from fitted diff.")
