"""2D projection for embedding diff (Phase 3 -- scaffolded).

Projects embeddings to 2D for the linked before/after scatter. Uses UMAP when
the ``[embeddings]`` extra is installed, and falls back to PCA (core dependency)
otherwise. The same fitted basis is applied to both A and B so the before/after
scatter is visually comparable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["project_2d", "Projector"]

Projector = Literal["umap", "pca", "auto"]


def project_2d(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    *,
    projector: Projector = "auto",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project two embedding matrices into a shared 2D basis.

    Parameters
    ----------
    emb_a, emb_b : numpy.ndarray
        Row-aligned embedding matrices.
    projector : {"umap", "pca", "auto"}
        Projection method. ``"auto"`` prefers UMAP (``[embeddings]`` extra) and
        falls back to PCA.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(n, 2)`` projections of A and B in the same basis.

    Raises
    ------
    NotImplementedError
        Always -- Phase 3.
    """
    raise NotImplementedError("TODO(phase 3): shared UMAP/PCA 2D projection basis.")
