"""2D projection for embedding diff.

Projects two embedding matrices into a *shared* 2D basis so the before/after
scatter is visually comparable. Uses UMAP when the ``[embeddings]`` extra is
installed, and falls back to PCA (a core dependency) otherwise. The basis is fit
on the stacked rows of both matrices and applied to each.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .._optional import optional_import
from .align import pad_to_common_dim

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
        Row-aligned embedding matrices (padded to a common dimension internally).
    projector : {"umap", "pca", "auto"}
        Projection method. ``"auto"`` prefers UMAP (``[embeddings]`` extra) and
        falls back to PCA.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(n, 2)`` projections of A and B in the same basis.
    """
    a, b = pad_to_common_dim(np.asarray(emb_a, float), np.asarray(emb_b, float))
    stacked = np.vstack([a, b])
    n = a.shape[0]

    backend = projector
    if backend == "auto":
        backend = "umap" if optional_import("umap") is not None else "pca"

    if backend == "umap":
        coords = _umap_project(stacked, seed)
    else:
        coords = _pca_project(stacked, seed)
    return coords[:n], coords[n:]


def _pca_project(stacked: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_components = 2 if stacked.shape[1] >= 2 else 1
    coords = PCA(n_components=n_components, random_state=seed).fit_transform(stacked)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    return coords


def _umap_project(stacked: np.ndarray, seed: int) -> np.ndarray:  # pragma: no cover - optional
    from .._optional import require_extra

    umap = require_extra("umap", "embeddings")
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(stacked)
