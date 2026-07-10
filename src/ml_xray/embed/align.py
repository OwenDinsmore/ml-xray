"""Cross-space alignment for embedding diff.

Embedding spaces of different dimensionality/orientation are aligned with
orthogonal Procrustes on shared ids before neighborhood comparison. When
alignment is not meaningful, comparison can fall back to dimension-free
neighborhood-set overlap.
"""

from __future__ import annotations

import numpy as np

__all__ = ["orthogonal_procrustes", "pad_to_common_dim"]


def pad_to_common_dim(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the narrower of two matrices so both share a column count.

    Parameters
    ----------
    a, b : numpy.ndarray
        Matrices of shape ``(n, d_a)`` and ``(n, d_b)``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``a`` and ``b`` padded to ``max(d_a, d_b)`` columns.
    """
    d = max(a.shape[1], b.shape[1])
    if a.shape[1] < d:
        a = np.hstack([a, np.zeros((a.shape[0], d - a.shape[1]), dtype=a.dtype)])
    if b.shape[1] < d:
        b = np.hstack([b, np.zeros((b.shape[0], d - b.shape[1]), dtype=b.dtype)])
    return a, b


def orthogonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_reflection: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ``source`` onto ``target`` with an optimal orthogonal transform.

    Solves ``min_R ||source @ R - target||_F`` over orthogonal ``R`` (the
    orthogonal Procrustes problem) via the SVD of ``source.T @ target``. When the
    spaces differ in dimensionality they are zero-padded to a common width first.

    Parameters
    ----------
    source, target : numpy.ndarray
        Row-aligned matrices of shape ``(n, d_a)`` and ``(n, d_b)``.
    allow_reflection : bool
        If ``False``, the transform is constrained to a proper rotation
        (``det(R) = +1``); otherwise reflections are permitted.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The aligned source (``source @ R``, padded) and the transform ``R``.
    """
    a, b = pad_to_common_dim(np.asarray(source, float), np.asarray(target, float))
    m = a.T @ b
    u, _s, vt = np.linalg.svd(m)
    r = u @ vt
    if not allow_reflection and np.linalg.det(r) < 0:
        u = u.copy()
        u[:, -1] *= -1
        r = u @ vt
    aligned = a @ r
    return aligned, r
