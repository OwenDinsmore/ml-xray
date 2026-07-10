"""Cross-space alignment for embedding diff (Phase 3 -- scaffolded).

Embedding spaces of different dimensionality/orientation are aligned with
orthogonal Procrustes on shared ids before neighborhood comparison. When
alignment is not meaningful, comparison can fall back to dimension-free
neighborhood-set overlap.
"""

from __future__ import annotations

import numpy as np

__all__ = ["orthogonal_procrustes"]


def orthogonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_reflection: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ``source`` onto ``target`` with an orthogonal transform.

    Parameters
    ----------
    source, target : numpy.ndarray
        Row-aligned matrices of shape ``(n, d)``. Padded to a common dimension
        when they differ.
    allow_reflection : bool
        Whether the optimal transform may include a reflection.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The aligned source and the orthogonal transform applied.

    Raises
    ------
    NotImplementedError
        Always -- Phase 3.
    """
    raise NotImplementedError("TODO(phase 3): orthogonal Procrustes alignment.")
