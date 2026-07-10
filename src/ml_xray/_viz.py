"""Viz backend selection.

``ml-xray`` computes and returns result objects; rendering is optional. This
module picks a charting backend at call time, preferring :mod:`riskplot`, then
:mod:`plotly`, then :mod:`matplotlib`. Nothing here is imported eagerly, so the
core package never hard-depends on any plotting library.
"""

from __future__ import annotations

from typing import Literal

from ._optional import have_module

__all__ = ["select_backend", "Backend"]

Backend = Literal["riskplot", "plotly", "matplotlib"]

# Preference order per the spec: riskplot -> plotly -> matplotlib.
_PREFERENCE: tuple[Backend, ...] = ("riskplot", "plotly", "matplotlib")


def select_backend(prefer: Backend | None = None) -> Backend | None:
    """Return the best available viz backend, or ``None`` if none is installed.

    Parameters
    ----------
    prefer : {"riskplot", "plotly", "matplotlib"}, optional
        A backend to try first before falling back to the default preference
        order.

    Returns
    -------
    {"riskplot", "plotly", "matplotlib"} or None
        The name of the first importable backend, or ``None`` when no supported
        plotting library is available.
    """
    order: list[Backend] = list(_PREFERENCE)
    if prefer is not None and prefer in order:
        order.remove(prefer)
        order.insert(0, prefer)

    for backend in order:
        if have_module(backend):
            return backend
    return None
