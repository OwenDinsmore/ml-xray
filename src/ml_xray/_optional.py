"""Helpers for lazily importing optional dependencies.

``ml-xray`` keeps heavy or narrow dependencies (``umap-learn``, ``riskplot``,
``plotly``, ``cleanlab``) behind packaging extras. Modules that want them import
lazily through :func:`require_extra` / :func:`optional_import` so that the core
package installs and imports with only ``numpy``, ``pandas``, ``scikit-learn``,
``scipy`` and ``jinja2``.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["require_extra", "optional_import", "have_module"]

# Maps an importable module name to the packaging extra that provides it.
_EXTRA_FOR_MODULE: dict[str, str] = {
    "umap": "embeddings",
    "riskplot": "viz",
    "plotly": "viz",
    "cleanlab": "noise",
}


def have_module(module: str) -> bool:
    """Return ``True`` if ``module`` can be imported, without raising.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"cleanlab"``.

    Returns
    -------
    bool
        Whether the module is importable in the current environment.
    """
    try:
        importlib.import_module(module)
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


def require_extra(module: str, extra: str | None = None) -> ModuleType:
    """Import ``module`` or raise a helpful error naming the packaging extra.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"umap"``.
    extra : str, optional
        The ``ml-xray`` extra that provides the module (e.g. ``"embeddings"``).
        When ``None`` it is looked up from a built-in table.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed, with an actionable message pointing at
        ``pip install "ml-xray[<extra>]"``.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        extra = extra or _EXTRA_FOR_MODULE.get(module, "all")
        raise ImportError(
            f"ml-xray needs the optional dependency '{module}' for this feature. "
            f'Install it with: pip install "ml-xray[{extra}]"'
        ) from exc


def optional_import(module: str) -> ModuleType | None:
    """Import ``module`` and return it, or ``None`` if it is unavailable.

    Parameters
    ----------
    module : str
        Importable module name.

    Returns
    -------
    module or None
        The imported module, or ``None`` when it cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except Exception:  # pragma: no cover - environment dependent
        return None
