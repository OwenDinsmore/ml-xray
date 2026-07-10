"""Core abstractions for dataset linting: :class:`Severity`, :class:`Finding`,
the :class:`Check` ABC, the :class:`LintContext` passed to checks, and a simple
name-based check registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import pandas as pd

__all__ = [
    "Severity",
    "Finding",
    "LintContext",
    "Check",
    "register_check",
    "get_registry",
    "default_checks",
]


class Severity(IntEnum):
    """Ordered severity levels for a :class:`Finding`.

    The integer ordering (``INFO < WARN < ERROR``) makes it easy to threshold,
    e.g. ``[f for f in findings if f.severity >= Severity.WARN]``.
    """

    INFO = 10
    WARN = 20
    ERROR = 30

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    @classmethod
    def from_name(cls, name: str) -> Severity:
        """Return the severity matching ``name`` (case-insensitive).

        Parameters
        ----------
        name : str
            One of ``"info"``, ``"warn"``, ``"error"`` (any case).

        Returns
        -------
        Severity
            The matching enum member.

        Raises
        ------
        ValueError
            If ``name`` is not a known severity.
        """
        try:
            return cls[name.strip().upper()]
        except KeyError as exc:
            valid = ", ".join(s.name.lower() for s in cls)
            raise ValueError(f"unknown severity {name!r}; expected one of: {valid}") from exc


@dataclass
class Finding:
    """A single problem surfaced by a :class:`Check`.

    Parameters
    ----------
    check : str
        Name of the check that produced the finding.
    severity : Severity
        How serious the finding is.
    message : str
        Human-readable, one-line description.
    column : str, optional
        The column the finding concerns, when applicable.
    detail : dict
        Structured numbers backing the finding (statistics, thresholds, counts).
    rows : list of int, optional
        Zero-based positional row indices the finding points at, when
        applicable (e.g. duplicated or mislabeled rows).
    """

    check: str
    severity: Severity
    message: str
    column: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    rows: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of the finding."""
        return {
            "check": self.check,
            "severity": self.severity.name,
            "message": self.message,
            "column": self.column,
            "detail": self.detail,
            "rows": self.rows,
        }


@dataclass
class LintContext:
    """Inputs shared with every check during a :class:`~ml_xray.lint.Linter` run.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset being linted. Feature columns plus the target column.
    target : str, optional
        Name of the target/label column, if any.
    split : pandas.Series, optional
        Per-row split labels (e.g. ``"train"``/``"test"``), aligned to ``df``'s
        index, used by drift and cross-split leakage checks.
    task : {"classification", "regression"}, optional
        The learning task. Inferred from the target when ``None``.
    seed : int
        Seed for any stochastic check, so runs are reproducible.
    """

    df: pd.DataFrame
    target: str | None = None
    split: pd.Series | None = None
    task: str | None = None
    seed: int = 0

    @property
    def feature_columns(self) -> list[str]:
        """Column names excluding the target (all columns if no target)."""
        cols = list(self.df.columns)
        if self.target is not None and self.target in cols:
            cols.remove(self.target)
        return cols

    @property
    def target_series(self) -> pd.Series | None:
        """The target column as a Series, or ``None`` when no target is set."""
        if self.target is not None and self.target in self.df.columns:
            return self.df[self.target]
        return None


class Check(ABC):
    """Base class for a dataset check.

    Subclasses set a class-level :attr:`name` and implement :meth:`run`. Checks
    are pure functions of the :class:`LintContext`: they must not mutate the
    input DataFrame and must be deterministic given ``ctx.seed``.
    """

    #: Short, stable identifier used in reports and the registry.
    name: str = "check"
    #: One-line description shown in report headers.
    description: str = ""

    @abstractmethod
    def run(self, ctx: LintContext) -> list[Finding]:
        """Analyze ``ctx`` and return zero or more :class:`Finding` objects.

        Parameters
        ----------
        ctx : LintContext
            The dataset and metadata to inspect.

        Returns
        -------
        list of Finding
            Findings produced by this check (empty if nothing is wrong).
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"


_REGISTRY: dict[str, type[Check]] = {}


def register_check(cls: type[Check]) -> type[Check]:
    """Class decorator registering a :class:`Check` subclass by its ``name``.

    Parameters
    ----------
    cls : type[Check]
        The check class to register.

    Returns
    -------
    type[Check]
        ``cls`` unchanged, so it can be used as a decorator.
    """
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a non-empty class attribute 'name'")
    _REGISTRY[cls.name] = cls
    return cls


def get_registry() -> dict[str, type[Check]]:
    """Return a copy of the name -> check-class registry."""
    return dict(_REGISTRY)


def default_checks() -> list[Check]:
    """Instantiate one of every registered check, in a stable order.

    Returns
    -------
    list of Check
        Fresh instances of all registered checks, ordered by name.
    """
    return [cls() for _, cls in sorted(_REGISTRY.items())]
