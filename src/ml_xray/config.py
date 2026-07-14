"""Project configuration for ml-xray.

Loads settings from an ``ml-xray.toml`` file or a ``[tool.ml-xray]`` table in
``pyproject.toml``, so a repository can pin which checks run, remap finding
severities, and pass per-check arguments without touching code.

Example ``ml-xray.toml``::

    # only run these checks (omit to run all registered checks)
    checks = ["leakage", "drift", "duplicates"]
    # never run these (applied after `checks`)
    disable = ["outliers"]
    seed = 7

    [severity]
    # downgrade every duplicates finding to INFO, silence rare-level notes
    duplicates = "info"
    "imbalance.rare_levels" = "ignore"
    # escalate a specific drift kind
    "drift.numeric" = "error"

    [check_args.outliers]
    ranges = { age = [0, 120] }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lint.base import Check, Finding, Severity, get_registry

__all__ = ["LintConfig", "load_toml"]


def load_toml(path: str | Path) -> dict[str, Any]:
    """Parse a TOML file into a dict, using ``tomllib`` (3.11+) or ``tomli``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a TOML file.

    Returns
    -------
    dict
        The parsed contents.

    Raises
    ------
    ImportError
        On Python < 3.11 without the ``tomli`` backport installed.
    """
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
        try:
            import tomli as toml_reader  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "Reading TOML config on Python < 3.11 needs 'tomli'. "
                'Install it with: pip install "ml-xray[dev]"'
            ) from exc
    with open(path, "rb") as fh:
        return toml_reader.load(fh)


@dataclass
class LintConfig:
    """Resolved lint configuration.

    Parameters
    ----------
    checks : list of str, optional
        Names of checks to run. ``None`` means every registered check.
    disable : list of str
        Check names to exclude (applied after ``checks``).
    severity : dict
        Maps a check name (``"drift"``) or ``"check.kind"`` (``"drift.numeric"``)
        to a target severity (``"info"``/``"warn"``/``"error"``) or ``"ignore"``
        to drop matching findings.
    check_args : dict
        Maps a check name to keyword arguments passed to its constructor
        (e.g. ``{"outliers": {"ranges": {"age": (0, 120)}}}``).
    seed : int
        Default seed for stochastic checks.
    fail_on : str, optional
        Default severity gate for the CLI (``"info"``/``"warn"``/``"error"``).
    """

    checks: list[str] | None = None
    disable: list[str] = field(default_factory=list)
    severity: dict[str, str] = field(default_factory=dict)
    check_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    seed: int = 0
    fail_on: str | None = None

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LintConfig:
        """Build a config from an already-parsed mapping."""
        return cls(
            checks=data.get("checks"),
            disable=list(data.get("disable", [])),
            severity={str(k): str(v).lower() for k, v in (data.get("severity") or {}).items()},
            check_args={str(k): dict(v) for k, v in (data.get("check_args") or {}).items()},
            seed=int(data.get("seed", 0)),
            fail_on=(data.get("fail_on") or None),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> LintConfig:
        """Load config from a TOML file.

        A ``pyproject.toml`` is read from its ``[tool.ml-xray]`` table; any other
        file is read from its top level.
        """
        path = Path(path)
        data = load_toml(path)
        if path.name == "pyproject.toml":
            data = data.get("tool", {}).get("ml-xray", {})
        return cls.from_dict(data)

    @classmethod
    def discover(cls, start: str | Path = ".") -> LintConfig:
        """Find and load config near ``start`` (searching upward).

        Looks for ``ml-xray.toml`` first, then a ``pyproject.toml`` containing a
        ``[tool.ml-xray]`` table, walking up parent directories. Returns a default
        config if none is found.
        """
        start = Path(start).resolve()
        directories = [start, *start.parents] if start.is_dir() else [start.parent, *start.parents]
        for directory in directories:
            explicit = directory / "ml-xray.toml"
            if explicit.is_file():
                return cls.from_toml(explicit)
            pyproject = directory / "pyproject.toml"
            if pyproject.is_file():
                data = load_toml(pyproject).get("tool", {}).get("ml-xray")
                if data is not None:
                    return cls.from_dict(data)
        return cls()

    # -- application ------------------------------------------------------

    def build_checks(self) -> list[Check]:
        """Instantiate the configured checks from the registry.

        Returns
        -------
        list of Check
            Selected, enabled check instances (with ``check_args`` applied),
            ordered by name.

        Raises
        ------
        ValueError
            If a named check is not registered.
        """
        registry = get_registry()
        names = self.checks if self.checks is not None else sorted(registry)
        disabled = set(self.disable)
        instances: list[Check] = []
        for name in names:
            if name in disabled:
                continue
            if name not in registry:
                valid = ", ".join(sorted(registry))
                raise ValueError(f"unknown check {name!r}; registered checks: {valid}")
            kwargs = self.check_args.get(name, {})
            instances.append(registry[name](**kwargs))
        return sorted(instances, key=lambda c: c.name)

    def apply_severity(self, findings: list[Finding]) -> list[Finding]:
        """Remap or drop findings per the ``severity`` table.

        A more specific ``"check.kind"`` rule wins over a bare ``"check"`` rule.
        Findings mapped to ``"ignore"`` are removed.

        Parameters
        ----------
        findings : list of Finding
            Findings to post-process.

        Returns
        -------
        list of Finding
            The remapped, filtered findings.
        """
        if not self.severity:
            return findings
        out: list[Finding] = []
        for f in findings:
            kind = f.detail.get("kind")
            rule = None
            if kind is not None and f"{f.check}.{kind}" in self.severity:
                rule = self.severity[f"{f.check}.{kind}"]
            elif f.check in self.severity:
                rule = self.severity[f.check]
            if rule is None:
                out.append(f)
                continue
            if rule == "ignore":
                continue
            f.severity = Severity.from_name(rule)
            out.append(f)
        return out
