"""The :class:`Linter` orchestrator and its :class:`LintReport` result object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ._util import infer_task
from .base import Check, Finding, LintContext, Severity, default_checks

if TYPE_CHECKING:  # pragma: no cover
    from ..config import LintConfig

__all__ = ["Linter", "LintReport"]


class Linter:
    """Run a suite of dataset checks and collect their findings.

    Parameters
    ----------
    checks : list of Check, optional
        Checks to run. When ``None`` (default), every registered check is used.
    target : str, optional
        Name of the target/label column. Enables leakage, label-noise, and
        imbalance checks that need a target.
    config : LintConfig, optional
        Configuration controlling check selection and severity overrides. When
        given and ``checks`` is ``None``, the checks are built from the config.

    Examples
    --------
    >>> linter = Linter(target="y")
    >>> report = linter.run(df, split=df["split"])
    >>> bool(report)  # False if any ERROR finding exists
    False
    """

    def __init__(
        self,
        checks: list[Check] | None = None,
        *,
        target: str | None = None,
        config: LintConfig | None = None,
    ) -> None:
        if checks is not None:
            self.checks = list(checks)
        elif config is not None:
            self.checks = config.build_checks()
        else:
            self.checks = default_checks()
        self.target = target
        self.config = config

    @classmethod
    def from_config(cls, config: LintConfig, *, target: str | None = None) -> Linter:
        """Build a linter from a :class:`~ml_xray.config.LintConfig`."""
        return cls(target=target, config=config)

    def run(
        self,
        df: pd.DataFrame,
        *,
        split: pd.Series | None = None,
        task: str | None = None,
        seed: int | None = None,
        time: str | None = None,
    ) -> LintReport:
        """Run all configured checks over ``df`` and return a report.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataset to lint (features plus the optional target column).
        split : pandas.Series, optional
            Per-row train/test/holdout labels for drift and cross-split leakage.
            Must be alignable to ``df`` (same length); its index is ignored.
        task : {"classification", "regression"}, optional
            The learning task. Inferred from the target when omitted.
        seed : int, optional
            Seed for stochastic checks, so reports are reproducible. Defaults to
            the config's seed (or ``0``) when not given.
        time : str, optional
            Name of a timestamp/ordering column enabling the temporal-leakage
            check.

        Returns
        -------
        LintReport
            The collected findings.

        Raises
        ------
        ValueError
            If ``target``/``time`` was set but is not a column of ``df``, or if
            ``split`` length does not match ``df``.
        """
        if self.target is not None and self.target not in df.columns:
            raise ValueError(f"target column {self.target!r} is not in the DataFrame")
        if time is not None and time not in df.columns:
            raise ValueError(f"time column {time!r} is not in the DataFrame")

        if seed is None:
            seed = self.config.seed if self.config is not None else 0

        split_series = self._prepare_split(df, split)
        resolved_task = task
        if resolved_task is None and self.target is not None:
            resolved_task = infer_task(df[self.target])

        ctx = LintContext(
            df=df,
            target=self.target,
            split=split_series,
            task=resolved_task,
            seed=seed,
            time=time,
        )

        findings: list[Finding] = []
        for check in self.checks:
            findings.extend(check.run(ctx))

        if self.config is not None:
            findings = self.config.apply_severity(findings)

        meta = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "target": self.target,
            "task": resolved_task,
            "has_split": split_series is not None,
            "time": time,
            "checks": [c.name for c in self.checks],
        }
        return LintReport(findings=findings, meta=meta)

    @staticmethod
    def _prepare_split(df: pd.DataFrame, split: pd.Series | None) -> pd.Series | None:
        if split is None:
            return None
        if len(split) != len(df):
            raise ValueError(f"split length {len(split)} does not match DataFrame length {len(df)}")
        # Realign positionally so checks can reset_index safely.
        return pd.Series(split.to_numpy(), index=df.index, name=getattr(split, "name", "split"))


class LintReport:
    """The structured result of a :meth:`Linter.run`.

    Parameters
    ----------
    findings : list of Finding
        All findings produced by the checks.
    meta : dict, optional
        Run metadata (row/column counts, target, task, checks run).
    """

    def __init__(self, findings: list[Finding], meta: dict[str, Any] | None = None) -> None:
        self.findings: list[Finding] = list(findings)
        self.meta: dict[str, Any] = meta or {}

    def worst(self, n: int = 10) -> list[Finding]:
        """Return the ``n`` most severe findings.

        Findings are sorted by severity (``ERROR`` first); within a severity,
        those pointing at more rows come first.

        Parameters
        ----------
        n : int
            Maximum number of findings to return.

        Returns
        -------
        list of Finding
            Up to ``n`` findings, most severe first.
        """
        ordered = sorted(
            self.findings,
            key=lambda f: (int(f.severity), len(f.rows or [])),
            reverse=True,
        )
        return ordered[:n]

    def by_severity(self, severity: Severity) -> list[Finding]:
        """Return findings with exactly the given severity."""
        return [f for f in self.findings if f.severity == severity]

    def counts(self) -> dict[str, int]:
        """Return a ``{severity_name: count}`` summary."""
        out = {s.name: 0 for s in Severity}
        for f in self.findings:
            out[f.severity.name] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the whole report."""
        return {
            "meta": self.meta,
            "counts": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LintReport:
        """Reconstruct a report from its :meth:`to_dict` form."""
        findings = [Finding.from_dict(f) for f in data.get("findings", [])]
        return cls(findings=findings, meta=dict(data.get("meta") or {}))

    def to_json(self, path: str) -> None:
        """Write the report to a JSON file (a machine-readable snapshot).

        Parameters
        ----------
        path : str
            Destination file path. Overwritten if it exists.
        """
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> LintReport:
        """Load a report previously saved with :meth:`to_json`."""
        import json

        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_html(self, path: str) -> None:
        """Render the report to a single self-contained HTML file.

        Parameters
        ----------
        path : str
            Destination file path. Overwritten if it exists.
        """
        from ..report import lint_report_html

        html = lint_report_html(self)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

    def __bool__(self) -> bool:
        """``False`` when any ``ERROR`` finding exists (useful for CI gates)."""
        return not any(f.severity >= Severity.ERROR for f in self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        c = self.counts()
        return (
            f"LintReport(findings={len(self.findings)}, "
            f"ERROR={c['ERROR']}, WARN={c['WARN']}, INFO={c['INFO']})"
        )
