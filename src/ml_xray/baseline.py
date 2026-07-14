"""Baseline snapshots and report diffing for regression tracking.

Save a :class:`~ml_xray.lint.LintReport` to JSON, then compare a later report
against it to separate *new* findings (regressions introduced by a change) from
pre-existing debt and *resolved* findings. This lets CI fail only on new
problems via ``ml-xray lint ... --baseline base.json --fail-on-new error``,
instead of blocking on a backlog the change did not cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lint.base import Finding, Severity
from .lint.linter import LintReport

__all__ = ["ReportDiff", "diff_reports"]


@dataclass
class ReportDiff:
    """The difference between a baseline report and a current report.

    Findings are matched by :meth:`~ml_xray.lint.Finding.fingerprint` (check,
    column, kind), so counts changing between runs does not count as new.

    Attributes
    ----------
    new : list of Finding
        Findings present now but not in the baseline (regressions).
    resolved : list of Finding
        Findings in the baseline but no longer present (fixed).
    persisting : list of Finding
        Findings present in both (current instances).
    escalated : list of tuple(Finding, Finding)
        ``(baseline, current)`` pairs whose severity increased.
    """

    new: list[Finding] = field(default_factory=list)
    resolved: list[Finding] = field(default_factory=list)
    persisting: list[Finding] = field(default_factory=list)
    escalated: list[tuple[Finding, Finding]] = field(default_factory=list)

    def new_at_or_above(self, severity: Severity) -> list[Finding]:
        """Return new findings whose severity is at or above ``severity``."""
        return [f for f in self.new if f.severity >= severity]

    def is_clean(self, *, threshold: Severity = Severity.ERROR) -> bool:
        """``True`` when no new finding reaches ``threshold`` (a CI gate helper)."""
        return not self.new_at_or_above(threshold)

    def counts(self) -> dict[str, int]:
        """Return a ``{category: count}`` summary of the diff."""
        return {
            "new": len(self.new),
            "resolved": len(self.resolved),
            "persisting": len(self.persisting),
            "escalated": len(self.escalated),
        }

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the diff."""
        return {
            "counts": self.counts(),
            "new": [f.to_dict() for f in self.new],
            "resolved": [f.to_dict() for f in self.resolved],
            "escalated": [
                {"baseline": old.to_dict(), "current": cur.to_dict()} for old, cur in self.escalated
            ],
        }

    def __bool__(self) -> bool:
        """``True`` if there is any difference (new, resolved, or escalated)."""
        return bool(self.new or self.resolved or self.escalated)


def diff_reports(baseline: LintReport, current: LintReport) -> ReportDiff:
    """Compare a ``current`` report against a ``baseline`` snapshot.

    Parameters
    ----------
    baseline : LintReport
        The reference report (e.g. loaded from a committed snapshot).
    current : LintReport
        The freshly computed report.

    Returns
    -------
    ReportDiff
        The new / resolved / persisting / escalated breakdown.
    """
    base_by_fp = {f.fingerprint(): f for f in baseline.findings}
    cur_by_fp = {f.fingerprint(): f for f in current.findings}

    diff = ReportDiff()
    for fp, cur in cur_by_fp.items():
        if fp not in base_by_fp:
            diff.new.append(cur)
        else:
            diff.persisting.append(cur)
            if cur.severity > base_by_fp[fp].severity:
                diff.escalated.append((base_by_fp[fp], cur))
    for fp, old in base_by_fp.items():
        if fp not in cur_by_fp:
            diff.resolved.append(old)
    return diff
