"""Command-line interface: ``ml-xray lint|slices|embed-diff|report``.

Phase 1 implements ``lint`` end-to-end (including a ``--fail-on`` gate for CI).
The ``slices``, ``embed-diff`` and ``report`` subcommands are wired up but raise
a clear "not yet implemented" message until their phases land.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="ml-xray",
        description="Framework-agnostic ML model & dataset analysis.",
    )
    parser.add_argument("--version", action="version", version=f"ml-xray {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- lint -------------------------------------------------------------
    p_lint = sub.add_parser("lint", help="Lint a dataset for QA issues.")
    p_lint.add_argument("data", help="Path to a CSV/Parquet dataset.")
    p_lint.add_argument("--target", default=None, help="Target/label column name.")
    p_lint.add_argument("--split", default=None, help="Column holding train/test split labels.")
    p_lint.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression"],
        help="Learning task (inferred from the target when omitted).",
    )
    p_lint.add_argument(
        "--fail-on",
        default=None,
        choices=["info", "warn", "error"],
        help="Exit non-zero if any finding is at or above this severity.",
    )
    p_lint.add_argument("--html", default=None, help="Write an HTML report to this path.")
    p_lint.add_argument("--json", default=None, help="Write the report dict as JSON to this path.")
    p_lint.add_argument("--seed", type=int, default=0, help="Seed for stochastic checks.")
    p_lint.set_defaults(func=_cmd_lint)

    # --- slices (Phase 2) -------------------------------------------------
    p_slices = sub.add_parser("slices", help="Slice discovery / error analysis (Phase 2).")
    p_slices.add_argument("preds", help="CSV with columns y_true,y_pred[,y_proba].")
    p_slices.add_argument("--features", default=None, help="Comma-separated feature columns.")
    p_slices.add_argument("--html", default=None, help="Write an HTML report to this path.")
    p_slices.set_defaults(func=_cmd_slices)

    # --- embed-diff (Phase 3) --------------------------------------------
    p_embed = sub.add_parser("embed-diff", help="Compare two embedding spaces (Phase 3).")
    p_embed.add_argument("a", help="Path to embedding matrix A (.npy).")
    p_embed.add_argument("b", help="Path to embedding matrix B (.npy).")
    p_embed.add_argument("--ids", default=None, help="CSV of row ids aligning A and B.")
    p_embed.add_argument("--html", default=None, help="Write an HTML report to this path.")
    p_embed.set_defaults(func=_cmd_embed_diff)

    # --- report (Phase 4) -------------------------------------------------
    p_report = sub.add_parser("report", help="Combine sections into one report (Phase 4).")
    p_report.add_argument("--html", default=None, help="Write the combined HTML report here.")
    p_report.set_defaults(func=_cmd_report)

    return parser


def _read_table(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _cmd_lint(args: argparse.Namespace) -> int:
    from .lint import Linter

    df = _read_table(args.data)
    split = None
    if args.split is not None:
        if args.split not in df.columns:
            print(f"error: split column {args.split!r} not found", file=sys.stderr)
            return 2
        split = df[args.split]
        df = df.drop(columns=[args.split])

    report = Linter(target=args.target).run(df, split=split, task=args.task, seed=args.seed)

    counts = report.counts()
    print(
        f"ml-xray lint: {len(report)} findings "
        f"(ERROR={counts['ERROR']}, WARN={counts['WARN']}, INFO={counts['INFO']})"
    )
    for f in report.worst(10):
        col = f" [{f.column}]" if f.column else ""
        print(f"  {f.severity.name:5s} {f.check}{col}: {f.message}")

    if args.html:
        report.to_html(args.html)
        print(f"wrote HTML report to {args.html}")
    if args.json:
        import json

        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        print(f"wrote JSON report to {args.json}")

    if args.fail_on is not None:
        from .lint.base import Severity

        threshold = Severity.from_name(args.fail_on)
        if any(f.severity >= threshold for f in report):
            print(f"gate failed: findings at or above {args.fail_on.upper()}", file=sys.stderr)
            return 1
    return 0


def _cmd_slices(args: argparse.Namespace) -> int:
    print(
        "ml-xray slices is not implemented yet (Phase 2). " "See the roadmap in the README.",
        file=sys.stderr,
    )
    return 3


def _cmd_embed_diff(args: argparse.Namespace) -> int:
    print(
        "ml-xray embed-diff is not implemented yet (Phase 3). " "See the roadmap in the README.",
        file=sys.stderr,
    )
    return 3


def _cmd_report(args: argparse.Namespace) -> int:
    print(
        "ml-xray report is not implemented yet (Phase 4). " "See the roadmap in the README.",
        file=sys.stderr,
    )
    return 3


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ml-xray`` console script.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (``0`` success, ``1`` gate failed, ``2`` usage error,
        ``3`` unimplemented subcommand).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
