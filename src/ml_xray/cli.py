"""Command-line interface: ``ml-xray lint|slices|embed-diff|report``.

All subcommands are implemented: ``lint`` (with config, baseline diffing, and CI
gates), ``slices`` (with interactive charts), ``embed-diff`` (with an approximate
backend and interactive projection), and the unified ``report``.
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
    p_lint.add_argument("--time", default=None, help="Timestamp column for the temporal check.")
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
    p_lint.add_argument(
        "--fail-on-new",
        default=None,
        choices=["info", "warn", "error"],
        help="With --baseline, exit non-zero only on NEW findings at/above this severity.",
    )
    p_lint.add_argument("--config", default=None, help="Path to ml-xray.toml / pyproject.toml.")
    p_lint.add_argument("--baseline", default=None, help="Baseline report JSON to diff against.")
    p_lint.add_argument("--save-baseline", default=None, help="Write this run's report JSON here.")
    p_lint.add_argument("--html", default=None, help="Write an HTML report to this path.")
    p_lint.add_argument("--json", default=None, help="Write the report dict as JSON to this path.")
    p_lint.add_argument("--seed", type=int, default=None, help="Seed for stochastic checks.")
    p_lint.set_defaults(func=_cmd_lint)

    # --- slices -----------------------------------------------------------
    p_slices = sub.add_parser("slices", help="Slice discovery / error analysis.")
    p_slices.add_argument("preds", help="CSV with columns y_true,y_pred[,y_proba] plus features.")
    p_slices.add_argument("--features", default=None, help="Comma-separated feature columns.")
    p_slices.add_argument("--metric", default="auto", help="f1/accuracy/mse/mae/auto.")
    p_slices.add_argument("--max-depth", type=int, default=2, help="Max features per slice.")
    p_slices.add_argument("--min-support", type=int, default=30, help="Min rows per slice.")
    p_slices.add_argument("--top-k", type=int, default=20, help="Number of slices to report.")
    p_slices.add_argument("--html", default=None, help="Write an HTML report to this path.")
    p_slices.add_argument(
        "--interactive", action="store_true", help="Embed an interactive plotly chart in the HTML."
    )
    p_slices.set_defaults(func=_cmd_slices)

    # --- embed-diff -------------------------------------------------------
    p_embed = sub.add_parser("embed-diff", help="Compare two embedding spaces.")
    p_embed.add_argument("a", help="Path to embedding matrix A (.npy).")
    p_embed.add_argument("b", help="Path to embedding matrix B (.npy).")
    p_embed.add_argument("--ids", default=None, help="CSV/TXT of row ids aligning A and B.")
    p_embed.add_argument("-k", type=int, default=10, help="Neighborhood size for k-NN overlap.")
    p_embed.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "exact", "approx"],
        help="k-NN backend (approx needs pynndescent).",
    )
    p_embed.add_argument("--html", default=None, help="Write an HTML report to this path.")
    p_embed.add_argument(
        "--interactive",
        action="store_true",
        help="Embed an interactive plotly scatter in the HTML.",
    )
    p_embed.set_defaults(func=_cmd_embed_diff)

    # --- report (unified) -------------------------------------------------
    p_report = sub.add_parser("report", help="Combine lint/slices/embed into one report.")
    p_report.add_argument("--lint", default=None, help="Dataset CSV/Parquet for the lint section.")
    p_report.add_argument("--target", default=None, help="Target column for the lint section.")
    p_report.add_argument("--split", default=None, help="Split column for the lint section.")
    p_report.add_argument("--slices", default=None, help="Predictions CSV for the slices section.")
    p_report.add_argument("--features", default=None, help="Comma-separated feature columns.")
    p_report.add_argument(
        "--embed",
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="Two .npy embedding matrices for the embed section.",
    )
    p_report.add_argument("--html", required=True, help="Write the combined HTML report here.")
    p_report.set_defaults(func=_cmd_report)

    return parser


def _read_table(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _cmd_lint(args: argparse.Namespace) -> int:
    from .lint import Linter
    from .lint.base import Severity

    config = None
    if args.config is not None:
        from .config import LintConfig

        config = LintConfig.from_toml(args.config)

    df = _read_table(args.data)
    split = None
    if args.split is not None:
        if args.split not in df.columns:
            print(f"error: split column {args.split!r} not found", file=sys.stderr)
            return 2
        split = df[args.split]
        df = df.drop(columns=[args.split])

    if args.time is not None and args.time not in df.columns:
        print(f"error: time column {args.time!r} not found", file=sys.stderr)
        return 2

    report = Linter(target=args.target, config=config).run(
        df, split=split, task=args.task, seed=args.seed, time=args.time
    )

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
    if args.save_baseline:
        report.to_json(args.save_baseline)
        print(f"wrote baseline snapshot to {args.save_baseline}")

    # Baseline diff: report new/resolved and (optionally) gate on new findings.
    diff = None
    if args.baseline:
        from .baseline import diff_reports
        from .lint.linter import LintReport

        baseline = LintReport.from_json(args.baseline)
        diff = diff_reports(baseline, report)
        c = diff.counts()
        print(
            f"vs baseline: {c['new']} new, {c['resolved']} resolved, "
            f"{c['escalated']} escalated, {c['persisting']} persisting"
        )
        for f in diff.new:
            col = f" [{f.column}]" if f.column else ""
            print(f"  NEW   {f.severity.name:5s} {f.check}{col}: {f.message}")

    if args.fail_on_new is not None:
        if diff is None:
            print("error: --fail-on-new requires --baseline", file=sys.stderr)
            return 2
        threshold = Severity.from_name(args.fail_on_new)
        offenders = diff.new_at_or_above(threshold)
        if offenders:
            print(
                f"gate failed: {len(offenders)} NEW finding(s) at or above "
                f"{args.fail_on_new.upper()}",
                file=sys.stderr,
            )
            return 1

    if args.fail_on is not None:
        threshold = Severity.from_name(args.fail_on)
        if any(f.severity >= threshold for f in report):
            print(f"gate failed: findings at or above {args.fail_on.upper()}", file=sys.stderr)
            return 1
    return 0


def _load_preds(path: str, features: str | None):
    """Load a predictions CSV into (features_df, y_true, y_pred, y_proba)."""
    df = _read_table(path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("predictions file must contain 'y_true' and 'y_pred' columns")
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_proba = df["y_proba"].to_numpy() if "y_proba" in df.columns else None
    reserved = {"y_true", "y_pred", "y_proba"}
    if features:
        cols = [c.strip() for c in features.split(",") if c.strip()]
    else:
        cols = [c for c in df.columns if c not in reserved]
    return df[cols], y_true, y_pred, y_proba


def _cmd_slices(args: argparse.Namespace) -> int:
    from .slices import SliceFinder

    try:
        X, y_true, y_pred, y_proba = _load_preds(args.preds, args.features)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    finder = SliceFinder(
        metric=args.metric,
        max_depth=args.max_depth,
        min_support=args.min_support,
        top_k=args.top_k,
    ).fit(X, y_true, y_pred, y_proba)
    report = finder.report()

    print(
        f"ml-xray slices: {len(report)} underperforming slices "
        f"(metric={report.metric}, baseline={report.baseline:.4g})"
    )
    for s in report.slices[:10]:
        print(
            f"  {s.describe()}  n={s.support}  {report.metric}={s.metric_value:.3g} "
            f"(delta {s.delta:+.3g}, p={s.p_value:.3g})"
        )
    if args.html:
        from .report import slice_report_html

        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(slice_report_html(report, interactive=args.interactive))
        print(f"wrote HTML report to {args.html}")
    return 0


def _load_ids(path: str | None, n: int):
    if path is None:
        return None
    import pandas as pd

    if path.endswith((".csv", ".parquet")):
        frame = _read_table(path)
        return frame.iloc[:, 0].tolist()
    return pd.read_csv(path, header=None).iloc[:, 0].tolist()


def _cmd_embed_diff(args: argparse.Namespace) -> int:
    import numpy as np

    from .embed import EmbeddingDiff

    a = np.load(args.a)
    b = np.load(args.b)
    ids = _load_ids(args.ids, a.shape[0])

    report = EmbeddingDiff(k=args.k, backend=args.backend).fit(a, b, ids=ids).report()
    print(
        f"ml-xray embed-diff: neighbor_overlap={report.neighbor_overlap:.3f} "
        f"cluster_stability(ARI)={report.cluster_stability:.3f} "
        f"mean_drift={float(report.per_point_drift.mean()):.3f} backend={report.backend}"
    )
    print("  top movers: " + ", ".join(str(m) for m in report.movers[:10]))
    if args.html:
        from .report import embed_report_html

        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(embed_report_html(report, interactive=args.interactive))
        print(f"wrote HTML report to {args.html}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import embed_section, lint_section, render_report, slice_section

    sections: list[str] = []

    if args.lint:
        from .lint import Linter

        df = _read_table(args.lint)
        split = None
        if args.split and args.split in df.columns:
            split = df[args.split]
            df = df.drop(columns=[args.split])
        report = Linter(target=args.target).run(df, split=split)
        sections.append(lint_section(report))

    if args.slices:
        from .slices import SliceFinder

        X, y_true, y_pred, y_proba = _load_preds(args.slices, args.features)
        sreport = SliceFinder().fit(X, y_true, y_pred, y_proba).report()
        sections.append(slice_section(sreport))

    if args.embed:
        import numpy as np

        from .embed import EmbeddingDiff
        from .report import _projection_data_uri

        a, b = np.load(args.embed[0]), np.load(args.embed[1])
        ereport = EmbeddingDiff().fit(a, b).report()
        sections.append(embed_section(ereport, image=_projection_data_uri(ereport)))

    if not sections:
        print("error: provide at least one of --lint, --slices, --embed", file=sys.stderr)
        return 2

    html = render_report(*sections, title="ml-xray — combined report")
    with open(args.html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote combined HTML report to {args.html}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ml-xray`` console script.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (``0`` success, ``1`` lint gate failed, ``2`` usage
        error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
