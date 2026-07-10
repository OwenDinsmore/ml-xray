# ml-xray documentation

`ml-xray` is a framework-agnostic ML model & dataset analysis engine spanning the
training lifecycle. Inputs are always arrays / DataFrames — never model objects.

- **Pre-training:** dataset QA / linting (`ml_xray.lint`) — *implemented*.
- **Post-training:** error analysis / slice discovery (`ml_xray.slices`) —
  *Phase 2, scaffolded*.
- **Post-training:** embedding diff (`ml_xray.embed`) — *Phase 3, scaffolded*.

See the [README](../README.md) for the scope box (what ml-xray is *not*),
installation extras, and the roadmap.

## Quickstart — dataset linting

```python
import pandas as pd
from ml_xray.lint import Linter

df = pd.read_csv("data.csv")
report = Linter(target="y").run(df, split=df.get("split"))

for finding in report.worst(10):
    print(finding.severity.name, finding.check, finding.column, "-", finding.message)

report.to_html("report.html")   # single self-contained HTML file

if not report:                  # False if any ERROR finding exists
    raise SystemExit("data quality gate failed")
```

## The lint checks

| Check | Severity signals | Notes |
| --- | --- | --- |
| `leakage` | ERROR on ~1.0 feature/target correlation, deterministic predictors, cross-split row overlap | The overlap case reports offending row indices. |
| `drift` | PSI / KS (numeric), PSI / Jensen–Shannon (categorical) | Needs a `split`; compares the reference split against every other. |
| `label_noise` | out-of-fold confidently-wrong rows; `cleanlab` when `[noise]` is installed | Classification targets only. |
| `duplicates` | exact duplicate rows (WARN), near-duplicate clusters (INFO) | Near-dups via a dependency-free MinHash + LSH. |
| `imbalance` | class imbalance ratio, rare categorical levels, single-value columns | |
| `outliers` | robust-z / IQR outliers, high-null columns, constant columns, out-of-range | Declared ranges via `OutliersCheck(ranges=...)`. |

## Extending with a custom check

```python
from ml_xray.lint import Check, Finding, Severity, register_check
from ml_xray.lint.base import LintContext


@register_check
class MyCheck(Check):
    name = "my_check"
    description = "Flags columns I care about."

    def run(self, ctx: LintContext) -> list[Finding]:
        findings = []
        for col in ctx.feature_columns:
            if ctx.df[col].isna().all():
                findings.append(
                    Finding(
                        check=self.name,
                        severity=Severity.ERROR,
                        message=f"Column '{col}' is entirely null.",
                        column=col,
                        detail={"null_fraction": 1.0},
                    )
                )
        return findings
```

Registered checks are picked up automatically by `Linter()` (which uses
`default_checks()` when no explicit list is passed).

## Reproducibility

Every stochastic check is seeded through `Linter.run(..., seed=...)`, so the same
input produces the same report.
