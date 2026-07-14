# ml-xray

**Framework-agnostic ML model & dataset analysis across the training lifecycle.**

`ml-xray` inspects the things that actually move a model's quality —
**leakage, drift, label noise, class imbalance, outliers** before training,
and **where a trained model fails** and **how its embeddings shifted** after
training. It takes arrays and DataFrames, never model objects, so it drops into
any stack (scikit-learn, XGBoost, LightGBM, PyTorch, or just a CSV of
predictions) with zero coupling.

**[→ Project site & showcase](https://owendinsmore.github.io/ml-xray/)** · [GitHub](https://github.com/OwenDinsmore/ml-xray)

```bash
pip install ml-xray
```

```bash
# Gate a training pipeline on data quality (non-zero exit on ERROR findings)
ml-xray lint data.csv --target y --split split --fail-on error --html report.html
```

```python
import pandas as pd
from ml_xray.lint import Linter

df = pd.read_csv("data.csv")
report = Linter(target="y").run(df, split=df["split"])

print(report.worst(5))       # the 5 most severe findings
report.to_html("report.html")

if not report:               # False if any ERROR finding exists
    raise SystemExit("data quality gate failed")
```

## Scope — what ml-xray is *not*

> - **Not a training framework.** It never trains your model for you; it
>   analyzes data and predictions you already have.
> - **Not experiment tracking.** MLflow / Weights & Biases / Aim own that.
> - **Not schema matching.** Valentine owns that.
> - **Not a generic dataset profiler.** ydata-profiling owns generic profiling.
>   `ml-xray` is opinionated toward decisions that affect a *model* — leakage,
>   drift, label noise, and where a trained model fails — not exhaustive column
>   statistics.

Inputs are always arrays / DataFrames (`y_true`, `y_pred`, `y_proba`, a feature
`DataFrame`, or plain embedding matrices). `ml-xray` never imports or requires a
model object.

## Capabilities

| Stage | Module | What it does |
| --- | --- | --- |
| Pre-training | `ml_xray.lint` | Dataset QA: leakage, drift, label noise, duplicates, imbalance, outliers. |
| Post-training | `ml_xray.slices` | Error analysis / slice discovery — find where a model underperforms. |
| Post-training | `ml_xray.embed` | Embedding diff — compare two embedding spaces. |

All three modules are implemented, return structured result objects, and emit
self-contained HTML report sections that `ml-xray report` stitches into one file.

### `ml_xray.lint` — dataset QA (Phase 1)

Each check is a pluggable `Check` subclass and emits structured `Finding`s with a
severity (`INFO` / `WARN` / `ERROR`), the offending column, backing numbers, and
row indices where applicable.

- **leakage** — feature/target correlation ≈ 1, features that deterministically
  predict the target, and duplicate rows shared across a train/test split (the
  most common silent leak).
- **drift** — per-column train-vs-test distribution drift via PSI, the
  Kolmogorov–Smirnov test (numeric), and Jensen–Shannon divergence
  (categorical).
- **label_noise** — mislabel candidates from out-of-fold predictions
  (confidently-wrong / low-margin rows); delegates to
  [`cleanlab`](https://github.com/cleanlab/cleanlab) when the `[noise]` extra is
  installed.
- **duplicates** — exact duplicate rows and near-duplicate clusters via
  MinHash + LSH.
- **imbalance** — class imbalance ratio, rare categorical levels, single-value
  columns.
- **outliers** — robust-z / IQR numeric outliers, high-null-fraction columns,
  and constant columns.
- **temporal_leakage** — given a time column (`Linter.run(..., time="ts")` or
  `--time`), flags training rows dated after a later split (future leaking into
  train) and features that are near-monotonic proxies for time.

### `ml_xray.slices` — slice discovery (Phase 2)

Automatically find where a trained model underperforms — e.g. *"accuracy is 0.49
on `region=EU` (n=632) vs 0.84 overall"*. Continuous features are discretized,
slices up to `max_depth` conjunctions are enumerated with Apriori-style pruning,
each is scored and tested for significance with a Benjamini–Hochberg correction
across all slices tested, and the survivors are ranked by
`|underperformance| × log(support)`.

```python
from ml_xray.slices import SliceFinder

report = SliceFinder(metric="auto", max_depth=2, min_support=30).fit(
    X, y_true, y_pred, y_proba
).report()

for s in report.slices[:5]:
    print(s.describe(), s.support, s.metric_value, s.delta, s.p_value)
```

### `ml_xray.embed` — embedding diff (Phase 3)

Compare two embedding spaces — v1 vs v2, or embeddings over time — to see what
moved: k-NN Jaccard **neighbor overlap** (local structure), **per-point drift**
(which items moved), and **cluster stability** via Adjusted Rand Index (global
structure). Spaces of different dimensionality are aligned with orthogonal
Procrustes for the projection scatter; the overlap metrics are dimension-free.

```python
from ml_xray.embed import EmbeddingDiff

report = EmbeddingDiff(k=10).fit(emb_a, emb_b, ids=ids).report()
print(report.neighbor_overlap, report.cluster_stability)
print(report.movers[:10])         # ids whose neighborhoods changed most
```

## Using it in CI

### Configuration (`ml-xray.toml` or `[tool.ml-xray]`)

Pin which checks run and how severe their findings are, without touching code:

```toml
# ml-xray.toml
checks = ["leakage", "drift", "duplicates", "imbalance"]   # omit to run all
disable = ["outliers"]
seed = 7

[severity]
duplicates = "info"                    # downgrade every duplicates finding
"imbalance.rare_levels" = "ignore"     # silence a specific kind
"drift.numeric" = "error"              # escalate another

[check_args.outliers]
ranges = { age = [0, 120] }            # declared valid ranges
```

```bash
ml-xray lint data.csv --target y --config ml-xray.toml --fail-on error
```

### Baseline / regression tracking

Gate CI on *new* problems a change introduces, not on pre-existing debt:

```bash
# once: snapshot the current state
ml-xray lint data.csv --target y --save-baseline .ml-xray-baseline.json

# in CI: fail only if the change adds a new ERROR-level finding
ml-xray lint data.csv --target y \
    --baseline .ml-xray-baseline.json --fail-on-new error
```

```python
from ml_xray import Linter, diff_reports
from ml_xray.lint.linter import LintReport

baseline = LintReport.from_json(".ml-xray-baseline.json")
current = Linter(target="y").run(df)
diff = diff_reports(baseline, current)
print(diff.counts())                 # {'new': ..., 'resolved': ..., ...}
if not diff.is_clean():              # any NEW error-level finding?
    raise SystemExit("new data-quality regressions")
```

### pre-commit hook

```yaml
repos:
  - repo: https://github.com/OwenDinsmore/ml-xray
    rev: v0.2.0
    hooks:
      - id: ml-xray-lint
        args: [data/train.csv, --target, y, --fail-on, error]
```

## Design principles

- **Framework-agnostic** array / DataFrame contracts.
- **Deterministic & reproducible** — everything is seeded; the same input
  produces the same report.
- **Report-first** — every module returns a structured result object and can
  emit a self-contained HTML report section.
- **Statistically honest** — findings carry support size and a
  significance/effect estimate; slice discovery corrects for multiple
  comparisons.
- **Lazy optional deps** — `umap-learn`, `riskplot`/`plotly`, and `cleanlab`
  live behind extras and degrade gracefully when absent.

## Installation extras

```bash
pip install ml-xray                 # core: numpy, pandas, scikit-learn, jinja2
pip install "ml-xray[embeddings]"   # umap-learn projections for embed-diff
pip install "ml-xray[viz]"          # riskplot / plotly rich charts
pip install "ml-xray[noise]"        # cleanlab confident-learning label noise
pip install "ml-xray[all]"          # everything
```

## CLI

```bash
ml-xray lint DATA --target y [--split col] [--time col] [--config ml-xray.toml] \
    [--baseline base.json] [--save-baseline base.json] \
    [--fail-on error] [--fail-on-new error] [--html out.html] [--json out.json]
ml-xray slices PREDS [--features cols] [--metric auto|accuracy|f1|mse|mae|roc_auc|log_loss] \
    [--html out.html] [--interactive]
ml-xray embed-diff A.npy B.npy [--ids ids.csv] [-k 10] [--backend auto|exact|approx] \
    [--html out.html] [--interactive]
ml-xray report --lint DATA --target y --slices PREDS --embed A.npy B.npy --html out.html
```

`--interactive` embeds a self-contained plotly chart (needs `ml-xray[viz]`);
`--backend approx` uses pynndescent (`ml-xray[embeddings]`) for large embedding
sets. Both degrade gracefully when the optional dependency is absent.

`PREDS` is a CSV with `y_true,y_pred[,y_proba]` columns plus the feature columns
to slice on.

## Roadmap

- **Phase 1 — done:** `ml_xray.lint` + HTML report + `ml-xray lint` CLI.
- **Phase 2 — done:** `ml_xray.slices` + `ml-xray slices`.
- **Phase 3 — done:** `ml_xray.embed` + `ml-xray embed-diff`.
- **Phase 4 — done:** unified `ml-xray report` stitching all sections into one
  self-contained HTML file, with a matplotlib viz backend.
- **v0.2 — done:** TOML config (check selection + severity overrides), baseline
  snapshots and regression diffing, `--fail-on-new` gate, probability-aware
  slice metrics (ROC-AUC, log-loss), a `pre-commit` hook, a **temporal-leakage
  check**, **numeric-range slice predicates** (`tenure < 6`), an **approximate
  k-NN backend** (pynndescent) for large embedding sets, and **interactive
  plotly charts** in the slice/embed HTML reports.

## License

[MIT](LICENSE)
