# ml-xray

**Framework-agnostic ML model & dataset analysis across the training lifecycle.**

`ml-xray` inspects the things that actually move a model's quality —
**leakage, drift, label noise, class imbalance, outliers** before training,
and **where a trained model fails** and **how its embeddings shifted** after
training. It takes arrays and DataFrames, never model objects, so it drops into
any stack (scikit-learn, XGBoost, LightGBM, PyTorch, or just a CSV of
predictions) with zero coupling.

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
| Pre-training | `ml_xray.lint` | Dataset QA: leakage, drift, label noise, duplicates, imbalance, outliers. **(implemented)** |
| Post-training | `ml_xray.slices` | Error analysis / slice discovery — find where a model underperforms. *(Phase 2, scaffolded)* |
| Post-training | `ml_xray.embed` | Embedding diff — compare two embedding spaces. *(Phase 3, scaffolded)* |

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
ml-xray lint DATA --target y [--split col] [--fail-on error] [--html out.html]
ml-xray slices PREDS --features cols [--html out.html]        # Phase 2
ml-xray embed-diff A.npy B.npy [--ids ids.csv] [--html out.html]   # Phase 3
ml-xray report ...                                            # Phase 4
```

## Roadmap

- **Phase 1 (shipped as v0.1):** `ml_xray.lint` + HTML report + `ml-xray lint`
  CLI + synthetic-defect tests.
- **Phase 2:** `ml_xray.slices` + `ml-xray slices`.
- **Phase 3:** `ml_xray.embed` + `ml-xray embed-diff`.
- **Phase 4:** unified `ml-xray report` and richer `riskplot` viz backend.

## License

[MIT](LICENSE)
