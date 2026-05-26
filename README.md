# ETF Trend Predictor

Predicting ETF trend direction using optimized technical indicators and
neural networks. Based on [Sagaceta-Mejía et al. (2024)](https://doi.org/10.1515/econ-2022-0073),
adapted for three European ETFs: **IEUR**, **FEZ**, and **EUFN**, with
**IVV** (S&P 500) as a developed-market benchmark.

---

## Quick Start

### Step 1 — pull the image

    docker pull milankalajdzic/etf-predictor:latest

> **Macs:** if the pull fails with add
> `--platform linux/amd64` to both the `pull` and `run` commands.
> Docker Desktop will emulate linux envi.

### Step 2 — create an empty folder for the rendered reports

The container publishes the three HTML files to `/output/` inside
itself.

    mkdir output         

### Step 3 — run the full pipeline

The default `CMD` is `make report`, which runs the **entire pipeline
from scratch**: download/process data → statistical analysis → MLP
and LSTM walk-forward training → render all three Quarto reports
→ copy them to `/output/`.

    # macOS / Linux / WSL
    docker run --rm \
        -v "$PWD/output:/output" \
        milankalajdzic/etf-predictor:latest

    # Windows PowerShell
    docker run --rm `
        -v "${PWD}/output:/output" `
        milankalajdzic/etf-predictor:latest

Expect **~10 minutes** of runtime — 4 tickers × ~22 folds × MLP + LSTM,
all on CPU. 

### Step 4 — open the reports

When the container exits, three self-contained HTML files appear in
your `output/` folder:

- `output/eda_report.html` — data preparation & feature engineering
- `output/statistical_analysis.html` — variance / correlation / RF / ARIMA
- `output/modeling_report.html` — MLP + LSTM walk-forward backtest

---

## Power-user options

### Build locally instead of pulling

    docker compose build
    docker compose run --rm \
        -v "$PWD/output:/output" \
        etf-predictor

### Open a shell inside the container

    docker compose run --rm etf-predictor bash

    # then, inside the container:
    make data       # download and process ETF data
    make test       # run 34 unit tests
    make eda        # regenerate EDA figures only
    make analysis   # statistical analysis only
    make modeling   # MLP + LSTM walk-forward only
    make report     # full pipeline + render all three reports + publish to /output
    make render     # *just* re-render Quarto reports from existing CSVs (fast)
    make publish    # copy reports/*.html → /output

---

## Dataset

| Ticker | Fund | Market | Data range |
|--------|------|--------|------------|
| **IEUR** | iShares Core MSCI Europe ETF | Broad European | 2014-06-12 → 2026-04-30 |
| **FEZ** | SPDR EURO STOXX 50 ETF | Eurozone large-cap | 2010-01-04 → 2026-04-30 |
| **EUFN** | iShares MSCI Europe Financials ETF | European financials | 2010-02-03 → 2026-04-30 |
| **IVV** | iShares Core S&P 500 ETF | US benchmark | 2010-01-04 → 2026-04-30 |

Data sourced from Yahoo Finance via yfinance. Raw data is cached as
parquet files under data/raw/ and is not committed to the repository.

Note: IEUR data starts June 2014 — Yahoo Finance data availability limitation.

---

## Data Preparation Pipeline

    Yahoo Finance (yfinance)
            ↓
    YahooFinanceLoader     → data/raw/*.parquet  (cached)
            ↓
    TechnicalIndicators    → ~260 features via pandas-ta 0.4.x
            ↓
    TargetBuilder          → Γ(t) ∈ {+1, -1}  (Open price direction)
            ↓
    MinMaxScaler           → all features scaled to [0, 1]
            ↓
    DataCleaner            → sparse columns dropped (>95% NaN),
                             remaining NaNs forward-filled
            ↓
    data/processed/*.parquet  ← modeling team entry point

**Processed dataset summary:**

| Ticker | Rows | Features | UP % | DOWN % |
|--------|------|----------|------|--------|
| IEUR | ~2400 | ~263 | 53.3 | 46.7 |
| FEZ | ~3500 | ~263 | 52.4 | 47.6 |
| EUFN | ~3500 | ~265 | 51.7 | 48.3 |
| IVV | ~3500 | ~263 | 55.7 | 44.3 |

---

## Statistical Analysis Pipeline

In addition to EDA, the project includes a statistical analysis layer for
feature diagnostics and time-series baselines.

This stage performs:

- feature variance analysis
- correlation redundancy analysis
- Random Forest feature importance
- ADF stationarity tests
- ARIMA(1,1,1) baseline forecasting

Run it with:

    make analysis

Outputs are saved to:

    reports/results/

The corresponding Quarto report can be rendered with:

    quarto render reports/statistical_analysis.qmd
---

## Project Structure

```text
├── src/etf_predictor/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── correlation_analysis.py
│   │   ├── feature_analysis.py
│   │   ├── feature_importance.py
│   │   ├── time_series.py
│   │   └── variance_analysis.py
│   └── data/
│       ├── __init__.py
│       ├── loader.py
│       ├── targets.py
│       ├── indicators.py
│       ├── preprocessing.py
│       ├── pipeline.py
│       └── visualization.py
├── tests/
│   ├── analysis/
│   │   ├── test_correlation_analysis.py
│   │   ├── test_feature_analysis.py
│   │   ├── test_feature_importance.py
│   │   └── test_time_series.py
│   └── data/
│       ├── test_loader.py
│       ├── test_targets.py
│       └── test_preprocessing.py
├── reports/
│   ├── eda_report.qmd
│   ├── statistical_analysis.qmd
│   └── results/
│       ├── *_summary.csv
│       ├── *_feature_importance.csv
│       ├── *_high_corr.csv
│       ├── *_variance.csv
│       └── *_arima_summary.csv
├── scripts/
│   ├── generate_eda.py
│   └── run_analysis.py
```
---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make data` | Download and process all ETF data |
| `make eda` | Generate all EDA figures to reports/figures/ |
| `make test` | Run 34 pytest unit tests |
| `make coverage` | Pytest with HTML coverage report |
| `make docs` | Build Sphinx HTML docs to docs/_build/html/ |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |
| `make install-dev` | Install deps + pre-commit hooks (local dev only) |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run pipeline in Docker |
| `make docker-test` | Run tests in Docker |
| `make analysis` | Run statistical analysis pipeline |
| `make modeling` | Train MLP + LSTM with walk-forward validation and equity plots |
---

## Rendering the Reports

Three Quarto reports live under `reports/`:

| Report | Source | Description |
|--------|--------|-------------|
| EDA | `reports/eda_report.qmd` | Data preparation, indicator construction, class balance |
| Statistical analysis | `reports/statistical_analysis.qmd` | Variance / correlation / ADF / Random Forest importance / ARIMA |
| Modeling | `reports/modeling_report.qmd` | MLP + LSTM walk-forward backtest, equity curves, top-10 feature ablation |

Render them all in one go (inside the container):

    make report

or one at a time:

    quarto render reports/eda_report.qmd
    quarto render reports/statistical_analysis.qmd
    quarto render reports/modeling_report.qmd

If `reports/` is bind-mounted with `-v "$PWD/reports:/app/reports"`,
the rendered HTML appears on the host automatically. Otherwise
copy it out:

    docker cp <container_name>:/app/reports/modeling_report.html reports/modeling_report.html



---

## Viewing the Sphinx Docs

    # inside the container
    make docs

    # copy the built HTML to your local machine (from PowerShell)
    mkdir docs\_build
    docker cp <container_name>:/app/docs/_build/html docs/_build/html

Then open docs/_build/html/index.html in your browser.

---

## Compatibility Notes

- pandas-ta 0.4.x removed the Strategy API — indicators are computed
  individually via the df.ta accessor
- yfinance 0.2+ returns MultiIndex columns — these are flattened in
  the loader
- Python 3.12 required (pandas-ta 0.4.x constraint)
- statsmodels is required for ADF testing and ARIMA baselines

---



## References

Sagaceta-Mejía, A. R., Sánchez-Gutiérrez, M. E., & Fresán-Figueroa, J. A.
(2024). An intelligent approach for predicting stock market movements in
emerging markets using optimized technical indicators and neural networks.
*Economics*, 18, 20220073. https://doi.org/10.1515/econ-2022-0073
