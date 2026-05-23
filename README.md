# ETF Trend Predictor

Predicting ETF trend direction using optimized technical indicators and
neural networks. Based on [Sagaceta-Mejía et al. (2024)](https://doi.org/10.1515/econ-2022-0073),
adapted for three European ETFs: **IEUR**, **FEZ**, and **EUFN**, with
**IVV** (S&P 500) as a developed-market benchmark.

---

## Quick Start (Docker — recommended)

    # 1. Pull the image from DockerHub
    docker pull ##!!!!!TBD!!!!!

    # 2. Or build locally
    docker compose build

    # 3. Open a shell inside the container
    docker compose run --rm etf-predictor bash

    # 4. Inside the container — run everything in order:
    make data       # download and process ETF data
    make test       # run 25 unit tests
    make eda        # generate EDA figures to reports/figures/
    PYTHONPATH=src python scripts/run_analysis.py   # run statistical analysis
    make docs       # build Sphinx HTML docs to docs/_build/html/

    # 5. Render the Quarto report (inside container)
    quarto render reports/eda_report.qmd
    quarto render reports/statistical_analysis.qmd

    # 6. Copy rendered report out to your local machine (from PowerShell)
    docker cp <container_name>:/app/reports/eda_report.html reports/eda_report.html

    # 7. Copy built docs out to your local machine (from PowerShell)
    docker cp <container_name>:/app/docs/_build/html docs/_build/html

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

    PYTHONPATH=src python scripts/run_analysis.py

Outputs are saved to:

    reports/results/

The corresponding Quarto report can be rendered with:

    quarto render reports/statistical_analysis.qmd
---

## Project Structure

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

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make data` | Download and process all ETF data |
| `make eda` | Generate all EDA figures to reports/figures/ |
| `make test` | Run 25 pytest unit tests |
| `make coverage` | Pytest with HTML coverage report |
| `make docs` | Build Sphinx HTML docs to docs/_build/html/ |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |
| `make install-dev` | Install deps + pre-commit hooks (local dev only) |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run pipeline in Docker |
| `make docker-test` | Run tests in Docker |

---

## Rendering the Report

The Quarto report must be rendered inside the Docker container where all
dependencies are available:

    # inside the container
    quarto render reports/eda_report.qmd
    quarto render reports/statistical_analysis.qmd

    # copy the HTML output to your local machine (from PowerShell)
    docker cp <container_name>:/app/reports/eda_report.html reports/eda_report.html
    docker cp <container_name>:/app/reports/eda_report_files reports/eda_report_files

Then open reports/eda_report.html in your browser.

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

---

## Requirements

- Docker Desktop (no local Python installation needed)
- All dependencies are pinned in pyproject.toml and baked into the image

---

## References

Sagaceta-Mejía, A. R., Sánchez-Gutiérrez, M. E., & Fresán-Figueroa, J. A.
(2024). An intelligent approach for predicting stock market movements in
emerging markets using optimized technical indicators and neural networks.
*Economics*, 18, 20220073. https://doi.org/10.1515/econ-2022-0073