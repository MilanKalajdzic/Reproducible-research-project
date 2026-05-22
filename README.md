# ETF Trend Predictor

Predicting ETF trend direction using optimized technical indicators and
neural networks. Based on [Sagaceta-Mejía et al. (2024)](https://doi.org/10.1515/econ-2022-0073),
adapted for three European ETFs: **IEUR**, **FEZ**, and **EUFN**, with
**IVV** (S&P 500) as a developed-market benchmark.

---

## Quick Start (Docker — recommended)

    # 1. Pull the image from DockerHub
    docker pull !!!TBD

    # 2. Or build locally
    docker compose build

    # 3. Download data and run the pipeline
    docker compose run --rm etf-predictor make data

    # 4. Generate EDA figures
    docker compose run --rm etf-predictor make eda

    # 5. Build Sphinx documentation
    docker compose run --rm etf-predictor make docs

    # 6. Run tests
    docker compose run --rm test

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
    DataCleaner            → sparse columns dropped, NaNs forward-filled
            ↓
    data/processed/*.parquet  ← modeling entry point

**Processed dataset summary:**

| Ticker | Rows | Features | UP % | DOWN % |
|--------|------|----------|------|--------|
| IEUR | ~2400 | ~263 | 53.3 | 46.7 |
| FEZ | ~3500 | ~263 | 52.4 | 47.6 |
| EUFN | ~3500 | ~265 | 51.7 | 48.3 |
| IVV | ~3500 | ~263 | 55.7 | 44.3 |

---

## Project Structure

    ├── src/etf_predictor/
    │   └── data/
    │       ├── loader.py        # YahooFinanceLoader — download + parquet cache
    │       ├── targets.py       # TargetBuilder      — Γ(t) label construction
    │       ├── indicators.py    # TechnicalIndicators — pandas-ta 0.4.x wrapper
    │       ├── preprocessing.py # MinMaxScaler, DataCleaner
    │       ├── pipeline.py      # DataPipeline       — orchestrates all steps
    │       └── visualization.py # EDAVisualizer      — EDA figures
    ├── tests/data/              # 25 pytest unit tests
    ├── reports/
    │   └── eda_report.qmd       # Quarto report
    ├── scripts/
    │   └── generate_eda.py      # EDA figure generation script
    ├── docs/source/             # Sphinx documentation source
    ├── Dockerfile               # Python 3.12-slim, single image
    ├── docker-compose.yml       # Services: etf-predictor, test, notebook
    ├── Makefile                 # Automation targets
    ├── pyproject.toml           # Dependencies + ruff linting config
    └── .pre-commit-config.yaml  # ruff linter + formatter hooks

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
| `make install-dev` | Install deps + pre-commit hooks (local dev) |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run pipeline in Docker |
| `make docker-test` | Run tests in Docker |

---

## Requirements

- Docker Desktop
- No local Python installation needed — everything runs inside the container


## References

Sagaceta-Mejía, A. R., Sánchez-Gutiérrez, M. E., & Fresán-Figueroa, J. A.
(2024). An intelligent approach for predicting stock market movements in
emerging markets using optimized technical indicators and neural networks.
*Economics*, 18, 20220073. https://doi.org/10.1515/econ-2022-0073