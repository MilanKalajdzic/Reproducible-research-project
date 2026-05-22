# ─────────────────────────────────────────────────────────────────────────────
# ETF Predictor — Makefile
# Usage:  make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON     := python
SRC        := src/etf_predictor
TESTS      := tests
REPORTS    := reports
DOCS_SRC   := docs/source
DOCS_BUILD := docs/_build/html
IMAGE_NAME := etf-predictor

.PHONY: help install install-dev lint format test coverage \
        data eda docs clean docker-build docker-run docker-test

# ── Default target ───────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ETF Predictor — available targets"
	@echo "──────────────────────────────────"
	@echo "  install       Install package (production deps)"
	@echo "  install-dev   Install package + dev deps + pre-commit hooks"
	@echo "  lint          Run ruff linter"
	@echo "  format        Run ruff formatter"
	@echo "  test          Run pytest"
	@echo "  coverage      Run pytest with HTML coverage report"
	@echo "  data          Download & process all ETF data"
	@echo "  eda           Generate all EDA figures to reports/figures/"
	@echo "  docs          Build Sphinx HTML documentation"
	@echo "  clean         Remove build/cache/data artefacts"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run pipeline inside Docker container"
	@echo "  docker-test   Run tests inside Docker container"
	@echo ""

# ── Installation ─────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebook]"
	pre-commit install

# ── Linting & formatting ─────────────────────────────────────────────────────
lint:
	ruff check $(SRC) $(TESTS)

format:
	ruff format $(SRC) $(TESTS)

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	pytest $(TESTS) -v

coverage:
	pytest $(TESTS) --cov=$(SRC) --cov-report=html --cov-report=term
	@echo "HTML report: htmlcov/index.html"

# ── Data pipeline ────────────────────────────────────────────────────────────
data:
	$(PYTHON) -c "\
from etf_predictor.data.pipeline import DataPipeline; \
p = DataPipeline(); \
datasets = p.run(); \
print(p.summary(datasets))"

# ── EDA figures ──────────────────────────────────────────────────────────────
eda:
	$(PYTHON) scripts/generate_eda.py

# ── Documentation ────────────────────────────────────────────────────────────
docs:
	sphinx-build -b html $(DOCS_SRC) $(DOCS_BUILD)
	@echo "Docs built: $(DOCS_BUILD)/index.html"

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache/ htmlcov/ .coverage
	rm -rf $(DOCS_BUILD)

clean-data:
	rm -rf data/raw/ data/processed/
	@echo "Raw and processed data removed."

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE_NAME):latest .

docker-run:
	docker compose run --rm etf-predictor make data

docker-test:
	docker compose run --rm test
