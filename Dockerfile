# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: base image with system deps
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

# Prevent .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages needed by some Python deps (lxml, pandas, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install ".[dev,notebook]"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: final image
# ─────────────────────────────────────────────────────────────────────────────
FROM deps AS final

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY reports/ ./reports/
COPY scripts/ ./scripts/
COPY Makefile ./
COPY .pre-commit-config.yaml ./

# Create data directories that will be populated at runtime
# (raw data is downloaded, not baked into the image)
RUN mkdir -p data/raw data/processed reports/figures

# Default command: run the full pipeline then drop into bash
CMD ["python", "-m", "etf_predictor.data.pipeline"]

# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────
LABEL maintainer="ETF Predictor Team" \
      description="ETF trend prediction — data preparation" \
      version="0.1.0"
