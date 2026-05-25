"""
Train the MLP signal model and the LSTM value model on every ETF in the
processed dataset, evaluate both with expanding-window walk-forward
validation

    reports/figures/equity_<TICKER>.png            equity-curve comparison
    reports/figures/fold_accuracy_<TICKER>.png     per-fold accuracy bars
    reports/results/<TICKER>_walkforward_metrics.csv   summary metrics
    reports/results/<TICKER>_<model>_folds.csv         per-fold metrics
    reports/results/<TICKER>_<model>_predictions.csv   day-level signals

Usage
    PYTHONPATH=src python scripts/run_modeling.py
    # or
    make modeling
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from etf_predictor.data.loader import YahooFinanceLoader
from etf_predictor.data.pipeline import DataPipeline
from etf_predictor.models.equity import (
    build_comparison,
    plot_equity_curves,
    plot_fold_metric,
)
from etf_predictor.models.lstm_value import LSTMValueModel
from etf_predictor.models.mlp_signal import MLPSignalModel
from etf_predictor.models.walk_forward import WalkForwardValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_modeling")

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--initial-train", type=int, default=750,
        help="Rows in the first training window (~3y of trading days).",
    )
    p.add_argument(
        "--test-size", type=int, default=120,
        help="Test rows per fold (~6 months of trading days).",
    )
    p.add_argument(
        "--max-folds", type=int, default=None,
        help="Optional cap on the number of folds (useful for smoke tests).",
    )
    p.add_argument(
        "--mlp-epochs", type=int, default=30,
        help="Training epochs for the MLP per fold.",
    )
    p.add_argument(
        "--lstm-epochs", type=int, default=20,
        help="Training epochs for the LSTM per fold.",
    )
    p.add_argument(
        "--lstm-seq-len", type=int, default=20,
        help="LSTM look-back window length.",
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Subset of tickers to run (defaults to all processed).",
    )
    return p.parse_args()


def load_processed_or_run_pipeline(
    tickers_filter: list[str] | None,
) -> dict[str, pd.DataFrame]:
    """Load the scaled/cleaned per-ticker frames, building them if absent."""
    processed_dir = Path("data/processed")
    available = sorted(processed_dir.glob("*_processed.parquet"))
    if not available:
        logger.info("No processed parquet files — running DataPipeline.")
        datasets = DataPipeline().run()
    else:
        datasets = {}
        for path in available:
            ticker = path.stem.replace("_processed", "")
            datasets[ticker] = pd.read_parquet(path)
        logger.info("Loaded processed parquet for: %s", list(datasets.keys()))

    if tickers_filter:
        datasets = {t: d for t, d in datasets.items() if t in tickers_filter}
        if not datasets:
            raise ValueError(
                f"None of --tickers={tickers_filter} matched available tickers."
            )
    return datasets


def load_unscaled_close(ticker: str, index: pd.Index) -> pd.Series:
    """Return raw close series aligned with *index* (downloads/caches if needed)."""
    loader = YahooFinanceLoader(
        tickers=[ticker], include_benchmark=False,
    )
    raw = loader.load().get(ticker)
    if raw is None or raw.empty:
        raise RuntimeError(f"Could not load raw OHLCV for {ticker}.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()
    close.index = pd.to_datetime(close.index)
    aligned = close.reindex(index).ffill().bfill()
    if aligned.isna().any():
        raise RuntimeError(
            f"Unscaled close for {ticker} still has NaN after alignment."
        )
    return aligned.rename("Close")


def run_one(ticker: str, df: pd.DataFrame, args: argparse.Namespace) -> None:
    logger.info("===== %s =====", ticker)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "Gamma" not in df.columns:
        raise KeyError(f"{ticker}: processed frame is missing the 'Gamma' column.")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    close_unscaled = load_unscaled_close(ticker, df.index)

    validator = WalkForwardValidator(
        initial_train_size=args.initial_train,
        test_size=args.test_size,
        max_folds=args.max_folds,
    )

    mlp_factory = lambda: MLPSignalModel(epochs=args.mlp_epochs)
    lstm_factory = lambda: LSTMValueModel(
        epochs=args.lstm_epochs,
        sequence_length=args.lstm_seq_len,
    )

    logger.info("[%s] Running MLP walk-forward …", ticker)
    mlp_result = validator.run_mlp(df, close_unscaled, model_factory=mlp_factory)
    logger.info("[%s] Running LSTM walk-forward …", ticker)
    lstm_result = validator.run_lstm(df, close_unscaled, model_factory=lstm_factory)

    curves, metrics = build_comparison(
        [mlp_result, lstm_result],
        close_unscaled=close_unscaled,
    )

    logger.info("[%s] summary metrics:\n%s", ticker, metrics.round(4).to_string())

    metrics.to_csv(RESULTS_DIR / f"{ticker}_walkforward_metrics.csv")
    mlp_result.fold_metrics.to_csv(
        RESULTS_DIR / f"{ticker}_MLP_folds.csv", index=False,
    )
    lstm_result.fold_metrics.to_csv(
        RESULTS_DIR / f"{ticker}_LSTM_folds.csv", index=False,
    )
    mlp_result.predictions.to_csv(RESULTS_DIR / f"{ticker}_MLP_predictions.csv")
    lstm_result.predictions.to_csv(RESULTS_DIR / f"{ticker}_LSTM_predictions.csv")
    curves.to_csv(RESULTS_DIR / f"{ticker}_equity_curves.csv")

    plot_equity_curves(
        curves,
        title=f"{ticker} — walk-forward equity (MLP vs LSTM vs Buy & Hold)",
        out_path=FIGURES_DIR / f"equity_{ticker}.png",
    )
    plot_fold_metric(
        {"MLP": mlp_result.fold_metrics, "LSTM": lstm_result.fold_metrics},
        metric="accuracy",
        out_path=FIGURES_DIR / f"fold_accuracy_{ticker}.png",
    )


def main() -> None:
    args = parse_args()
    datasets = load_processed_or_run_pipeline(args.tickers)
    for ticker, df in datasets.items():
        try:
            run_one(ticker, df, args)
        except Exception:  # noqa: BLE001
            logger.exception("[%s] modeling run failed — continuing.", ticker)


if __name__ == "__main__":
    main()
