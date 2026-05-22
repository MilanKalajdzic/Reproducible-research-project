"""
indicators.py
-------------
Wraps the pandas-ta library to compute the full set of technical
indicators used in Sagaceta-Mejía et al. (2024), Section 2.3.

pandas-ta is applied to the six base features sourced from
Yahoo Finance: Open, High, Low, Close, Adj Close, Volume.
This expands the feature set to ~216 daily features.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Categories available in pandas-ta that we want to compute.
# Matches the paper's taxonomy: Candles, Cycles, Momentum, Overlap,
# Performance, Statistics, Trend, Utility, Volatility, Volume.
ALL_CATEGORIES = [
    "candles",
    "cycles",
    "momentum",
    "overlap",
    "performance",
    "statistics",
    "trend",
    "utility",
    "volatility",
    "volume",
]


class TechnicalIndicators:
    """Compute technical indicators using pandas-ta.

    Applies the full pandas-ta strategy (or a filtered subset of
    categories) to a raw OHLCV DataFrame and returns the augmented
    DataFrame. Indicator computation is the most expensive step in
    the pipeline; results are optionally cached.

    Parameters
    ----------
    categories : list[str] or None
        Subset of pandas-ta categories to compute. Pass ``None``
        (default) to compute all categories, replicating the paper's
        ~210 indicators.
    exclude_cols : list[str] or None
        Column names to drop after indicator computation (e.g.
        intermediate columns not needed downstream).

    Examples
    --------
    >>> ti = TechnicalIndicators()
    >>> df_indicators = ti.compute(df_ohlcv)
    >>> print(df_indicators.shape)
    (2516, 216)
    """

    def __init__(
        self,
        categories: Optional[list[str]] = None,
        exclude_cols: Optional[list[str]] = None,
    ) -> None:
        self.categories = categories or ALL_CATEGORIES
        self.exclude_cols = exclude_cols or []

        # Validate categories
        invalid = set(self.categories) - set(ALL_CATEGORIES)
        if invalid:
            raise ValueError(
                f"Unknown pandas-ta categories: {invalid}. "
                f"Valid options: {ALL_CATEGORIES}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured technical indicators for *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame with columns
            ``["Open", "High", "Low", "Close", "Adj Close", "Volume"]``
            and a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Original columns plus all computed indicator columns.
            Column count should approach ~216 for the full strategy.

        Raises
        ------
        ImportError
            If ``pandas_ta`` is not installed.
        """
        try:
            import pandas_ta as ta  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pandas-ta is required. Install it with: "
                "pip install pandas-ta"
            ) from exc

        result = df.copy()

        # pandas-ta expects lowercase column names for some indicators
        result = self._rename_for_ta(result)

        logger.info(
            "Computing indicators for categories: %s", self.categories
        )

        # Build a custom strategy from the selected categories
        strategy = ta.Strategy(
            name="etf_predictor_strategy",
            description="All selected categories from pandas-ta",
            ta=[
                {"kind": indicator}
                for indicator in self._get_indicator_list()
            ],
        )

        result.ta.strategy(strategy, verbose=False)

        # Restore original column case
        result = self._restore_col_names(result, df)

        # Drop any explicitly excluded columns
        cols_to_drop = [
            c for c in self.exclude_cols if c in result.columns
        ]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)
            logger.debug("Dropped columns: %s", cols_to_drop)

        n_new = result.shape[1] - df.shape[1]
        logger.info(
            "Indicator computation complete. "
            "Original columns: %d, New indicator columns: %d, Total: %d",
            df.shape[1],
            n_new,
            result.shape[1],
        )
        return result

    def indicator_names(self) -> list[str]:
        """Return the list of indicator function names that will be computed.

        Returns
        -------
        list[str]
            Sorted list of pandas-ta indicator names.
        """
        return sorted(self._get_indicator_list())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_indicator_list(self) -> list[str]:
        """Return all pandas-ta indicator names for selected categories."""
        try:
            import pandas_ta as ta  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pandas-ta is required.") from exc

        indicators: list[str] = []
        for category in self.categories:
            # pandas-ta organises indicators in ta.Category dict
            cat_indicators = ta.Category.get(category, [])
            indicators.extend(cat_indicators)
        return indicators

    @staticmethod
    def _rename_for_ta(df: pd.DataFrame) -> pd.DataFrame:
        """Rename 'Adj Close' → 'Adj_Close' for pandas-ta compatibility."""
        return df.rename(columns={"Adj Close": "Adj_Close"})

    @staticmethod
    def _restore_col_names(
        result: pd.DataFrame, original: pd.DataFrame
    ) -> pd.DataFrame:
        """Restore 'Adj_Close' → 'Adj Close' in the result."""
        if "Adj_Close" in result.columns and "Adj Close" in original.columns:
            result = result.rename(columns={"Adj_Close": "Adj Close"})
        return result
