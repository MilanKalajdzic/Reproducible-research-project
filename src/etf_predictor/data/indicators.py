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
    DataFrame.

    Parameters
    ----------
    categories : list[str] or None
        Subset of pandas-ta categories to compute. Pass ``None``
        (default) to compute all categories, replicating the paper's
        ~210 indicators.
    exclude_cols : list[str] or None
        Column names to drop after indicator computation.

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
        result = self._rename_for_ta(result)

        logger.info(
            "Computing indicators for categories: %s", self.categories
        )

        # pandas-ta 0.4.x removed Strategy — call each indicator
        # individually and collect results into a list of DataFrames.
        indicator_frames: list[pd.DataFrame] = []
        for indicator in self._get_indicator_list():
            try:
                fn = getattr(result.ta, indicator, None)
                if fn is None:
                    continue
                out = fn()
                if out is None:
                    continue
                if isinstance(out, pd.Series):
                    out = out.to_frame()
                elif isinstance(out, tuple):
                    out = pd.concat(
                        [x.to_frame() if isinstance(x, pd.Series) else x
                         for x in out if isinstance(x, (pd.Series, pd.DataFrame))],
                        axis=1
                    )
                if not isinstance(out, pd.DataFrame) or out.empty:
                    continue
                indicator_frames.append(out)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping indicator %s: %s", indicator, exc)

        if indicator_frames:
            indicators_df = pd.concat(indicator_frames, axis=1)
            # Drop duplicate columns that some indicators produce
            indicators_df = indicators_df.loc[
                :, ~indicators_df.columns.duplicated()
            ]
            result = pd.concat([result, indicators_df], axis=1)

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
        """Return all pandas-ta indicator names for selected categories.

        Falls back to inspecting the ta accessor directly if the
        Category dict is empty (API changed in 0.4.x).
        """
        try:
            import pandas_ta as ta  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pandas-ta is required.") from exc

        indicators: list[str] = []
        for category in self.categories:
            cat_indicators = getattr(ta, "Category", {}).get(category, [])
            indicators.extend(cat_indicators)

        # Fallback for new API where Category dict may be empty
        if not indicators:
            dummy = pd.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [1.5, 2.5, 3.5],
                    "low": [0.5, 1.5, 2.5],
                    "close": [1.2, 2.2, 3.2],
                    "volume": [1000.0, 1100.0, 1200.0],
                }
            )
            skip = {
                "strategy", "indicators", "categories",
                "ticker", "trades", "above", "below",
                "above_value", "below_value", "cross",
                "cross_value", "long_run", "short_run",
                "datetime_ordered", "reverse", "to_utc",
                "adjusted", "cores",
            }
            indicators = [
                m for m in dir(dummy.ta)
                if not m.startswith("_")
                and callable(getattr(dummy.ta, m))
                and m not in skip
            ]
            logger.info(
                "Category dict empty — using %d indicators from accessor.",
                len(indicators),
            )

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