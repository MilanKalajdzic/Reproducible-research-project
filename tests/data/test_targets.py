"""Tests for etf_predictor.data.targets."""

import pandas as pd
import pytest

from etf_predictor.data.targets import DOWN_LABEL, UP_LABEL, TargetBuilder


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {"Open": [40.0, 42.0, 41.0, 43.0, 43.0, 44.0]},
        index=dates,
    )


class TestTargetBuilder:
    def test_output_has_gamma_column(self, sample_df):
        builder = TargetBuilder()
        result = builder.build(sample_df)
        assert "Gamma" in result.columns

    def test_up_label_when_open_rises(self, sample_df):
        builder = TargetBuilder()
        result = builder.build(sample_df)
        # Day 1 (index 0 after drop): Open went 40→42, should be UP
        assert result["Gamma"].iloc[0] == UP_LABEL

    def test_down_label_when_open_falls(self, sample_df):
        builder = TargetBuilder()
        result = builder.build(sample_df)
        # Day 2 (index 1): Open went 42→41, should be DOWN
        assert result["Gamma"].iloc[1] == DOWN_LABEL

    def test_equal_open_is_down(self, sample_df):
        # When delta == 0, the paper assigns -1 ("otherwise")
        builder = TargetBuilder()
        result = builder.build(sample_df)
        # Day 4 (index 3): 43→43, should be DOWN
        assert result["Gamma"].iloc[3] == DOWN_LABEL

    def test_first_row_dropped(self, sample_df):
        builder = TargetBuilder()
        result = builder.build(sample_df)
        assert len(result) == len(sample_df) - 1

    def test_missing_open_raises(self):
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        builder = TargetBuilder()
        with pytest.raises(KeyError, match="Open"):
            builder.build(df)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            TargetBuilder(horizon=0)

    def test_custom_target_col(self, sample_df):
        builder = TargetBuilder(target_col="direction")
        result = builder.build(sample_df)
        assert "direction" in result.columns

    def test_class_balance_returns_series(self, sample_df):
        builder = TargetBuilder()
        result = builder.build(sample_df)
        balance = builder.class_balance(result)
        assert abs(balance.sum() - 1.0) < 1e-6
