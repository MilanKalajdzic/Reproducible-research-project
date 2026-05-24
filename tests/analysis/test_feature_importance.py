import numpy as np
import pandas as pd

from etf_predictor.analysis.feature_importance import FeatureImportanceAnalyzer


def test_top_features_returns_expected_shape() -> None:
    rng = np.random.default_rng(42)
    n = 300

    signal = rng.normal(size=n)
    noise_1 = rng.normal(size=n)
    noise_2 = rng.normal(size=n)

    y = pd.Series(np.where(signal > 0, 1, -1))
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise_1": noise_1,
            "noise_2": noise_2,
        }
    )

    analyzer = FeatureImportanceAnalyzer(random_state=42)
    top = analyzer.top_features(X, y, top_n=2)

    assert len(top) == 2
    assert {"feature", "importance"}.issubset(top.columns)
    assert "signal" in top["feature"].values


def test_fit_importance_stores_accuracy() -> None:
    rng = np.random.default_rng(42)
    n = 300

    signal = rng.normal(size=n)
    y = pd.Series(np.where(signal > 0, 1, -1))
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=n),
        }
    )

    analyzer = FeatureImportanceAnalyzer(random_state=42)
    result = analyzer.fit_importance(X, y)

    assert "accuracy" in result.attrs
    assert 0 <= result.attrs["accuracy"] <= 1
