from pathlib import Path

import pandas as pd

from etf_predictor.analysis.correlation_analysis import CorrelationAnalyzer
from etf_predictor.analysis.feature_analysis import FeatureAnalyzer
from etf_predictor.analysis.feature_importance import FeatureImportanceAnalyzer
from etf_predictor.data.pipeline import DataPipeline


def main() -> None:
    pipeline = DataPipeline()
    datasets = pipeline.run()

    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in datasets.items():
        print(f"\n===== {ticker} =====")

        X = df.drop(columns=["Gamma"])
        y = df["Gamma"]

        analyzer = FeatureAnalyzer(X)
        variance_series = analyzer.variance().sort_values(ascending=False)

        print("\nTop 10 variances:")
        print(variance_series.head(10))
        variance_series.to_csv(
            results_dir / f"{ticker}_variance.csv",
            header=["variance"],
        )

        corr_analyzer = CorrelationAnalyzer(X)
        high_corr = corr_analyzer.highly_correlated_pairs(threshold=0.95)

        print("\nHighly correlated pairs (> 0.95):")
        print(len(high_corr))

        if not high_corr.empty:
            print(high_corr.head(10))

        high_corr.to_csv(
            results_dir / f"{ticker}_high_corr.csv",
            index=False,
        )

        print("\nCorrelation matrix shape:")
        print(analyzer.correlation_matrix().shape)

        importance_analyzer = FeatureImportanceAnalyzer()
        top_features = importance_analyzer.top_features(X, y, top_n=10)

        print("\nTop 10 feature importances:")
        print(top_features)

        full_importance = importance_analyzer.fit_importance(X, y)

        print("\nRandom Forest test accuracy:")
        print(round(full_importance.attrs["accuracy"], 4))

        full_importance.to_csv(
            results_dir / f"{ticker}_feature_importance.csv",
            index=False,
        )

        summary_df = pd.DataFrame(
            {
                "ticker": [ticker],
                "n_features": [X.shape[1]],
                "n_high_corr_pairs": [len(high_corr)],
                "rf_test_accuracy": [full_importance.attrs["accuracy"]],
                "gamma_up_share": [(y == 1).mean()],
                "gamma_down_share": [(y == -1).mean()],
            }
        )
        summary_df.to_csv(
            results_dir / f"{ticker}_summary.csv",
            index=False,
        )

        print("\nTarget distribution:")
        print(y.value_counts(normalize=True))


if __name__ == "__main__":
    main()