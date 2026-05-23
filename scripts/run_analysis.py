from etf_predictor.data.pipeline import DataPipeline
from etf_predictor.analysis.feature_analysis import FeatureAnalyzer
from etf_predictor.analysis.correlation_analysis import CorrelationAnalyzer

def main() -> None:
    pipeline = DataPipeline()
    datasets = pipeline.run()

    for ticker, df in datasets.items():
        print(f"\n===== {ticker} =====")

        X = df.drop(columns=["Gamma"])
        y = df["Gamma"]

        analyzer = FeatureAnalyzer(X)

        print("\nTop 10 variances:")
        print(analyzer.variance().sort_values(ascending=False).head(10))

        corr_analyzer = CorrelationAnalyzer(X)
        high_corr = corr_analyzer.highly_correlated_pairs(threshold=0.95)

        print("\nHighly correlated pairs (> 0.95):")
        print(len(high_corr))

        if not high_corr.empty:
            print(high_corr.head(10))

        print("\nCorrelation matrix shape:")
        print(analyzer.correlation_matrix().shape)

        print("\nTarget distribution:")
        print(y.value_counts(normalize=True))


if __name__ == "__main__":
    main()