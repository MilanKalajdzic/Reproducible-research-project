"""Neural network models for ETF trend prediction.

Two complementary approaches:

- ``MLPSignalModel`` — multilayer perceptron that directly classifies the
  binary trend direction Γ(t) ∈ {+1, -1}.
- ``LSTMValueModel`` — recurrent network that regresses the next-day
  ``Close`` price; the sign of the predicted return is used as the
  trading signal.

Both models share a common interface (``fit``/``predict_signal``) so they
can be swapped into the same walk-forward backtest harness.
"""

from etf_predictor.models.mlp_signal import MLPSignalModel
from etf_predictor.models.lstm_value import LSTMValueModel

__all__ = ["MLPSignalModel", "LSTMValueModel"]
