"""
lstm_value.py
-------------
LSTM regressor that predicts the next-day Close price from a fixed 
window of past features, then converts the predicted price into signal

"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    """Single-layer LSTM with a fully-connected regression head."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMValueModel:
    """LSTM regressor on next-day Close, converted to a ±1 trading signal.

    Parameters
    ----------
    sequence_length : int
        Length of the look-back window fed to the LSTM.
    hidden_size : int
        Number of LSTM hidden units.
    num_layers : int
        Stacked LSTM layers.
    dropout : float
        Dropout applied between layers and before the head.
    learning_rate : float
        Adam learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    weight_decay : float
        L2 regularisation for Adam.
    price_col : str
        Column in ``X`` (or attached via ``fit``) holding the unscaled
        ``Close`` series used to (a) build the regression target and
        (b) derive the trading signal at inference time.
    device : str or None
        ``"cpu"``, ``"cuda"`` or ``None`` to auto-detect.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        price_col: str = "Close",
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.price_col = price_col
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[_LSTMNet] = None
        self._feature_cols: Optional[list[str]] = None
        # Min/Max for Close, fit on training data — used to invert the
        # scaled regression target back to price space.
        self._close_min: Optional[float] = None
        self._close_max: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        close_unscaled: pd.Series,
    ) -> "LSTMValueModel":
        """Train the LSTM to predict next-day Close.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled feature matrix indexed by date.
        close_unscaled : pd.Series
            Unscaled ``Close`` series aligned with ``X.index``. Used to
            build the regression target and to back-transform predictions
            into price space.

        Returns
        -------
        LSTMValueModel
            ``self`` for chaining.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if not X.index.equals(close_unscaled.index):
            close_unscaled = close_unscaled.reindex(X.index)
        if close_unscaled.isna().any():
            raise ValueError(
                "close_unscaled contains NaN after aligning to X.index."
            )

        self._feature_cols = list(X.columns)
        self._close_min = float(close_unscaled.min())
        self._close_max = float(close_unscaled.max())
        denom = self._close_max - self._close_min
        if denom == 0:
            raise ValueError("Close series has zero variance — cannot scale.")

        close_scaled = (close_unscaled - self._close_min) / denom

        seqs, targets = self._make_sequences(
            X.to_numpy(dtype=np.float32),
            close_scaled.to_numpy(dtype=np.float32),
        )
        if len(seqs) == 0:
            raise ValueError(
                f"Not enough rows ({len(X)}) for sequence_length="
                f"{self.sequence_length}."
            )

        x_t = torch.from_numpy(seqs).to(self.device)
        y_t = torch.from_numpy(targets).to(self.device)
        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._model = _LSTMNet(
            n_features=x_t.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimiser = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                preds = self._model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimiser.step()
                total += loss.item() * xb.size(0)
            avg = total / len(x_t)
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info(
                    "LSTM epoch %d/%d  loss=%.6f",
                    epoch + 1, self.epochs, avg,
                )
        return self

    def predict_close(
        self,
        X: pd.DataFrame,
        history: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Predict next-day Close (unscaled) for each row of *X*.

        Because the LSTM needs ``sequence_length`` prior rows of features
        to make a prediction, callers should pass the *most recent*
        training rows in ``history`` so that predictions can start from
        the very first row of ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature rows to predict for.
        history : pd.DataFrame or None
            Optional feature rows immediately preceding ``X`` (typically
            the tail of the training set). If provided, all rows in
            ``X`` will receive a prediction. If ``None``, the first
            ``sequence_length - 1`` predictions will be NaN.

        Returns
        -------
        pd.Series
            Predicted next-day Close, indexed by ``X.index``.
        """
        self._check_fitted()
        if history is None:
            feature_df = X[self._feature_cols]
        else:
            feature_df = pd.concat(
                [history[self._feature_cols], X[self._feature_cols]]
            )

        arr = feature_df.to_numpy(dtype=np.float32)
        windows = []
        for i in range(self.sequence_length - 1, len(arr)):
            windows.append(arr[i - self.sequence_length + 1: i + 1])
        if not windows:
            return pd.Series(
                [np.nan] * len(X), index=X.index, name="pred_close"
            )

        x_t = torch.from_numpy(np.stack(windows)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            preds_scaled = self._model(x_t).cpu().numpy()

        preds_close = (
            preds_scaled * (self._close_max - self._close_min)
            + self._close_min
        )

        # Align predictions with the requested output index. Each window
        # ends at row (sequence_length - 1 + i) of feature_df.
        last_idx = feature_df.index[self.sequence_length - 1:]
        full = pd.Series(preds_close, index=last_idx, name="pred_close")
        return full.reindex(X.index)

    def predict_signal(
        self,
        X: pd.DataFrame,
        current_close: pd.Series,
        history: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Convert predicted-close into a ±1 trading signal.

        Parameters
        ----------
        X : pd.DataFrame
            Feature rows to predict for.
        current_close : pd.Series
            Today's (unscaled) Close price, aligned with ``X.index``.
            Used to compute the implied return.
        history : pd.DataFrame or None
            See :meth:`predict_close`.

        Returns
        -------
        np.ndarray
            Signals in ``{+1, -1}``; NaN-prediction rows default to -1
            so they do not contribute spurious long positions.
        """
        pred = self.predict_close(X, history=history)
        if not current_close.index.equals(X.index):
            current_close = current_close.reindex(X.index)
        ret = (pred - current_close) / current_close
        signal = np.where(ret.fillna(0.0) > 0, 1, -1).astype(np.int8)
        return signal

    def _make_sequences(
        self, features: np.ndarray, target_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (n_samples, seq_len, n_features) and next-day targets."""
        seqs, ys = [], []
        for i in range(self.sequence_length, len(features)):
            seqs.append(features[i - self.sequence_length: i])
            ys.append(target_scaled[i])
        return (
            np.stack(seqs) if seqs else np.empty((0,)),
            np.asarray(ys, dtype=np.float32),
        )

    def _check_fitted(self) -> None:
        if self._model is None or self._close_min is None:
            raise RuntimeError("Call fit() before predict_*().")
