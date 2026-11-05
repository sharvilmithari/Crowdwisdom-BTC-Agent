"""
Models and Baselines Module.
- Simple Persistence / No-change baseline (future price = current close).
- Ridge / ElasticNet linear baseline.
- Random Forest and Gradient Boosting regressors for n+0..n+5 horizons.
- Model serialization and inference pipeline.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import setup_logger

logger = setup_logger("models")


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Persistence baseline: forecasts future price equal to current close."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Assumes current close is provided or first column
        if isinstance(X, pd.DataFrame):
            return X["close"].values
        return X[:, 0]


class HorizonForecaster:
    """Multi-horizon wrapper training an estimator per horizon n+0..n+5.
    By default predicts stationary returns (target_ret_h) and converts to price forecasts,
    preventing non-stationary tree model extrapolation failure.
    """

    def __init__(self, model_type: str = "gradient_boosting", random_state: int = 42, predict_returns: bool = True):
        self.model_type = model_type
        self.random_state = random_state
        self.predict_returns = predict_returns
        self.models: Dict[int, Any] = {}
        self.scalers: Dict[int, StandardScaler] = {}

    def _get_base_estimator(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=40, max_depth=5, random_state=self.random_state, n_jobs=-1)
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        else:
            return GradientBoostingRegressor(n_estimators=40, max_depth=3, learning_rate=0.03, random_state=self.random_state)

    def fit(self, X: pd.DataFrame, df_targets: pd.DataFrame, feature_cols: List[str], horizons: List[int] = [0, 1, 2, 3, 4, 5]):
        """Fits separate models for each forecast horizon on training fold only."""
        X_mat = X[feature_cols].copy().fillna(0.0)

        for h in horizons:
            if h == 0:
                continue

            target_col = f"target_ret_{h}" if (self.predict_returns and f"target_ret_{h}" in df_targets.columns) else f"target_close_{h}"
            y = df_targets[target_col].values

            # Filter valid (non-null) targets
            valid_mask = ~np.isnan(y)
            X_clean = X_mat.loc[valid_mask].values
            y_clean = y[valid_mask]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            model = self._get_base_estimator()
            model.fit(X_scaled, y_clean)

            self.scalers[h] = scaler
            self.models[h] = model

        return self

    def predict(self, X: pd.DataFrame, feature_cols: List[str], horizons: List[int] = [0, 1, 2, 3, 4, 5]) -> Dict[int, np.ndarray]:
        """Predicts price for each horizon."""
        X_mat = X[feature_cols].copy().fillna(0.0).values
        closes = X["close"].values if "close" in X.columns else None
        predictions = {}

        for h in horizons:
            if h == 0:
                predictions[0] = closes.copy() if closes is not None else np.zeros(len(X))
            elif h in self.models:
                scaler = self.scalers[h]
                X_scaled = scaler.transform(X_mat)
                raw_pred = self.models[h].predict(X_scaled)
                if self.predict_returns and closes is not None:
                    predictions[h] = closes * (1.0 + raw_pred)
                else:
                    predictions[h] = raw_pred
            else:
                predictions[h] = closes.copy() if closes is not None else np.zeros(len(X))

        return predictions

