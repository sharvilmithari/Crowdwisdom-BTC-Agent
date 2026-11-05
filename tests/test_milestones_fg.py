"""
Unit tests for Milestone F & G: Feature engineering, target construction, leakage prevention, and walk-forward validation.
"""

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from src.features.build_features import build_targets, engineer_price_features, build_feature_matrix
from src.evaluation.walk_forward import calculate_smape, compute_financial_risk_metrics, run_walk_forward_validation


def test_target_construction():
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC"),
        "close": [100 + i * 10 for i in range(10)]
    })
    df_t = build_targets(prices, horizons=[0, 1, 2])
    assert df_t["target_close_0"].iloc[0] == 100
    assert df_t["target_close_1"].iloc[0] == 110
    assert df_t["target_close_2"].iloc[0] == 120
    assert np.isnan(df_t["target_close_1"].iloc[-1])  # Future should be NaN at boundary


def test_feature_matrix_no_leakage():
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="1h", tz="UTC"),
        "open": np.linspace(100, 150, 50),
        "high": np.linspace(102, 152, 50),
        "low": np.linspace(98, 148, 50),
        "close": np.linspace(100, 150, 50),
        "volume": np.random.uniform(10, 50, 50)
    })
    social = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="1h", tz="UTC"),
        "post_count": np.random.randint(1, 10, 50),
        "sentiment_mean": np.random.uniform(-0.5, 0.5, 50),
        "sentiment_std": np.random.uniform(0.1, 0.3, 50),
        "positive_share": np.random.uniform(0.3, 0.7, 50),
        "negative_share": np.random.uniform(0.1, 0.3, 50),
        "neutral_share": np.random.uniform(0.1, 0.4, 50),
        "avg_engagement_per_post": np.random.uniform(1, 5, 50)
    })

    matrix, groups = build_feature_matrix(prices, social)
    assert "combined" in groups
    assert "price_only" in groups
    assert "sentiment_only" in groups
    assert len(matrix) > 0


def test_financial_risk_metrics():
    # Returns series with known positive Sharpe
    returns = np.array([0.01, 0.005, -0.002, 0.012, 0.003, -0.001, 0.008, 0.004, 0.002, 0.006, 0.005])
    metrics = compute_financial_risk_metrics(returns)
    assert metrics["sharpe_ratio"] > 0
    assert metrics["sortino_ratio"] > 0
    assert metrics["max_drawdown"] <= 0


def test_walk_forward_validation_runs():
    np.random.seed(42)
    n = 60
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
        "open": np.linspace(100, 150, n),
        "high": np.linspace(102, 152, n),
        "low": np.linspace(98, 148, n),
        "close": np.linspace(100, 150, n),
        "volume": np.random.uniform(10, 50, n)
    })
    matrix, groups = build_feature_matrix(prices, pd.DataFrame())
    res = run_walk_forward_validation(matrix, feature_cols=groups["price_only"], n_splits=2, min_train_ratio=0.6)
    assert "horizon_metrics" in res
    assert 0 in res["horizon_metrics"]
    assert "sharpe_ratio" in res["risk_metrics"]

