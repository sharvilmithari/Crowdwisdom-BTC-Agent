from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("feature_engineering")


def build_targets(df_prices: pd.DataFrame, horizons: List[int] = [0, 1, 2, 3, 4, 5]) -> pd.DataFrame:
    """Builds future close targets for n+0..n+5.
    - n+0 is defined as the current bar close (nowcast).
    - n+1..n+5 are subsequent future bar closes.
    Also creates target percentage returns: target_ret_n = (target_close_n / close) - 1
    """
    df = df_prices.copy().sort_values("timestamp").reset_index(drop=True)

    for h in horizons:
        if h == 0:
            df["target_close_0"] = df["close"]
            df["target_ret_0"] = 0.0
        else:
            df[f"target_close_{h}"] = df["close"].shift(-h)
            df[f"target_ret_{h}"] = (df[f"target_close_{h}"] / df["close"]) - 1.0

    return df


def engineer_price_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates price-derived features strictly without future leakage."""
    df = df_prices.copy().sort_values("timestamp").reset_index(drop=True)

    # Log returns
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_ret_6"] = np.log(df["close"] / df["close"].shift(6))
    df["log_ret_12"] = np.log(df["close"] / df["close"].shift(12))
    df["log_ret_24"] = np.log(df["close"] / df["close"].shift(24))

    # Rolling Volatility (standard deviation of returns)
    df["volatility_6h"] = df["log_ret_1"].rolling(6).std()
    df["volatility_24h"] = df["log_ret_1"].rolling(24).std()

    # Moving average ratios
    sma_6 = df["close"].rolling(6).mean()
    sma_24 = df["close"].rolling(24).mean()
    df["sma_ratio_6_24"] = sma_6 / sma_24

    # High-Low Range
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]

    # Volume momentum
    vol_mean_24 = df["volume"].rolling(24).mean()
    df["vol_ratio"] = df["volume"] / (vol_mean_24 + 1e-6)

    return df


def engineer_sentiment_features(df_social_agg: pd.DataFrame) -> pd.DataFrame:
    """Calculates social features and lags."""
    df = df_social_agg.copy().sort_values("timestamp").reset_index(drop=True)

    # Lags for sentiment score
    df["sentiment_lag_1"] = df["sentiment_mean"].shift(1)
    df["sentiment_lag_2"] = df["sentiment_mean"].shift(2)
    df["sentiment_lag_3"] = df["sentiment_mean"].shift(3)

    # Rolling sentiment trends
    df["sentiment_roll_mean_6"] = df["sentiment_mean"].rolling(6, min_periods=1).mean()
    df["sentiment_roll_std_6"] = df["sentiment_mean"].rolling(6, min_periods=1).std().fillna(0.0)

    # Social volume momentum
    post_roll_6 = df["post_count"].rolling(6, min_periods=1).mean()
    df["social_volume_ratio"] = df["post_count"] / (post_roll_6 + 1e-6)

    return df


def build_feature_matrix(
    df_prices: pd.DataFrame,
    df_social_agg: pd.DataFrame,
    horizons: List[int] = [0, 1, 2, 3, 4, 5]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Merges price bars and aggregated social sentiment on timestamps.
    Returns:
    - Merged DataFrame
    - Dict with feature subsets: 'price_only', 'sentiment_only', 'combined'
    """
    df_p = engineer_price_features(df_prices)
    df_t = build_targets(df_p, horizons=horizons)

    if not df_social_agg.empty:
        df_s = engineer_sentiment_features(df_social_agg)
        # Left join on price timestamps
        merged = pd.merge(df_t, df_s, on="timestamp", how="left", suffixes=("", "_social"))
        # Missing social data handling: distinguish explicit zero posts
        merged["post_count"] = merged["post_count"].fillna(0)
        merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
        merged["sentiment_dispersion"] = merged["sentiment_std"].fillna(0.0)
        merged["positive_share"] = merged["positive_share"].fillna(0.0)
        merged["negative_share"] = merged["negative_share"].fillna(0.0)
        merged["neutral_share"] = merged["neutral_share"].fillna(1.0)
        merged["avg_engagement_per_post"] = merged["avg_engagement_per_post"].fillna(0.0)
        merged["sentiment_roll_mean_6"] = merged["sentiment_roll_mean_6"].fillna(0.0)
        merged["sentiment_lag_1"] = merged["sentiment_lag_1"].fillna(0.0)
    else:
        merged = df_t.copy()
        for col in ["post_count", "sentiment_mean", "sentiment_dispersion", "positive_share", "negative_share", "neutral_share", "avg_engagement_per_post", "sentiment_roll_mean_6", "sentiment_lag_1"]:
            merged[col] = 0.0

    price_features = [
        "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12", "log_ret_24",
        "volatility_6h", "volatility_24h", "sma_ratio_6_24", "hl_spread", "vol_ratio"
    ]

    sentiment_features = [
        "post_count", "sentiment_mean", "sentiment_dispersion",
        "positive_share", "negative_share", "neutral_share",
        "avg_engagement_per_post", "sentiment_roll_mean_6", "sentiment_lag_1"
    ]

    feature_groups = {
        "price_only": price_features,
        "sentiment_only": sentiment_features,
        "combined": price_features + sentiment_features
    }

    # Drop early rows with NaN from rolling calculations
    valid = merged.dropna(subset=price_features).reset_index(drop=True)
    return valid, feature_groups

