"""
Post Preprocessing & Interval Aggregation Module.
- Validates canonical post schema.
- Cleans and deduplicates posts.
- Runs sentiment extraction.
- Aggregates features to hourly/configurable bars aligned with price bars.
- Explicitly handles missing social intervals (avoids filling missing periods with fake zero sentiment).
"""

from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.sentiment.finbert import get_sentiment_analyzer, BaseSentimentAnalyzer
from src.utils.logging_config import setup_logger

logger = setup_logger("social_aggregation")


def clean_posts(df_posts: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filters empty texts, removes exact duplicate post_ids, and ensures UTC timestamps."""
    initial_count = len(df_posts)
    if initial_count == 0:
        return df_posts, {"initial": 0, "empty_removed": 0, "duplicate_removed": 0, "retained": 0}

    df = df_posts.copy()

    # Drop empty or whitespace text
    df = df[df["text"].notnull() & (df["text"].str.strip() != "")]
    empty_removed = initial_count - len(df)

    # Drop duplicate post IDs
    pre_dup_count = len(df)
    df = df.drop_duplicates(subset=["post_id"], keep="first")
    duplicate_removed = pre_dup_count - len(df)

    # Ensure UTC timezone
    if df["created_at_utc"].dt.tz is None:
        df["created_at_utc"] = df["created_at_utc"].dt.tz_localize(timezone.utc)
    else:
        df["created_at_utc"] = df["created_at_utc"].dt.tz_convert(timezone.utc)

    stats = {
        "initial": initial_count,
        "empty_removed": empty_removed,
        "duplicate_removed": duplicate_removed,
        "retained": len(df)
    }
    return df, stats


def enrich_posts_with_sentiment(
    df_posts: pd.DataFrame,
    analyzer: BaseSentimentAnalyzer = None,
    batch_size: int = 128
) -> pd.DataFrame:
    """Enriches post records with sentiment probabilities, score, and label."""
    if df_posts.empty:
        return df_posts

    if analyzer is None:
        analyzer = get_sentiment_analyzer(prefer_finbert=False)

    df = df_posts.copy()
    texts = df["text"].tolist()
    sentiment_records = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = analyzer.analyze_batch(batch)
        sentiment_records.extend(results)

    sent_df = pd.DataFrame(sentiment_records)
    for col in sent_df.columns:
        df[col] = sent_df[col].values

    return df


def aggregate_social_features(
    df_enriched_posts: pd.DataFrame,
    freq: str = "1h"
) -> pd.DataFrame:
    """Aggregates social features by interval (e.g., 1h).
    Calculates:
    - post_count: int
    - sentiment_mean: float
    - sentiment_std: float (dispersion)
    - positive_share: float
    - negative_share: float
    - neutral_share: float
    - sum_likes, sum_reposts, sum_views: float
    - avg_engagement_per_post: float
    """
    if df_enriched_posts.empty:
        return pd.DataFrame()

    df = df_enriched_posts.copy()
    # Floor to interval bar
    df["bar_timestamp"] = df["created_at_utc"].dt.floor(freq)

    grouped = df.groupby("bar_timestamp")

    agg_df = pd.DataFrame()
    agg_df["post_count"] = grouped["post_id"].count()
    agg_df["sentiment_mean"] = grouped["sentiment_score"].mean()
    agg_df["sentiment_median"] = grouped["sentiment_score"].median()
    agg_df["sentiment_std"] = grouped["sentiment_score"].std().fillna(0.0)

    # Share calculations
    agg_df["positive_share"] = grouped["sentiment_label"].apply(lambda s: (s == "positive").mean())
    agg_df["negative_share"] = grouped["sentiment_label"].apply(lambda s: (s == "negative").mean())
    agg_df["neutral_share"] = grouped["sentiment_label"].apply(lambda s: (s == "neutral").mean())

    # Engagement aggregations
    agg_df["sum_likes"] = grouped["likes"].sum(min_count=1) if "likes" in df.columns else 0.0
    agg_df["sum_reposts"] = grouped["reposts"].sum(min_count=1) if "reposts" in df.columns else 0.0
    agg_df["sum_views"] = grouped["views"].sum(min_count=1) if "views" in df.columns else 0.0
    
    # Combined engagement
    likes = agg_df["sum_likes"].fillna(0)
    reposts = agg_df["sum_reposts"].fillna(0)
    agg_df["avg_engagement_per_post"] = (likes + reposts) / agg_df["post_count"]

    # Model metadata tracking
    agg_df["sentiment_model"] = grouped["model_name"].first()
    agg_df["is_fallback"] = grouped["is_fallback"].first()

    agg_df = agg_df.reset_index().rename(columns={"bar_timestamp": "timestamp"})
    return agg_df

