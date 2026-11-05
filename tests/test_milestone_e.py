"""
Unit tests for Milestone E: preprocessing, sentiment engine, and interval aggregation.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
from src.sentiment.finbert import LexiconFallbackAnalyzer
from src.sentiment.aggregate import clean_posts, enrich_posts_with_sentiment, aggregate_social_features


def test_lexicon_fallback_analyzer():
    analyzer = LexiconFallbackAnalyzer()
    texts = [
        "Extremely bullish on Bitcoin, ready for massive rally!",
        "Bitcoin dump and crash incoming, terrible drop.",
        "Today is a regular day for financial markets."
    ]
    results = analyzer.analyze_batch(texts)
    assert len(results) == 3
    assert results[0]["sentiment_label"] == "positive"
    assert results[0]["sentiment_score"] > 0
    assert results[1]["sentiment_label"] == "negative"
    assert results[1]["sentiment_score"] < 0
    assert results[2]["sentiment_label"] == "neutral"
    assert results[0]["is_fallback"] is True
    assert results[0]["model_name"] == "lexicon_fallback_v1"


def test_clean_posts_and_deduplication():
    now = datetime.now(timezone.utc)
    posts = pd.DataFrame([
        {"post_id": "p1", "text": "Valid tweet", "created_at_utc": now},
        {"post_id": "p2", "text": "   ", "created_at_utc": now},  # empty
        {"post_id": "p1", "text": "Duplicate tweet", "created_at_utc": now},  # duplicate id
    ])
    cleaned, stats = clean_posts(posts)
    assert len(cleaned) == 1
    assert stats["empty_removed"] == 1
    assert stats["duplicate_removed"] == 1


def test_aggregation_pipeline():
    now = datetime(2026, 3, 1, 10, 15, tzinfo=timezone.utc)
    posts = pd.DataFrame([
        {"post_id": "p1", "text": "Bullish rally #BTC", "created_at_utc": now, "likes": 10, "reposts": 2, "views": 100},
        {"post_id": "p2", "text": "More gains coming!", "created_at_utc": now + timedelta(minutes=10), "likes": 5, "reposts": 1, "views": 50},
        {"post_id": "p3", "text": "Crash and dump", "created_at_utc": now + timedelta(hours=1, minutes=5), "likes": 0, "reposts": 0, "views": 10}
    ])

    enriched = enrich_posts_with_sentiment(posts)
    agg = aggregate_social_features(enriched, freq="1h")
    
    assert len(agg) == 2
    # First hour has 2 posts
    assert agg["post_count"].iloc[0] == 2
    assert agg["sentiment_mean"].iloc[0] > 0
    assert agg["sum_likes"].iloc[0] == 15
    # Second hour has 1 post
    assert agg["post_count"].iloc[1] == 1
    assert agg["sentiment_mean"].iloc[1] < 0

