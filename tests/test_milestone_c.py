"""
Unit tests for Apify X ingestion, schema mapping, and resilience.
"""

from datetime import datetime, timezone
import pytest
from src.ingestion.apify_x import (
    verify_apify_connection,
    map_raw_item_to_canonical,
    ingest_x_posts
)


def test_verify_apify_connection_no_keys():
    status = verify_apify_connection(None, None)
    assert not status.success
    assert "not configured" in status.message


def test_map_raw_item_to_canonical():
    mock_raw = {
        "id": "1849204820482",
        "text": "Bitcoin reaches $100k milestone! #BTC",
        "createdAt": "2026-03-01T12:00:00Z",
        "likeCount": 150,
        "retweetCount": 45,
        "viewCount": 25000,
        "author": {
            "followers": 12000
        },
        "url": "https://x.com/crypto/status/1849204820482",
        "lang": "en"
    }

    mapped = map_raw_item_to_canonical(mock_raw, run_id="test_run", asset="BTC")
    assert mapped["post_id"] == "1849204820482"
    assert "Bitcoin" in mapped["text"]
    assert mapped["likes"] == 150.0
    assert mapped["reposts"] == 45.0
    assert mapped["views"] == 25000.0
    assert mapped["author_followers"] == 12000.0
    assert mapped["source_platform"] == "x"
    assert mapped["asset"] == "BTC"
    assert mapped["collection_run_id"] == "test_run"


def test_map_raw_item_missing_fields_not_fabricated():
    # Only minimal fields
    mock_minimal = {
        "id": "999",
        "text": "Test tweet"
    }
    mapped = map_raw_item_to_canonical(mock_minimal, run_id="test_run", asset="BTC")
    assert mapped["post_id"] == "999"
    assert mapped["likes"] is None
    assert mapped["reposts"] is None
    assert mapped["views"] is None
    assert mapped["author_followers"] is None


def test_ingest_x_posts_demo_mode():
    df, summary = ingest_x_posts(use_demo=True, save_raw=True)
    assert not df.empty
    assert summary["total_posts"] > 0
    assert "field_availability_rates" in summary
    assert "likes" in summary["field_availability_rates"]
    assert df["source_platform"].iloc[0] == "x"


def test_apify_mask_token():
    from src.ingestion.apify import mask_token
    assert mask_token(None) == "NOT_CONFIGURED"
    assert mask_token("") == "NOT_CONFIGURED"
    assert mask_token("short") == "********"
    assert mask_token("apify_api_1234567890abcdef") == "apif...cdef"


def test_apify_live_test_output_format():
    import json
    from pathlib import Path
    live_test_file = Path("data/raw/x/btc_x_live_test.json")
    if live_test_file.exists():
        with open(live_test_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        assert len(items) <= 10
        if len(items) > 0:
            sample = items[0]
            assert "id" in sample
            assert "text" in sample or "fullText" in sample
            assert "createdAt" in sample


def test_generate_date_batches():
    from src.ingestion.historical_x import generate_date_batches
    batches = generate_date_batches(days_back=180, batch_size_days=7)
    assert len(batches) == 26
    # Each batch should be a tuple of (start_date, end_date) in YYYY-MM-DD
    for start_str, end_str in batches:
        assert len(start_str) == 10
        assert len(end_str) == 10
        assert start_str <= end_str


def test_estimate_historical_x_collection_cost():
    from src.ingestion.historical_x import estimate_historical_x_collection_cost
    est = estimate_historical_x_collection_cost(days_back=180, target_posts_per_day=25, batch_size_days=7)
    assert est["days_back"] == 180
    assert est["total_batches"] == 26
    assert est["estimated_total_items"] == 4500
    assert est["cost_breakdown"]["total_estimated_usd"] > 0
    assert est["apify_limits_assessment"]["within_free_budget"] is True


def test_map_raw_tweet_to_canonical():
    from src.ingestion.historical_x import map_raw_tweet_to_canonical
    raw_sample = {
        "id": "1234567890",
        "text": "Bitcoin bull run starting! $BTC",
        "createdAt": "2026-10-15T14:30:00Z",
        "author": {
            "userName": "crypto_trader",
            "followers": 5400
        },
        "likeCount": 120,
        "replyCount": 15,
        "retweetCount": 40,
        "quoteCount": 5,
        "url": "https://x.com/crypto_trader/status/1234567890"
    }

    mapped = map_raw_tweet_to_canonical(raw_sample, batch_label="test_batch")
    assert mapped["post_id"] == "1234567890"
    assert mapped["text"] == "Bitcoin bull run starting! $BTC"
    assert mapped["author_username"] == "crypto_trader"
    assert mapped["author_followers"] == 5400.0
    assert mapped["likes"] == 120.0
    assert mapped["replies"] == 15.0
    assert mapped["reposts"] == 40.0
    assert mapped["quotes"] == 5.0
    assert mapped["source_platform"] == "x"
    assert mapped["asset"] == "BTC"
    assert mapped["collection_batch"] == "test_batch"


def test_dry_run_historical_collector():
    from src.ingestion.historical_x import HistoricalXCollector
    collector = HistoricalXCollector()
    manifest = collector.run_collection(days_back=30, batch_size_days=7, dry_run=True)
    assert manifest["dry_run"] is True
    assert manifest["total_batches_planned"] == 5
    assert "cost_estimation" in manifest


def test_historical_supplementary_dataset_persistence():
    from pathlib import Path
    import json
    import pandas as pd

    raw_dir = Path("data/raw/x")
    parquet_path = raw_dir / "btc_x_historical_180d.parquet"
    csv_path = raw_dir / "btc_x_historical_180d.csv"
    manifest_path = raw_dir / "historical_collection_manifest.json"

    assert parquet_path.exists(), "btc_x_historical_180d.parquet must exist"
    assert csv_path.exists(), "btc_x_historical_180d.csv must exist"
    assert manifest_path.exists(), "historical_collection_manifest.json must exist"

    df = pd.read_parquet(parquet_path)
    assert len(df) > 4000
    expected_cols = ["post_id", "text", "created_at_utc", "author_username", "likes", "replies", "reposts", "source_platform", "asset"]
    for c in expected_cols:
        assert c in df.columns
    assert df["source_platform"].iloc[0] == "x"
    assert df["asset"].iloc[0] == "BTC"
    assert df["created_at_utc"].dt.tz is not None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["asset"] == "BTC"
    assert manifest["total_posts_retained"] == len(df)
    assert "platform_distinction_notes" in manifest




