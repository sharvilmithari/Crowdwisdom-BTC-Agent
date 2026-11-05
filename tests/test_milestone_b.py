"""
Unit tests for price ingestion, data quality reporting, and provider fallback.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from src.ingestion.prices import (
    DemoPriceProvider,
    validate_and_clean_prices,
    get_price_dataset,
    ingest_btc_prices,
    parse_interval_to_timedelta
)


def test_demo_price_provider():
    provider = DemoPriceProvider()
    df = provider.fetch_ohlcv("BTC-USD", interval="1h")
    assert not df.empty
    assert "close" in df.columns
    assert "timestamp" in df.columns
    assert df["asset"].iloc[0] == "BTC-USD"


def test_validate_and_clean_prices_duplicates_and_monotonic():
    now = datetime.now(timezone.utc)
    # Create dataset with intentional duplicate and unordered timestamp
    data = {
        "timestamp": [
            now - timedelta(hours=3),
            now - timedelta(hours=1),
            now - timedelta(hours=2),
            now - timedelta(hours=2),  # duplicate
        ],
        "open": [100.0, 103.0, 101.0, 102.0],
        "high": [105.0, 106.0, 104.0, 104.0],
        "low": [99.0, 100.0, 100.0, 100.0],
        "close": [102.0, 105.0, 102.0, 103.0],
        "volume": [10.0, 20.0, 15.0, 15.0],
        "asset": ["BTC-USD"] * 4
    }
    raw_df = pd.DataFrame(data)
    cleaned, report = validate_and_clean_prices(raw_df, expected_interval="1h")

    assert len(cleaned) == 3
    assert report.duplicate_timestamps == 1
    assert report.is_monotonic is True
    assert cleaned["timestamp"].is_monotonic_increasing


def test_price_gap_detection():
    now = datetime.now(timezone.utc)
    # Consecutive bars with a 5-hour gap
    data = {
        "timestamp": [
            now - timedelta(hours=10),
            now - timedelta(hours=9),
            now - timedelta(hours=2),  # 7-hour gap
            now - timedelta(hours=1),
        ],
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [105.0, 105.0, 105.0, 105.0],
        "low": [99.0, 99.0, 99.0, 99.0],
        "close": [101.0, 102.0, 103.0, 104.0],
        "volume": [10.0, 10.0, 10.0, 10.0],
        "asset": ["BTC-USD"] * 4
    }
    raw_df = pd.DataFrame(data)
    cleaned, report = validate_and_clean_prices(raw_df, expected_interval="1h")

    assert report.detected_gaps == 1
    assert report.total_bars == 4


def test_get_price_dataset_end_to_end(tmp_path):
    df, report = get_price_dataset(provider_type="demo", interval="1h", save_raw=True)
    assert not df.empty
    assert report.total_bars > 0
    assert report.is_monotonic


def test_parse_interval_to_timedelta():
    assert parse_interval_to_timedelta("1h") == timedelta(hours=1)
    assert parse_interval_to_timedelta("2h") == timedelta(hours=2)
    assert parse_interval_to_timedelta("1d") == timedelta(days=1)
    assert parse_interval_to_timedelta("30m") == timedelta(minutes=30)


def test_ingest_btc_prices_demo(tmp_path):
    csv_file = tmp_path / "btc_usd_1h.csv"
    parquet_file = tmp_path / "btc_prices_1h.parquet"

    df, report = ingest_btc_prices(
        symbol="BTC-USD",
        days_back=7,
        interval="1h",
        output_csv_path=csv_file,
        output_parquet_path=parquet_file,
        provider_type="demo",
        save_raw=True
    )

    assert not df.empty
    assert set(["timestamp", "open", "high", "low", "close", "volume", "asset"]).issubset(df.columns)
    assert df["timestamp"].dt.tz is not None
    assert report.is_monotonic is True
    assert report.total_bars == len(df)
    assert report.frequency == "1h"
    assert report.coverage_pct > 0.0

    assert csv_file.exists()
    assert parquet_file.exists()


def test_no_fabrication_of_missing_observations():
    # If there is a 3-hour gap, ensure we do NOT insert synthetic rows
    now = datetime.now(timezone.utc)
    data = {
        "timestamp": [now - timedelta(hours=5), now - timedelta(hours=1)],
        "open": [100.0, 105.0],
        "high": [102.0, 107.0],
        "low": [99.0, 104.0],
        "close": [101.0, 106.0],
        "volume": [10.0, 20.0],
        "asset": ["BTC-USD", "BTC-USD"]
    }
    raw_df = pd.DataFrame(data)
    cleaned, report = validate_and_clean_prices(raw_df, expected_interval="1h")

    # Length must remain 2 (no fabricated rows)
    assert len(cleaned) == 2
    assert report.detected_gaps == 1
    assert len(report.gap_details) == 1
    assert report.gap_details[0]["missing_bars"] == 3


def test_saved_btc_usd_1h_csv():
    from pathlib import Path
    csv_path = Path("data/raw/prices/btc_usd_1h.csv")
    assert csv_path.exists(), "btc_usd_1h.csv should exist after running ingestion"
    df = pd.read_csv(csv_path)
    assert not df.empty
    assert len(df) >= 100  # 180 days has ~4300 rows
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in expected_cols:
        assert col in df.columns
    # Check UTC string format
    ts_sample = str(df["timestamp"].iloc[0])
    assert "+00:00" in ts_sample or "Z" in ts_sample or "UTC" in ts_sample


