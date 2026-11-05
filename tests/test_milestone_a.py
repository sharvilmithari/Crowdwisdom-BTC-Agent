"""
Unit tests for configuration, logging redaction, and demo data synthesis.
"""

import logging
from config.settings import AppConfig, get_config
from src.utils.logging_config import setup_logger, SecretRedactingFormatter
from src.utils.demo_data import generate_demo_posts, generate_demo_prices


def test_config_defaults():
    cfg = get_config()
    assert cfg.asset_symbol == "BTC-USD"
    assert "Bitcoin" in cfg.social_queries
    assert cfg.horizons == [0, 1, 2, 3, 4, 5]
    assert cfg.mask_secret("apify_api_1234567890abcdef") == "apif...cdef"
    assert cfg.mask_secret(None) == "NOT CONFIGURED"


def test_secret_redaction_in_logs():
    formatter = SecretRedactingFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Connecting with apify_api_token: 'secret_token_12345678' and key tvly-9999888877776666",
        args=(),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "secret_token_12345678" not in formatted
    assert "tvly-9999888877776666" not in formatted


def test_demo_post_generation():
    posts = generate_demo_posts(n_posts=50, days_back=7)
    assert len(posts) == 50
    assert "post_id" in posts.columns
    assert "text" in posts.columns
    assert "created_at_utc" in posts.columns
    assert posts["source_platform"].iloc[0] == "x"
    assert posts["created_at_utc"].dt.tz is not None  # Must be timezone-aware


def test_demo_price_generation():
    prices = generate_demo_prices(days_back=5, interval_hours=1)
    assert len(prices) > 0
    assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(prices.columns)
    assert prices["timestamp"].dt.tz is not None

