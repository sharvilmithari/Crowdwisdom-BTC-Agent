"""
Unit tests for news ingestion, URL canonicalization, deduplication, and connection status.
"""

from src.ingestion.news_tavily import (
    canonicalize_url,
    normalize_title,
    deduplicate_news,
    verify_tavily_connection,
    verify_exa_connection,
    ingest_news
)


def test_canonicalize_url():
    url_with_params = "https://www.coindesk.com/markets/2026/03/btc-update/?utm_source=twitter&ref=antigravity#comments"
    clean = canonicalize_url(url_with_params)
    assert clean == "https://www.coindesk.com/markets/2026/03/btc-update"


def test_normalize_title():
    t1 = "Bitcoin Crosses $100K: What's Next?!"
    t2 = "bitcoin crosses 100k whats next"
    assert normalize_title(t1) == normalize_title(t2)


def test_deduplicate_news():
    items = [
        {"title": "BTC Surges to Record", "url": "https://news.com/btc?source=1", "provider": "test"},
        {"title": "btc surges to record!!", "url": "https://news.com/btc?source=2", "provider": "test"},
        {"title": "Different Article", "url": "https://news.com/different", "provider": "test"}
    ]
    unique = deduplicate_news(items)
    assert len(unique) == 2
    assert unique[0]["title"] == "BTC Surges to Record"
    assert unique[1]["title"] == "Different Article"


def test_verify_connections_unconfigured():
    t_status = verify_tavily_connection(None)
    assert not t_status.success
    assert "not configured" in t_status.message

    e_status = verify_exa_connection("")
    assert not e_status.success
    assert "not configured" in e_status.message


def test_ingest_news_demo_mode():
    news, meta = ingest_news(use_demo=True, save_raw=True)
    assert len(news) > 0
    assert meta["total_unique"] == len(news)
    assert "provider" in news[0]
    assert "News context is strictly isolated" in meta["note"]

