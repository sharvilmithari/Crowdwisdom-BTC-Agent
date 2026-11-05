"""
News Ingestion Module (Tavily & Exa).
- Collects recent BTC news context.
- Implements URL canonicalization & near-duplicate headline deduplication.
- Preserves source attribution and publication timestamps.
- Separates news strictly from the ML feature matrix (used for qualitative context only).
- Provides offline demo/mock mode for zero-cost operation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse, urlunparse

from config.settings import AppConfig, get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("news_ingestion")


class NewsConnectionStatus:
    def __init__(self, service: str, success: bool, message: str):
        self.service = service
        self.success = success
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "success": self.success,
            "message": self.message
        }


def canonicalize_url(url: str) -> str:
    """Normalizes URL by stripping query parameters and tracking fragments."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        # Reconstruct without query parameters or fragment
        clean = urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip('/'), '', '', ''))
        return clean
    except Exception:
        return url.strip()


def normalize_title(title: str) -> str:
    """Simplifies headline text for duplicate detection."""
    if not title:
        return ""
    # Lowercase and remove punctuation
    clean = re.sub(r'[^\w\s]', '', title.lower()).strip()
    return " ".join(clean.split())


def deduplicate_news(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates articles by canonical URL and simplified title."""
    seen_urls = set()
    seen_titles = set()
    unique_articles = []

    for art in articles:
        canon_url = canonicalize_url(art.get("url", ""))
        clean_title = normalize_title(art.get("title", ""))

        if canon_url and canon_url in seen_urls:
            continue
        if clean_title and clean_title in seen_titles:
            continue

        if canon_url:
            seen_urls.add(canon_url)
        if clean_title:
            seen_titles.add(clean_title)

        unique_articles.append(art)

    return unique_articles


def verify_tavily_connection(api_key: Optional[str]) -> NewsConnectionStatus:
    """Safely verifies Tavily API credentials without logging secrets."""
    if not api_key or not api_key.strip():
        return NewsConnectionStatus("Tavily", False, "Tavily API key is not configured.")

    try:
        import requests
        headers = {"Content-Type": "application/json"}
        # Minimal search to verify key
        payload = {
            "api_key": api_key.strip(),
            "query": "Bitcoin",
            "max_results": 1,
            "search_depth": "basic"
        }
        resp = requests.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            return NewsConnectionStatus("Tavily", True, "Tavily API connection successful.")
        else:
            return NewsConnectionStatus("Tavily", False, f"Tavily returned HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        return NewsConnectionStatus("Tavily", False, f"Connection failed: {str(e)}")


def verify_exa_connection(api_key: Optional[str]) -> NewsConnectionStatus:
    """Safely verifies Exa API credentials without logging secrets."""
    if not api_key or not api_key.strip():
        return NewsConnectionStatus("Exa", False, "Exa API key is not configured.")

    try:
        import requests
        headers = {
            "x-api-key": api_key.strip(),
            "Content-Type": "application/json"
        }
        payload = {"query": "Bitcoin", "numResults": 1}
        resp = requests.post("https://api.exa.ai/search", json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            return NewsConnectionStatus("Exa", True, "Exa API connection successful.")
        else:
            return NewsConnectionStatus("Exa", False, f"Exa returned HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        return NewsConnectionStatus("Exa", False, f"Connection failed: {str(e)}")


def fetch_tavily_news(
    query: str = "Bitcoin BTC news",
    max_results: int = 10,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetches Bitcoin news via Tavily."""
    import requests
    retrieval_time = datetime.now(timezone.utc).isoformat()
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "news",
        "include_answer": False
    }

    resp = requests.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source": item.get("source") or urlparse(item.get("url", "")).netloc,
            "snippet": item.get("content", ""),
            "published_date": item.get("published_date"),
            "retrieval_timestamp": retrieval_time,
            "provider": "tavily"
        })
    return results


def generate_demo_news(n_items: int = 6) -> List[Dict[str, Any]]:
    """Generates synthetic contextual news items for offline execution."""
    retrieval_time = datetime.now(timezone.utc).isoformat()
    headlines = [
        ("SEC Approves Next Phase of Digital Asset Regulations", "Bloomberg", "https://bloomberg.com/crypto/sec-update"),
        ("Bitcoin Hashrate Hits All-Time High Amid Miner Efficiency Gains", "CoinDesk", "https://coindesk.com/markets/hashrate-record"),
        ("Global Institutional Fund Inflows Surge in BTC Spot ETFs", "Reuters", "https://reuters.com/business/finance/crypto-etf-inflows"),
        ("Federal Reserve Holds Interest Rates Steady, Crypto Markets Consolidate", "CNBC", "https://cnbc.com/2026/fed-rates-crypto"),
        ("On-Chain Metrics Indicate Strong Accumulation by Long-Term Holders", "Glassnode Insights", "https://insights.glassnode.com/btc-accumulation"),
        ("Major Tech Conglomerate Explores Lightning Network Integration", "TechCrunch", "https://techcrunch.com/fintech/lightning-network")
    ]

    articles = []
    for title, source, url in headlines[:n_items]:
        articles.append({
            "title": title,
            "url": url,
            "source": source,
            "snippet": f"Summary: Market observers note that {title.lower()} is driving renewed interest across global crypto markets.",
            "published_date": retrieval_time,
            "retrieval_timestamp": retrieval_time,
            "provider": "demo"
        })
    return articles


def ingest_news(
    query: str = "Bitcoin BTC news",
    max_results: int = 10,
    use_demo: bool = False,
    save_raw: bool = True,
    config: Optional[AppConfig] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Main pipeline function for news ingestion with deduplication and persistence."""
    cfg = config or get_config()

    if use_demo or not cfg.tavily_api_key:
        logger.info("Ingesting news using offline demo provider.")
        raw_news = generate_demo_news(n_items=min(max_results, 6))
    else:
        logger.info(f"Ingesting news via Tavily for query: '{query}'")
        try:
            raw_news = fetch_tavily_news(query=query, max_results=max_results, api_key=cfg.tavily_api_key)
        except Exception as e:
            logger.warning(f"Tavily query failed: {e}. Falling back to demo news context.")
            raw_news = generate_demo_news(n_items=min(max_results, 6))

    deduped_news = deduplicate_news(raw_news)

    if save_raw and deduped_news:
        cfg.raw_news_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        news_file = cfg.raw_news_dir / f"news_{timestamp_str}.json"
        with open(news_file, "w", encoding="utf-8") as f:
            json.dump(deduped_news, f, indent=2)
        logger.info(f"Saved {len(deduped_news)} news items to {news_file}")

    metadata = {
        "total_retrieved": len(raw_news),
        "total_unique": len(deduped_news),
        "deduplicated_count": len(raw_news) - len(deduped_news),
        "query": query,
        "note": "News context is strictly isolated from quantitative ML feature matrix."
    }

    return deduped_news, metadata

