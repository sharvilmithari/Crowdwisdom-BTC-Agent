"""
Generator for realistic synthetic/demo data to enable offline end-to-end testing,
saving API calls and preventing quota exhaustion.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import pandas as pd
import numpy as np


def generate_demo_posts(n_posts: int = 500, days_back: int = 30) -> pd.DataFrame:
    """Generates canonical X social post records adhering to specification."""
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=days_back)
    
    sample_texts = [
        "Bitcoin breaking out past resistance! Extremely bullish on $BTC! 🚀",
        "BTC looking weak here, might retest support. Caution advised.",
        "Just accumulated more #Bitcoin for the long term HODL.",
        "Crypto market dump incoming? Seeing bearish divergence on BTC.",
        "Massive volume on BTC-USD hourly chart. What are your targets?",
        "Bitcoin adoption continues to accelerate globally.",
        "Stop loss triggered on BTC short. Bulls are in control.",
        "Consolidation day for $BTC. Neutral until breakout.",
        "Remember to self custody your Bitcoin keys!",
        "Institutional inflows into BTC hitting record highs this week."
    ]

    records = []
    for i in range(n_posts):
        post_time = start_time + timedelta(
            seconds=random.randint(0, int((now - start_time).total_seconds()))
        )
        post_id = f"demo_post_{100000 + i}"
        has_views = random.random() > 0.3
        has_followers = random.random() > 0.1
        
        records.append({
            "post_id": post_id,
            "text": random.choice(sample_texts),
            "created_at_utc": post_time.isoformat(),
            "likes": random.randint(0, 250),
            "reposts": random.randint(0, 80),
            "views": random.randint(500, 15000) if has_views else None,
            "author_followers": random.randint(50, 50000) if has_followers else None,
            "source_platform": "x",
            "asset": "BTC",
            "post_url": f"https://x.com/demo_user_{i % 20}/status/{post_id}",
            "language": "en",
            "collection_run_id": "demo_run_001"
        })

    df = pd.DataFrame(records)
    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True)
    df = df.sort_values("created_at_utc").reset_index(drop=True)
    return df


def generate_demo_prices(days_back: int = 30, interval_hours: int = 1) -> pd.DataFrame:
    """Generates synthetic BTC-USD OHLCV time series data."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    total_hours = days_back * 24
    
    timestamps = [now - timedelta(hours=total_hours - i) for i in range(total_hours + 1)]
    
    # Random walk for realistic price
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.008, len(timestamps))
    base_price = 60000.0
    price_levels = base_price * np.exp(np.cumsum(returns))
    
    records = []
    for ts, close in zip(timestamps, price_levels):
        high = close * (1.0 + abs(np.random.normal(0, 0.004)))
        low = close * (1.0 - abs(np.random.normal(0, 0.004)))
        open_price = (high + low) / 2.0
        volume = float(np.random.lognormal(mean=5.0, sigma=1.0) * 100)
        
        records.append({
            "timestamp": ts,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": round(volume, 4),
            "asset": "BTC-USD"
        })
        
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

