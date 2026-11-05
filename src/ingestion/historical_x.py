"""
Historical X (Twitter) Data Collection Module.
- Multi-batch date-windowed collection for 180 days of Bitcoin posts.
- Actor: 'apidojo/tweet-scraper'.
- Pre-collection cost and resource estimation.
- Ingestion of supplementary open historical Bitcoin X datasets.
- Canonical mapping preserving tweet text, ID, timestamp, author username, likes,
  replies, reposts, quotes, and follower counts without fabricating data.
- Manifest and coverage tracking.
- Clear separation between X public social data and StockTwits investor data.
"""

from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd

from config.settings import AppConfig, get_config
from src.utils.logging_config import setup_logger
from src.ingestion.apify import get_apify_credentials, mask_token

logger = setup_logger("historical_x_collection")


def estimate_historical_x_collection_cost(
    days_back: int = 180,
    target_posts_per_day: int = 25,
    batch_size_days: int = 7,
    actor_id: str = "apidojo/tweet-scraper"
) -> Dict[str, Any]:
    """Estimates the Apify monetary and compute cost before launching a collection run.
    
    Actor pricing parameters (from Apify Store pay-per-event model):
    - apify-default-dataset-item: $0.0004 per tweet
    - Compute unit rate: ~$0.20 per CU
    - Average compute duration per batch: ~15 seconds @ 4GB RAM (~0.0167 CU ≈ $0.0033 per batch)
    """
    total_batches = (days_back + batch_size_days - 1) // batch_size_days
    estimated_items = days_back * target_posts_per_day
    
    # Cost components
    item_cost_usd = estimated_items * 0.0004
    estimated_cu_per_batch = 0.0167
    compute_cost_usd = total_batches * estimated_cu_per_batch * 0.20
    total_cost_usd = item_cost_usd + compute_cost_usd

    return {
        "days_back": days_back,
        "batch_size_days": batch_size_days,
        "total_batches": total_batches,
        "target_posts_per_day": target_posts_per_day,
        "estimated_total_items": estimated_items,
        "cost_breakdown": {
            "dataset_items_charge_usd": round(item_cost_usd, 4),
            "compute_units_charge_usd": round(compute_cost_usd, 4),
            "total_estimated_usd": round(total_cost_usd, 4)
        },
        "apify_limits_assessment": {
            "free_plan_monthly_credits_usd": 5.00,
            "within_free_budget": total_cost_usd <= 5.00,
            "free_tier_actor_restriction": (
                "Actor 'apidojo/tweet-scraper' limits runs on Apify Free tier accounts "
                "to 10 items per run. Unrestricted scrapes require an Apify Paid Subscription."
            ),
            "x_search_indexing_limitation": (
                "Twitter/X standard search API indexes tweets reliably for only the last 7-14 days. "
                "Queries covering 30-180 days ago yield sparse or zero results on standard search."
            )
        }
    }


def generate_date_batches(
    days_back: int = 180,
    batch_size_days: int = 7,
    end_date: Optional[datetime] = None
) -> List[Tuple[str, str]]:
    """Generates contiguous (start_date, end_date) tuples in 'YYYY-MM-DD' format."""
    end_dt = end_date or datetime.now(timezone.utc)
    batches = []

    current_end = end_dt
    for _ in range(0, days_back, batch_size_days):
        current_start = max(end_dt - timedelta(days=days_back), current_end - timedelta(days=batch_size_days))
        batches.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_end = current_start
        if current_end <= (end_dt - timedelta(days=days_back)):
            break

    # Return chronological order (oldest to newest)
    return list(reversed(batches))


def map_raw_tweet_to_canonical(item: Dict[str, Any], batch_label: str) -> Dict[str, Any]:
    """Extracts all required fields from raw Apify actor item without fabricating data."""
    raw_id = item.get("id") or item.get("id_str") or item.get("tweet_id")
    post_id = str(raw_id) if raw_id is not None else ""

    text = item.get("text") or item.get("fullText") or item.get("tweet") or ""

    # Timestamp normalization
    raw_ts = item.get("createdAt") or item.get("created_at") or item.get("timestamp")
    created_at_utc = None
    if raw_ts:
        try:
            created_at_utc = pd.to_datetime(raw_ts, utc=True).to_pydatetime()
        except Exception:
            created_at_utc = None

    # Author details
    author_obj = item.get("author") or item.get("user") or {}
    username = None
    followers = None
    if isinstance(author_obj, dict):
        username = author_obj.get("userName") or author_obj.get("screen_name") or author_obj.get("name")
        followers = author_obj.get("followers") or author_obj.get("followers_count") or author_obj.get("followersCount")

    # Metrics (strictly preserved as None if absent; never fabricated)
    likes = item.get("likeCount") or item.get("favorite_count")
    replies = item.get("replyCount") or item.get("reply_count")
    reposts = item.get("retweetCount") or item.get("retweet_count")
    quotes = item.get("quoteCount") or item.get("quote_count")

    return {
        "post_id": post_id,
        "text": str(text),
        "created_at_utc": created_at_utc,
        "author_username": str(username) if username else None,
        "author_followers": float(followers) if followers is not None else None,
        "likes": float(likes) if likes is not None else None,
        "replies": float(replies) if replies is not None else None,
        "reposts": float(reposts) if reposts is not None else None,
        "quotes": float(quotes) if quotes is not None else None,
        "source_platform": "x",
        "asset": "BTC",
        "post_url": item.get("url") or item.get("twitterUrl"),
        "collection_batch": batch_label
    }


def ingest_supplementary_historical_x(
    dataset_url: str = "https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/resolve/main/data/train.parquet",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    include_live_scraped: bool = True,
    config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ingests, cleans, and canonicalizes 180 days of historical Bitcoin X data from an open archive.
    
    Guarantees:
    - Never fabricates missing posts or engagement metrics.
    - Preserves existing test files (btc_x_test.json and btc_x_live_test.json).
    - Normalizes timestamps to UTC.
    - Removes duplicate post IDs.
    - Distinguishes X data from StockTwits data clearly in metadata and records.
    - Saves canonical dataset to data/raw/x/btc_x_historical_180d.parquet and .csv.
    - Records collection manifest at data/raw/x/historical_collection_manifest.json.
    """
    cfg = config or get_config()
    raw_dir = cfg.raw_x_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    start_dt = start_date or pd.to_datetime("2026-05-08 10:00:00", utc=True)
    end_dt = end_date or pd.to_datetime("2026-11-04 23:59:59", utc=True)

    logger.info(f"Loading supplementary historical X data from: {dataset_url}")
    df_raw = pd.read_parquet(dataset_url)
    logger.info(f"Loaded raw dataset with {len(df_raw)} records.")

    # Save raw archive snapshot for traceability
    raw_snapshot_path = raw_dir / "btc_x_historical_180d_raw.parquet"
    if not raw_snapshot_path.exists():
        df_raw.to_parquet(raw_snapshot_path, index=False)
        logger.info(f"Saved raw snapshot to {raw_snapshot_path}")

    # Standardize timestamp to UTC
    df_raw["created_at_utc"] = pd.to_datetime(df_raw["created_at"], utc=True)

    # Filter for Bitcoin-related terms and the 180-day window
    btc_filter = (
        df_raw["text"].str.contains(r"Bitcoin|BTC|#BTC|\$BTC", case=False, na=False) |
        df_raw["tickers"].astype(str).str.contains("BTC", case=False, na=False)
    )
    window_filter = (df_raw["created_at_utc"] >= start_dt) & (df_raw["created_at_utc"] <= end_dt)

    df_filtered = df_raw[btc_filter & window_filter].copy()
    logger.info(f"Found {len(df_filtered)} Bitcoin posts within the 180-day window.")

    # Canonical mapping
    canonical_rows = []
    for _, row in df_filtered.iterrows():
        pid = str(row.get("tweet_id", ""))
        txt = str(row.get("text", ""))
        ts = row.get("created_at_utc")
        user = row.get("user_screen_name") or row.get("user_name")
        likes = row.get("likes")
        replies = row.get("replies")
        reposts = row.get("retweets")
        quotes = row.get("quoted_tweet")
        url = row.get("url")

        canonical_rows.append({
            "post_id": pid,
            "text": txt,
            "created_at_utc": ts,
            "author_username": str(user) if pd.notnull(user) else None,
            "author_followers": None,  # Not present in this raw source, never fabricate
            "likes": float(likes) if pd.notnull(likes) else None,
            "replies": float(replies) if pd.notnull(replies) else None,
            "reposts": float(reposts) if pd.notnull(reposts) else None,
            "quotes": float(quotes) if pd.notnull(quotes) and isinstance(quotes, (int, float)) else None,
            "source_platform": "x",
            "asset": "BTC",
            "post_url": str(url) if pd.notnull(url) else None,
            "collection_batch": "open_historical_archive"
        })

    # Include recent live scraped tweets from today (Nov 4, 2026) if available
    live_test_file = raw_dir / "btc_x_live_test.json"
    if include_live_scraped and live_test_file.exists():
        with open(live_test_file, "r", encoding="utf-8") as f:
            live_items = json.load(f)
        for item in live_items:
            canonical_rows.append(map_raw_tweet_to_canonical(item, batch_label="apify_live_collection"))
        logger.info(f"Integrated {len(live_items)} live scraped tweets from {live_test_file}")

    # Build DataFrame
    df_canonical = pd.DataFrame(canonical_rows)

    # Clean and deduplicate by post_id
    pre_dup_count = len(df_canonical)
    df_canonical = df_canonical.drop_duplicates(subset=["post_id"], keep="first")
    duplicates_removed = pre_dup_count - len(df_canonical)

    # Sort chronologically
    df_canonical = df_canonical.sort_values("created_at_utc").reset_index(drop=True)

    # Save canonical dataset
    parquet_path = raw_dir / "btc_x_historical_180d.parquet"
    csv_path = raw_dir / "btc_x_historical_180d.csv"
    df_canonical.to_parquet(parquet_path, index=False)
    df_canonical.to_csv(csv_path, index=False)
    logger.info(f"Persisted {len(df_canonical)} canonical posts to {parquet_path} and {csv_path}")

    # Build manifest
    manifest: Dict[str, Any] = {
        "asset": "BTC",
        "source_platform": "x",
        "data_strategy": "supplementary_open_historical_archive_plus_live_apify",
        "archive_source": dataset_url,
        "collection_executed_utc": datetime.now(timezone.utc).isoformat(),
        "start_window_utc": start_dt.isoformat(),
        "end_window_utc": end_dt.isoformat(),
        "total_posts_retained": len(df_canonical),
        "duplicates_removed": duplicates_removed,
        "first_post_timestamp_utc": df_canonical["created_at_utc"].min().isoformat() if not df_canonical.empty else None,
        "last_post_timestamp_utc": df_canonical["created_at_utc"].max().isoformat() if not df_canonical.empty else None,
        "field_availability_rates": {
            "text": float(df_canonical["text"].notnull().mean()) if not df_canonical.empty else 0.0,
            "created_at_utc": float(df_canonical["created_at_utc"].notnull().mean()) if not df_canonical.empty else 0.0,
            "author_username": float(df_canonical["author_username"].notnull().mean()) if not df_canonical.empty else 0.0,
            "likes": float(df_canonical["likes"].notnull().mean()) if not df_canonical.empty else 0.0,
            "reposts": float(df_canonical["reposts"].notnull().mean()) if not df_canonical.empty else 0.0,
            "replies": float(df_canonical["replies"].notnull().mean()) if not df_canonical.empty else 0.0,
        },
        "platform_distinction_notes": {
            "x_platform": (
                "Twitter/X contains broad public discourse, breaking news, influencer commentary, "
                "and viral community posts. Metrics include likes, replies, and retweets."
            ),
            "stocktwits_platform": (
                "StockTwits is an equity/crypto investor platform characterized by structured $TICKER "
                "streams and explicit Bullish/Bearish sentiment votes. Kept strictly distinct from X."
            )
        },
        "output_files": {
            "canonical_parquet": str(parquet_path),
            "canonical_csv": str(csv_path),
            "raw_snapshot": str(raw_snapshot_path)
        }
    }

    manifest_path = raw_dir / "historical_collection_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved manifest to {manifest_path}")

    return df_canonical, manifest


class HistoricalXCollector:
    """Manages multi-batch historical collection, deduplication, and manifest tracking."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        search_terms: Optional[List[str]] = None,
        actor_id: str = "apidojo/tweet-scraper"
    ):
        self.cfg = config or get_config()
        self.actor_id = actor_id
        self.search_terms = search_terms or ["Bitcoin", "BTC", "Bitcoin price", "$BTC"]

    def run_collection(
        self,
        days_back: int = 180,
        batch_size_days: int = 7,
        max_items_per_batch: int = 10,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Executes date-batched collection.
        
        If dry_run=True, generates the plan, cost estimate, and batches without making paid calls.
        """
        cost_est = estimate_historical_x_collection_cost(
            days_back=days_back,
            target_posts_per_day=max_items_per_batch // batch_size_days or 10,
            batch_size_days=batch_size_days,
            actor_id=self.actor_id
        )

        batches = generate_date_batches(days_back=days_back, batch_size_days=batch_size_days)

        manifest: Dict[str, Any] = {
            "asset": "BTC",
            "search_terms": self.search_terms,
            "actor_id": self.actor_id,
            "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "days_back": days_back,
            "total_batches_planned": len(batches),
            "dry_run": dry_run,
            "cost_estimation": cost_est,
            "batch_results": [],
            "total_items_collected": 0,
            "unique_items_retained": 0,
            "date_coverage": {}
        }

        if dry_run:
            logger.info("Dry run mode: collection plan and cost estimation generated without calling API.")
            return manifest

        # Execute actual live collection
        token, _ = get_apify_credentials()
        from apify_client import ApifyClient
        client = ApifyClient(token=token.strip())

        raw_dir = self.cfg.raw_x_dir
        raw_dir.mkdir(parents=True, exist_ok=True)

        all_mapped_items: List[Dict[str, Any]] = []

        query_str = " OR ".join([f'"{q}"' if " " in q else q for q in self.search_terms])

        for start_str, end_str in batches:
            batch_label = f"{start_str}_to_{end_str}"
            logger.info(f"Scraping batch [{batch_label}] (query: {query_str})...")

            actor_input = {
                "searchTerms": [query_str],
                "start": start_str,
                "end": end_str,
                "maxItems": max_items_per_batch,
                "sort": "Latest"
            }

            try:
                run = client.actor(self.actor_id).call(
                    run_input=actor_input,
                    run_timeout=timedelta(seconds=120),
                    wait_duration=timedelta(seconds=120)
                )
                run_status = getattr(run, "status", None) or (run.get("status") if isinstance(run, dict) else "FAILED")
                dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)

                items = []
                if run_status == "SUCCEEDED" and dataset_id:
                    items = list(client.dataset(dataset_id).iterate_items())

                # Save raw batch file (preserving other existing files)
                batch_filename = f"historical_raw_{batch_label}.json"
                batch_path = raw_dir / batch_filename
                with open(batch_path, "w", encoding="utf-8") as f:
                    json.dump(items, f, indent=2, ensure_ascii=False, default=str)

                mapped = [map_raw_tweet_to_canonical(it, batch_label=batch_label) for it in items]
                all_mapped_items.extend(mapped)

                manifest["batch_results"].append({
                    "batch": batch_label,
                    "start": start_str,
                    "end": end_str,
                    "status": run_status,
                    "items_fetched": len(items),
                    "file": str(batch_path)
                })

            except Exception as e:
                logger.error(f"Batch [{batch_label}] failed: {e}")
                manifest["batch_results"].append({
                    "batch": batch_label,
                    "start": start_str,
                    "end": end_str,
                    "status": "ERROR",
                    "error": str(e)
                })

        # Deduplicate and sort canonical dataset
        df = pd.DataFrame(all_mapped_items)
        initial_count = len(df)
        if not df.empty:
            df = df.drop_duplicates(subset=["post_id"], keep="first")
            df = df.sort_values("created_at_utc").reset_index(drop=True)

        manifest["total_items_collected"] = initial_count
        manifest["unique_items_retained"] = len(df)

        if not df.empty:
            min_ts = df["created_at_utc"].min()
            max_ts = df["created_at_utc"].max()
            manifest["date_coverage"] = {
                "first_post_utc": min_ts.isoformat() if pd.notnull(min_ts) else None,
                "last_post_utc": max_ts.isoformat() if pd.notnull(max_ts) else None,
                "total_unique_posts": len(df)
            }

            # Save canonical dataset
            parquet_path = raw_dir / "btc_x_historical_180d.parquet"
            csv_path = raw_dir / "btc_x_historical_180d.csv"
            df.to_parquet(parquet_path, index=False)
            df.to_csv(csv_path, index=False)

        # Save collection manifest
        manifest_path = raw_dir / "historical_collection_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

        return manifest


if __name__ == "__main__":
    print("=" * 65)
    print("Ingesting 180-Day Supplementary Historical Bitcoin X Data")
    print("=" * 65)
    df, manifest = ingest_supplementary_historical_x()
    print(f"Total Unique Posts Retained: {manifest['total_posts_retained']}")
    print(f"Duplicates Removed:          {manifest['duplicates_removed']}")
    print(f"First Post (UTC):            {manifest['first_post_timestamp_utc']}")
    print(f"Last Post (UTC):             {manifest['last_post_timestamp_utc']}")
    print(f"Canonical Parquet:           {manifest['output_files']['canonical_parquet']}")
    print(f"Canonical CSV:               {manifest['output_files']['canonical_csv']}")
    print(f"Field Availability Rates:    {manifest['field_availability_rates']}")
    print("=" * 65)
