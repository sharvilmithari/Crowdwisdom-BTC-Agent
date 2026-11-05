from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd

from config.settings import AppConfig, get_config
from src.utils.logging_config import setup_logger
from src.utils.demo_data import generate_demo_posts

logger = setup_logger("apify_ingestion")


class ApifyConnectionStatus:
    def __init__(self, success: bool, message: str, actor_id: Optional[str] = None):
        self.success = success
        self.message = message
        self.actor_id = actor_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "actor_id": self.actor_id
        }


def verify_apify_connection(token: Optional[str], actor_id: Optional[str]) -> ApifyConnectionStatus:
    """Safely tests the Apify token and actor validity with a minimal call.
    Never logs or exposes the token.
    """
    if not token or not token.strip():
        return ApifyConnectionStatus(False, "Apify API token is not configured.")
    if not actor_id or not actor_id.strip():
        return ApifyConnectionStatus(False, "Apify Actor ID is not configured.")

    try:
        from apify_client import ApifyClient
        client = ApifyClient(token=token.strip())
        user_info = client.user().get()
        if not user_info:
            return ApifyConnectionStatus(False, "Invalid Apify token or unauthorized access.")

        # Check actor metadata
        actor_client = client.actor(actor_id.strip())
        actor_meta = actor_client.get()
        if not actor_meta:
            return ApifyConnectionStatus(False, f"Actor '{actor_id}' not found or inaccessible.")

        return ApifyConnectionStatus(
            True,
            f"Connected successfully as user '{user_info.get('username', 'user')}'. Actor verified.",
            actor_id=actor_id
        )
    except Exception as e:
        logger.warning(f"Apify connection check failed: {type(e).__name__}")
        return ApifyConnectionStatus(False, f"Connection failed: {str(e)}")


def map_raw_item_to_canonical(item: Dict[str, Any], run_id: str, asset: str = "BTC") -> Dict[str, Any]:
    """Maps varied Apify actor item schemas to canonical post schema.
    Canonical fields:
    - post_id: string
    - text: string
    - created_at_utc: datetime (timezone-aware)
    - likes: nullable numeric
    - reposts: nullable numeric
    - views: nullable numeric
    - author_followers: nullable numeric
    - source_platform: 'x'
    - asset: BTC
    - post_url: nullable string
    - language: nullable string
    - collection_run_id: string
    """
    # 1. Post ID
    raw_id = item.get("id") or item.get("id_str") or item.get("tweet_id") or item.get("postId")
    post_id = str(raw_id) if raw_id is not None else ""

    # 2. Text
    text = item.get("text") or item.get("full_text") or item.get("tweet") or item.get("message") or ""

    # 3. Timestamp
    raw_ts = item.get("created_at") or item.get("createdAt") or item.get("timestamp") or item.get("date")
    created_at_utc = None
    if raw_ts:
        try:
            ts_parsed = pd.to_datetime(raw_ts, utc=True)
            created_at_utc = ts_parsed.to_pydatetime()
        except Exception:
            created_at_utc = datetime.now(timezone.utc)
    else:
        created_at_utc = datetime.now(timezone.utc)

    # 4. Engagement fields (strictly preserve None, never fabricate)
    likes = item.get("likeCount") or item.get("favorite_count") or item.get("likes") or item.get("favoriteCount")
    reposts = item.get("retweetCount") or item.get("retweet_count") or item.get("reposts") or item.get("retweets")
    views = item.get("viewCount") or item.get("views") or item.get("impressions")
    
    # Author followers
    user_obj = item.get("author") or item.get("user") or {}
    followers = None
    if isinstance(user_obj, dict):
        followers = user_obj.get("followers") or user_obj.get("followers_count") or user_obj.get("followersCount")
    if followers is None:
        followers = item.get("authorFollowers") or item.get("followersCount")

    # URL
    post_url = item.get("url") or item.get("twitterUrl") or (f"https://x.com/i/web/status/{post_id}" if post_id else None)
    
    # Language
    lang = item.get("lang") or item.get("language")

    return {
        "post_id": post_id,
        "text": str(text),
        "created_at_utc": created_at_utc,
        "likes": float(likes) if likes is not None else None,
        "reposts": float(reposts) if reposts is not None else None,
        "views": float(views) if views is not None else None,
        "author_followers": float(followers) if followers is not None else None,
        "source_platform": "x",
        "asset": asset,
        "post_url": post_url,
        "language": lang,
        "collection_run_id": run_id
    }


def run_apify_collection(
    actor_id: str,
    actor_input: Dict[str, Any],
    token: Optional[str] = None,
    timeout_secs: int = 180,
    max_retries: int = 3,
    config: Optional[AppConfig] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Executes an Apify actor run with polling and backoff retries.
    Returns (raw_items_list, run_id).
    """
    cfg = config or get_config()
    auth_token = token or cfg.apify_api_token

    if not auth_token:
        raise ValueError("Apify token not provided and not set in environment.")

    from apify_client import ApifyClient
    client = ApifyClient(token=auth_token.strip())

    run_id = f"apify_{int(time.time())}"
    retries = 0
    delay = 2

    while retries < max_retries:
        try:
            logger.info(f"Starting Apify Actor '{actor_id}' run...")
            run = client.actor(actor_id).call(
                run_input=actor_input,
                run_timeout=timedelta(seconds=timeout_secs),
                wait_duration=timedelta(seconds=timeout_secs)
            )
            
            if not run or run.get("status") != "SUCCEEDED":
                status = run.get("status") if run else "FAILED"
                raise RuntimeError(f"Apify actor run did not succeed. Status: {status}")

            dataset_id = run.get("defaultDatasetId")
            logger.info(f"Apify run succeeded. Fetching dataset items from: {dataset_id}")
            
            dataset_items = list(client.dataset(dataset_id).iterate_items())
            run_id = run.get("id", run_id)
            return dataset_items, run_id

        except Exception as e:
            retries += 1
            logger.warning(f"Apify execution failed attempt {retries}/{max_retries}: {e}")
            if retries >= max_retries:
                raise
            time.sleep(delay)
            delay *= 2

    return [], run_id


def ingest_x_posts(
    use_demo: bool = False,
    actor_id: Optional[str] = None,
    actor_input: Optional[Dict[str, Any]] = None,
    save_raw: bool = True,
    config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Primary pipeline entry point for X data ingestion.
    Supports either live Apify run or offline demo mode.
    """
    cfg = config or get_config()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if use_demo or cfg.is_demo_mode():
        logger.info("Executing X ingestion in offline demo mode.")
        df = generate_demo_posts(n_posts=250, days_back=14)
        df["collection_run_id"] = run_id
        raw_items = df.to_dict(orient="records")
    else:
        chosen_actor = actor_id or cfg.apify_actor_id
        chosen_input = actor_input or cfg.apify_actor_input_schema
        raw_items, actual_run_id = run_apify_collection(
            actor_id=chosen_actor,
            actor_input=chosen_input,
            token=cfg.apify_api_token,
            config=cfg
        )
        run_id = actual_run_id
        canonical_rows = [map_raw_item_to_canonical(item, run_id=run_id, asset="BTC") for item in raw_items]
        df = pd.DataFrame(canonical_rows)
        if not df.empty:
            df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True)

    # Save raw and canonical records
    if save_raw and raw_items:
        cfg.raw_x_dir.mkdir(parents=True, exist_ok=True)
        raw_json_path = cfg.raw_x_dir / f"{run_id}_raw.json"
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_items, f, default=str)
            
        parquet_path = cfg.raw_x_dir / f"{run_id}_canonical.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Persisted {len(df)} records to {parquet_path}")

    # Compute coverage / availability metrics
    summary = {
        "run_id": run_id,
        "total_posts": len(df),
        "source_platform": "x",
        "date_range": [
            df["created_at_utc"].min().isoformat() if not df.empty else None,
            df["created_at_utc"].max().isoformat() if not df.empty else None
        ],
        "field_availability_rates": {
            "likes": float((df["likes"].notnull().sum() / len(df))) if len(df) else 0.0,
            "reposts": float((df["reposts"].notnull().sum() / len(df))) if len(df) else 0.0,
            "views": float((df["views"].notnull().sum() / len(df))) if len(df) else 0.0,
            "author_followers": float((df["author_followers"].notnull().sum() / len(df))) if len(df) else 0.0,
        }
    }

    return df, summary

