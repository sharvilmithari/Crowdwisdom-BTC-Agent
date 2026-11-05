from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional, Dict, Any, Tuple, List

from dotenv import load_dotenv
from config.settings import AppConfig, get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("apify_ingestion")


def mask_token(token: Optional[str]) -> str:
    """Safely masks an API token for logs/diagnostics."""
    if not token or not token.strip():
        return "NOT_CONFIGURED"
    clean = token.strip()
    if len(clean) <= 8:
        return "********"
    return f"{clean[:4]}...{clean[-4:]}"


def get_apify_credentials(
    project_root: Optional[Path] = None,
    default_actor: str = "apidojo/tweet-scraper"
) -> Tuple[str, str]:
    """Reads APIFY_API_TOKEN and APIFY_ACTOR_ID safely from .env.
    
    Raises:
        ValueError: If APIFY_API_TOKEN is missing or blank.
    """
    root = project_root or Path(__file__).resolve().parent.parent.parent
    env_file = root / ".env"
    
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv(override=True)
        
    token = os.getenv("APIFY_API_TOKEN", "").strip()
    actor_id = os.getenv("APIFY_ACTOR_ID", default_actor).strip() or default_actor
    
    if not token:
        raise ValueError("APIFY_API_TOKEN is not configured in .env file.")
        
    return token, actor_id


def verify_apify_connection(token: Optional[str] = None, actor_id: Optional[str] = None) -> Dict[str, Any]:
    """Safely verifies Apify credentials and actor accessibility.
    Never exposes raw tokens.
    """
    try:
        if token is None or actor_id is None:
            t, a = get_apify_credentials()
            token = token or t
            actor_id = actor_id or a
    except ValueError as e:
        return {"success": False, "message": str(e), "actor_id": actor_id}

    masked = mask_token(token)
    logger.info(f"Verifying Apify connection for actor '{actor_id}' with token [{masked}]")

    try:
        from apify_client import ApifyClient
        client = ApifyClient(token=token.strip())
        user_info = client.user().get()
        if not user_info:
            return {"success": False, "message": "Invalid Apify token or unauthorized.", "actor_id": actor_id}

        username = user_info.get("username", "user") if isinstance(user_info, dict) else getattr(user_info, "username", "user")
        
        # Verify actor existence
        actor_client = client.actor(actor_id.strip())
        actor_meta = actor_client.get()
        if not actor_meta:
            return {"success": False, "message": f"Actor '{actor_id}' not found or inaccessible.", "actor_id": actor_id}

        return {
            "success": True,
            "message": f"Connected successfully as Apify user '{username}'. Actor '{actor_id}' verified.",
            "actor_id": actor_id,
            "username": username
        }
    except Exception as e:
        logger.warning(f"Apify connection verification error: {type(e).__name__}")
        return {"success": False, "message": f"Connection check failed: {type(e).__name__}: {str(e)}", "actor_id": actor_id}


def run_apify_integration_test(
    output_path: Optional[Path] = None,
    query: str = "Bitcoin OR BTC",
    max_items: int = 10,
    timeout_secs: int = 180,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Executes a bounded, safe live integration test against the Apify Tweet Scraper actor.
    
    Requirements fulfilled:
    1. Reads APIFY_API_TOKEN and APIFY_ACTOR_ID from .env
    2. Uses Apify actor 'apidojo/tweet-scraper'
    3. Strictly adheres to actor schema (searchTerms, maxItems, sort)
    4. Searches for query (default 'Bitcoin OR BTC')
    5. Limits items to 10 output items (no 180 days scrape)
    6. Saves returned results to data/raw/x/btc_x_live_test.json without overwriting btc_x_test.json
    7. Masks tokens to prevent accidental credential leakage
    8. Handles timeouts and API exceptions clearly
    """
    root = project_root or Path(__file__).resolve().parent.parent.parent
    dest_path = output_path or (root / "data" / "raw" / "x" / "btc_x_live_test.json")

    report: Dict[str, Any] = {
        "success": False,
        "actor_id": None,
        "status": "NOT_STARTED",
        "items_collected": 0,
        "saved_path": str(dest_path),
        "fields_checked": {},
        "error": None,
        "duration_seconds": 0.0
    }

    start_time = time.time()

    # Step 1: Read credentials
    try:
        token, actor_id = get_apify_credentials(project_root=root)
        report["actor_id"] = actor_id
    except Exception as e:
        report["status"] = "CREDENTIAL_ERROR"
        report["error"] = str(e)
        report["duration_seconds"] = round(time.time() - start_time, 2)
        logger.error(f"Apify credential resolution failed: {e}")
        return report

    masked = mask_token(token)
    logger.info(f"Initiating safe Apify integration test with actor '{actor_id}' [token: {masked}]")

    # Step 2: Prepare schema-compliant input
    actor_input = {
        "searchTerms": [query],
        "maxItems": int(max_items),
        "sort": "Latest"
    }

    # Step 3: Execute Apify run with timeout and error handling
    try:
        from apify_client import ApifyClient
        client = ApifyClient(token=token.strip())

        logger.info(f"Triggering actor call (query='{query}', maxItems={max_items}, timeout={timeout_secs}s)...")
        run = client.actor(actor_id).call(
            run_input=actor_input,
            run_timeout=timedelta(seconds=timeout_secs),
            wait_duration=timedelta(seconds=timeout_secs)
        )

        if not run:
            report["status"] = "FAILED"
            report["error"] = "Actor call returned empty response or timed out."
            report["duration_seconds"] = round(time.time() - start_time, 2)
            return report

        run_status = run.get("status") if isinstance(run, dict) else getattr(run, "status", "UNKNOWN")
        run_id = run.get("id") if isinstance(run, dict) else getattr(run, "id", "unknown_run")
        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "default_dataset_id", None)

        report["run_id"] = run_id
        report["status"] = run_status

        if run_status != "SUCCEEDED":
            report["status"] = run_status
            report["error"] = f"Actor run ended with status '{run_status}' instead of 'SUCCEEDED'."
            report["duration_seconds"] = round(time.time() - start_time, 2)
            return report

        if not dataset_id:
            report["status"] = "NO_DATASET"
            report["error"] = "Actor succeeded but no defaultDatasetId was returned."
            report["duration_seconds"] = round(time.time() - start_time, 2)
            return report

        # Step 4: Retrieve dataset items
        logger.info(f"Fetching dataset items from dataset ID '{dataset_id}'...")
        items = list(client.dataset(dataset_id).iterate_items())
        report["items_collected"] = len(items)

        # Step 5: Save safely without overwriting existing test data
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Successfully saved {len(items)} items to '{dest_path}'")

        # Step 6: Validate field presence across items
        if items:
            has_text = any(bool(it.get("text") or it.get("fullText")) for it in items)
            has_timestamp = any(bool(it.get("createdAt") or it.get("created_at")) for it in items)
            has_likes = any(it.get("likeCount") is not None for it in items)
            has_replies = any(it.get("replyCount") is not None for it in items)
            has_retweets = any(it.get("retweetCount") is not None for it in items)

            report["fields_checked"] = {
                "text_available": has_text,
                "timestamps_available": has_timestamp,
                "likes_available": has_likes,
                "replies_available": has_replies,
                "retweets_available": has_retweets,
                "sample_keys": sorted(list(items[0].keys())) if items else []
            }
        else:
            report["fields_checked"] = {
                "text_available": False,
                "timestamps_available": False,
                "likes_available": False,
                "replies_available": False,
                "retweets_available": False,
                "sample_keys": []
            }

        report["success"] = True
        report["duration_seconds"] = round(time.time() - start_time, 2)
        return report

    except Exception as e:
        logger.error(f"Apify run encountered error: {type(e).__name__}: {e}")
        report["status"] = "EXCEPTION"
        report["error"] = f"{type(e).__name__}: {str(e)}"
        report["duration_seconds"] = round(time.time() - start_time, 2)
        return report


if __name__ == "__main__":
    print("=" * 60)
    print("Running Safe Apify Integration Test")
    print("=" * 60)
    res = run_apify_integration_test()
    print(f"Actor Run Succeeded: {res['success']}")
    print(f"Status: {res['status']}")
    print(f"Items Collected: {res['items_collected']}")
    print(f"Saved Path: {res['saved_path']}")
    print(f"Duration: {res['duration_seconds']}s")
    if res.get("fields_checked"):
        fc = res["fields_checked"]
        print("Fields Verified:")
        print(f"  - Tweet text: {fc.get('text_available')}")
        print(f"  - Timestamps: {fc.get('timestamps_available')}")
        print(f"  - Likes: {fc.get('likes_available')}")
        print(f"  - Replies: {fc.get('replies_available')}")
        print(f"  - Retweets: {fc.get('retweets_available')}")
    if res.get("error"):
        print(f"Error: {res['error']}")
    print("=" * 60)
