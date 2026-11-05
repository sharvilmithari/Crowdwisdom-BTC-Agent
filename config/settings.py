from pathlib import Path
from typing import Optional, Dict, Any
import json
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Project root: crowdwisdom-btc-agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")

class AppConfig(BaseModel):
    # API credentials (masked/protected)
    apify_api_token: Optional[str] = Field(default_factory=lambda: os.getenv("APIFY_API_TOKEN"))
    apify_actor_id: Optional[str] = Field(default_factory=lambda: os.getenv("APIFY_ACTOR_ID", "apidojo/tweet-scraper"))
    tavily_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    exa_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("EXA_API_KEY"))

    # Asset settings
    asset_symbol: str = "BTC-USD"
    social_queries: list[str] = ["Bitcoin", "BTC", "#Bitcoin", "#BTC", "$BTC"]
    lookback_days: int = 180
    default_bar_interval: str = "1h"

    # Prediction horizons
    horizons: list[int] = [0, 1, 2, 3, 4, 5]

    # Directories
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_x_dir: Path = PROJECT_ROOT / "data" / "raw" / "x"
    raw_prices_dir: Path = PROJECT_ROOT / "data" / "raw" / "prices"
    raw_news_dir: Path = PROJECT_ROOT / "data" / "raw" / "news"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Apify actor input schema (customizable JSON)
    apify_actor_input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "searchTerms": ["Bitcoin", "BTC", "#BTC"],
            "maxItems": 100,
            "sort": "Latest",
            "lang": "en"
        }
    )

    def is_demo_mode(self) -> bool:
        """Returns True if any critical scraping API keys are absent."""
        return not bool(self.apify_api_token and self.tavily_api_key)

    def mask_secret(self, secret: Optional[str]) -> str:
        """Helper to safely mask credentials for UI display/logging."""
        if not secret:
            return "NOT CONFIGURED"
        if len(secret) <= 8:
            return "********"
        return f"{secret[:4]}...{secret[-4:]}"


def get_config() -> AppConfig:
    return AppConfig()

