from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from config.settings import AppConfig, get_config
from src.utils.logging_config import setup_logger
from src.utils.demo_data import generate_demo_prices

logger = setup_logger("price_ingestion")


class PriceQualityReport:
    """Encapsulates data quality checks on price time series."""

    def __init__(
        self,
        total_bars: int,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        is_monotonic: bool,
        duplicate_timestamps: int,
        null_counts: Dict[str, int],
        detected_gaps: int,
        frequency: str = "1h",
        gap_details: Optional[List[Dict[str, Any]]] = None,
        expected_bars: int = 0,
        coverage_pct: float = 100.0,
        notes: str = ""
    ):
        self.total_bars = total_bars
        self.start_time = start_time
        self.end_time = end_time
        self.is_monotonic = is_monotonic
        self.duplicate_timestamps = duplicate_timestamps
        self.null_counts = null_counts
        self.detected_gaps = detected_gaps
        self.frequency = frequency
        self.gap_details = gap_details or []
        self.expected_bars = expected_bars
        self.coverage_pct = coverage_pct
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bars": self.total_bars,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "is_monotonic": self.is_monotonic,
            "duplicate_timestamps": self.duplicate_timestamps,
            "null_counts": self.null_counts,
            "detected_gaps": self.detected_gaps,
            "frequency": self.frequency,
            "gap_details": self.gap_details,
            "expected_bars": self.expected_bars,
            "coverage_pct": self.coverage_pct,
            "notes": self.notes
        }


class BasePriceProvider(ABC):
    """Abstract base class ensuring replaceable price providers."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        days_back: Optional[int] = None,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """Fetches OHLCV data. Returns dataframe with canonical columns:
        [timestamp, open, high, low, close, volume, asset]
        """
        pass


class DemoPriceProvider(BasePriceProvider):
    """Offline demo provider for fast, deterministic, quota-free execution."""

    def fetch_ohlcv(
        self,
        symbol: str = "BTC-USD",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        days_back: Optional[int] = None,
        interval: str = "1h"
    ) -> pd.DataFrame:
        logger.info(f"Using DemoPriceProvider for {symbol} (offline mode)")
        days = days_back or 30
        if start_date and end_date:
            days = max(1, (end_date - start_date).days)
        elif period and period.endswith("d"):
            try:
                days = int(period[:-1])
            except ValueError:
                days = 30
        df = generate_demo_prices(days_back=days, interval_hours=1)
        df["asset"] = symbol
        return df


class YFinancePriceProvider(BasePriceProvider):
    """Yahoo Finance price provider via yfinance library."""

    def fetch_ohlcv(
        self,
        symbol: str = "BTC-USD",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        days_back: Optional[int] = None,
        interval: str = "1h"
    ) -> pd.DataFrame:
        import yfinance as yf

        logger.info(f"Fetching {symbol} from Yahoo Finance (interval={interval})...")
        ticker = yf.Ticker(symbol)

        # Note on Yahoo Finance intraday limitations:
        # - Hourly (1h) data is available for up to 730 days from the current date.
        # - 1m/5m data is restricted to 7 days and 60 days respectively.
        kwargs: Dict[str, Any] = {"interval": interval}

        if start_date:
            kwargs["start"] = start_date.strftime("%Y-%m-%d")
            if end_date:
                kwargs["end"] = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        elif period:
            kwargs["period"] = period
        elif days_back:
            kwargs["period"] = f"{int(days_back)}d"
        else:
            kwargs["period"] = "180d"

        try:
            hist = ticker.history(**kwargs)
        except Exception as e:
            logger.error(f"Failed to fetch data from yfinance: {e}")
            raise

        if hist.empty:
            logger.warning(f"Yahoo Finance returned empty history for {symbol}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "asset"])

        hist = hist.reset_index()
        # Find timestamp column (Datetime or Date)
        date_col = "Datetime" if "Datetime" in hist.columns else "Date"

        # Construct canonical dataframe
        df = pd.DataFrame()
        df["timestamp"] = pd.to_datetime(hist[date_col], utc=True)
        df["open"] = hist["Open"].astype(float)
        df["high"] = hist["High"].astype(float)
        df["low"] = hist["Low"].astype(float)
        df["close"] = hist["Close"].astype(float)
        df["volume"] = hist["Volume"].astype(float)
        df["asset"] = symbol

        return df


def parse_interval_to_timedelta(interval: str) -> timedelta:
    """Parses interval string (e.g. '1h', '2h', '1d', '30m') to timedelta."""
    interval = interval.strip().lower()
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    elif interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))
    elif interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    return timedelta(hours=1)


def validate_and_clean_prices(
    df: pd.DataFrame,
    expected_interval: str = "1h"
) -> Tuple[pd.DataFrame, PriceQualityReport]:
    """Validates monotonic timestamps, duplicates, missingness, and checks for gaps.
    
    Important: Missing observations are NEVER fabricated with synthetic prices.
    """
    if df.empty:
        report = PriceQualityReport(
            total_bars=0,
            start_time=None,
            end_time=None,
            is_monotonic=True,
            duplicate_timestamps=0,
            null_counts={},
            detected_gaps=0,
            frequency=expected_interval,
            gap_details=[],
            expected_bars=0,
            coverage_pct=0.0,
            notes="DataFrame is empty."
        )
        return df, report

    cleaned = df.copy()

    # 1. Normalize timestamps to UTC
    if cleaned["timestamp"].dt.tz is None:
        cleaned["timestamp"] = cleaned["timestamp"].dt.tz_localize(timezone.utc)
    else:
        cleaned["timestamp"] = cleaned["timestamp"].dt.tz_convert(timezone.utc)

    # 2. Check and remove duplicate timestamps
    dup_count = int(cleaned.duplicated(subset=["timestamp"]).sum())
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate timestamps in price data. Dropping duplicates.")
        cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last")

    # 3. Sort chronologically
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)
    is_monotonic = bool(cleaned["timestamp"].is_monotonic_increasing)

    # 4. Check null values without fabricating missing rows
    null_counts = cleaned[["open", "high", "low", "close", "volume"]].isnull().sum().to_dict()

    # 5. Gap detection & coverage analysis
    step_delta = parse_interval_to_timedelta(expected_interval)
    threshold = step_delta * 1.5

    time_diffs = cleaned["timestamp"].diff()
    gap_indices = cleaned.index[time_diffs > threshold].tolist()

    gap_details: List[Dict[str, Any]] = []
    for idx in gap_indices:
        t_prev = cleaned.loc[idx - 1, "timestamp"]
        t_curr = cleaned.loc[idx, "timestamp"]
        delta_hrs = (t_curr - t_prev).total_seconds() / 3600.0
        missing_bars = int(round((t_curr - t_prev) / step_delta)) - 1
        gap_details.append({
            "gap_start": t_prev.isoformat(),
            "gap_end": t_curr.isoformat(),
            "gap_hours": round(delta_hrs, 2),
            "missing_bars": missing_bars
        })

    detected_gaps = len(gap_details)

    start_time = cleaned["timestamp"].min().to_pydatetime()
    end_time = cleaned["timestamp"].max().to_pydatetime()

    # Calculate expected bars in elapsed continuous window
    total_elapsed = (end_time - start_time).total_seconds()
    expected_bars = int(round(total_elapsed / step_delta.total_seconds())) + 1 if total_elapsed >= 0 else len(cleaned)
    coverage_pct = round(100.0 * len(cleaned) / expected_bars, 2) if expected_bars > 0 else 100.0

    notes = (
        f"Processed {len(cleaned)} bars at frequency {expected_interval}. "
        f"Coverage: {coverage_pct}% of continuous span ({expected_bars} expected). "
        f"Gaps detected: {detected_gaps}. Duplicates removed: {dup_count}."
    )

    report = PriceQualityReport(
        total_bars=len(cleaned),
        start_time=start_time,
        end_time=end_time,
        is_monotonic=is_monotonic,
        duplicate_timestamps=dup_count,
        null_counts=null_counts,
        detected_gaps=detected_gaps,
        frequency=expected_interval,
        gap_details=gap_details,
        expected_bars=expected_bars,
        coverage_pct=coverage_pct,
        notes=notes
    )

    return cleaned, report


def ingest_btc_prices(
    symbol: str = "BTC-USD",
    days_back: int = 180,
    interval: str = "1h",
    output_csv_path: Optional[Path] = None,
    output_parquet_path: Optional[Path] = None,
    provider_type: str = "yfinance",
    save_raw: bool = True,
    config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, PriceQualityReport]:
    """Reusable pipeline function for Stage 3 Bitcoin price data ingestion.
    
    Requirements fulfilled:
    1. Asset: BTC-USD
    2. Data source: Yahoo Finance via yfinance
    3. Hourly OHLCV data for 180 days
    4. Canonical columns: timestamp, open, high, low, close, volume, asset
    5. UTC normalized timestamps
    6. Monotonically sorted, duplicate timestamps removed
    7. Validates missing values and detects gaps without fabricating observations
    8. Persists to data/raw/prices/btc_usd_1h.csv and btc_prices_1h.parquet
    9. Preserves existing files; does not overwrite unrelated data
    10. Configurable frequency and window
    """
    cfg = config or get_config()

    if provider_type == "yfinance":
        provider: BasePriceProvider = YFinancePriceProvider()
    else:
        provider = DemoPriceProvider()

    logger.info(f"Ingesting {symbol} price data ({days_back} days, interval={interval}, provider={provider_type})...")
    raw_df = provider.fetch_ohlcv(
        symbol=symbol,
        days_back=days_back,
        interval=interval
    )

    cleaned_df, report = validate_and_clean_prices(raw_df, expected_interval=interval)

    if save_raw and not cleaned_df.empty:
        cfg.raw_prices_dir.mkdir(parents=True, exist_ok=True)

        # 1. Primary CSV format required: data/raw/prices/btc_usd_1h.csv
        csv_file = output_csv_path or (cfg.raw_prices_dir / f"btc_usd_{interval}.csv")
        cleaned_df.to_csv(csv_file, index=False)
        logger.info(f"Saved {len(cleaned_df)} price bars to CSV: {csv_file}")

        # 2. Parquet cache for high-performance downstream ML pipelines
        parquet_file = output_parquet_path or (cfg.raw_prices_dir / f"btc_prices_{interval}.parquet")
        cleaned_df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved {len(cleaned_df)} price bars to Parquet: {parquet_file}")

    return cleaned_df, report


def get_price_dataset(
    provider_type: str = "demo",
    symbol: str = "BTC-USD",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    period: Optional[str] = None,
    days_back: Optional[int] = 180,
    interval: str = "1h",
    save_raw: bool = True,
    config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, PriceQualityReport]:
    """Unified entry point to ingest and validate prices, preserving backward compatibility."""
    cfg = config or get_config()

    if provider_type == "yfinance":
        provider: BasePriceProvider = YFinancePriceProvider()
    else:
        provider = DemoPriceProvider()

    raw_df = provider.fetch_ohlcv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period=period,
        days_back=days_back,
        interval=interval
    )
    cleaned_df, report = validate_and_clean_prices(raw_df, expected_interval=interval)

    if save_raw and not cleaned_df.empty:
        cfg.raw_prices_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cfg.raw_prices_dir / f"btc_usd_{interval}.csv"
        cleaned_df.to_csv(csv_path, index=False)
        parquet_path = cfg.raw_prices_dir / f"btc_prices_{interval}.parquet"
        cleaned_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved {len(cleaned_df)} price bars to {parquet_path} and {csv_path}")

    return cleaned_df, report


if __name__ == "__main__":
    print("=" * 60)
    print("Executing Stage 3: Bitcoin Historical Price Data Ingestion")
    print("=" * 60)
    df, quality_report = ingest_btc_prices(symbol="BTC-USD", days_back=180, interval="1h")
    print(f"Asset: BTC-USD")
    print(f"Data Frequency: {quality_report.frequency}")
    print(f"Number of Rows: {quality_report.total_bars}")
    print(f"First Timestamp (UTC): {quality_report.start_time}")
    print(f"Last Timestamp (UTC):  {quality_report.end_time}")
    print(f"Is Monotonic Increasing: {quality_report.is_monotonic}")
    print(f"Duplicate Timestamps: {quality_report.duplicate_timestamps}")
    print(f"Detected Gaps: {quality_report.detected_gaps}")
    print(f"Coverage Percentage: {quality_report.coverage_pct}%")
    print(f"Null Counts: {quality_report.null_counts}")
    print(f"Notes: {quality_report.notes}")
    print("=" * 60)
