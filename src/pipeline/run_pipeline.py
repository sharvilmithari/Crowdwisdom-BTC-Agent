"""
End-to-End Offline Execution Pipeline for Stages 5, 6, and 7.
- Uses existing 180-day price and social datasets locally (0 API credit usage).
- Stage 5: Clean posts, enrich with sentiment, and aggregate hourly.
- Stage 6: Build leakage-safe feature matrix and forward targets (n+0 to n+5).
- Stage 7: Chronological walk-forward validation, financial risk metrics (Sharpe, Sortino, Max Drawdown),
  model training, and evaluation reports generation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from config.settings import get_config
from src.features.build_features import build_feature_matrix
from src.sentiment.aggregate import clean_posts, enrich_posts_with_sentiment, aggregate_social_features
from src.sentiment.finbert import get_sentiment_analyzer
from src.models.baselines import HorizonForecaster, PersistenceBaseline
from src.evaluation.walk_forward import run_walk_forward_validation, calculate_metrics
from src.reporting.evaluation_report import generate_evaluation_report, generate_one_page_sentiment_outlook
from src.utils.logging_config import setup_logger

logger = setup_logger("pipeline_runner")


def run_stage_5_social_sentiment(
    raw_x_path: Path,
    output_parquet: Path,
    output_csv: Path,
    freq: str = "1h"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Stage 5: Social Sentiment Processing & Hourly Aggregation."""
    logger.info(f"Stage 5: Reading raw X posts from {raw_x_path}...")
    if not raw_x_path.exists():
        raise FileNotFoundError(f"Raw X data file not found: {raw_x_path}")

    df_x = pd.read_parquet(raw_x_path)
    df_clean, clean_stats = clean_posts(df_x)
    logger.info(f"Stage 5: Cleaned {clean_stats['initial']} posts -> {clean_stats['retained']} retained.")

    analyzer = get_sentiment_analyzer(prefer_finbert=False)
    logger.info(f"Stage 5: Enriching posts with sentiment ({analyzer.__class__.__name__})...")
    df_enriched = enrich_posts_with_sentiment(df_clean, analyzer=analyzer, batch_size=256)

    logger.info(f"Stage 5: Aggregating social features to {freq} frequency...")
    df_agg = aggregate_social_features(df_enriched, freq=freq)

    # Save processed social data
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_parquet(output_parquet, index=False)
    df_agg.to_csv(output_csv, index=False)
    logger.info(f"Stage 5: Saved {len(df_agg)} hourly social bars to {output_parquet}")

    metadata = {
        "total_posts_cleaned": clean_stats["retained"],
        "hourly_bars": len(df_agg),
        "analyzer": analyzer.__class__.__name__,
        "first_timestamp": str(df_agg["timestamp"].min()),
        "last_timestamp": str(df_agg["timestamp"].max())
    }
    return df_enriched, df_agg, metadata


def run_stage_6_feature_engineering(
    raw_prices_path: Path,
    df_social_agg: pd.DataFrame,
    output_matrix_parquet: Path,
    output_matrix_csv: Path,
    horizons: list = [0, 1, 2, 3, 4, 5]
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """Stage 6: Leakage-Safe Feature Engineering & Target Construction."""
    logger.info(f"Stage 6: Loading prices from {raw_prices_path}...")
    if not raw_prices_path.exists():
        raise FileNotFoundError(f"Price data file not found: {raw_prices_path}")

    df_prices = pd.read_csv(raw_prices_path)
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], utc=True)

    logger.info("Stage 6: Aligning price bars and social features, building targets...")
    feature_matrix, feature_groups = build_feature_matrix(df_prices, df_social_agg, horizons=horizons)

    output_matrix_parquet.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_parquet(output_matrix_parquet, index=False)
    feature_matrix.to_csv(output_matrix_csv, index=False)
    logger.info(f"Stage 6: Saved {len(feature_matrix)} feature rows to {output_matrix_parquet}")

    return feature_matrix, feature_groups


def run_stage_7_model_evaluation(
    feature_matrix: pd.DataFrame,
    feature_groups: Dict[str, list],
    reports_dir: Path,
    models_dir: Path,
    n_splits: int = 4,
    horizons: list = [0, 1, 2, 3, 4, 5]
) -> Dict[str, Any]:
    """Stage 7: Walk-Forward Validation, Financial Risk Metrics, Model Training & Reporting."""
    logger.info("Stage 7: Running Walk-Forward Validation for Combined features...")
    wf_combined = run_walk_forward_validation(
        feature_matrix,
        feature_cols=feature_groups["combined"],
        horizons=horizons,
        n_splits=n_splits,
        model_type="gradient_boosting"
    )

    logger.info("Stage 7: Running Walk-Forward Validation for Price-Only features...")
    wf_price_only = run_walk_forward_validation(
        feature_matrix,
        feature_cols=feature_groups["price_only"],
        horizons=horizons,
        n_splits=n_splits,
        model_type="gradient_boosting"
    )

    logger.info("Stage 7: Running Walk-Forward Validation for Sentiment-Only features...")
    wf_sentiment_only = run_walk_forward_validation(
        feature_matrix,
        feature_cols=feature_groups["sentiment_only"],
        horizons=horizons,
        n_splits=n_splits,
        model_type="gradient_boosting"
    )

    comparisons = {
        "combined": wf_combined,
        "price_only": wf_price_only,
        "sentiment_only": wf_sentiment_only
    }

    # Generate Evaluation Report Markdown
    eval_report_path = reports_dir / "evaluation_report.md"
    generate_evaluation_report(wf_combined, comparisons, eval_report_path)
    logger.info(f"Stage 7: Evaluation report saved to {eval_report_path}")

    # Train final full-dataset forecaster for deployment/dashboard
    final_forecaster = HorizonForecaster(model_type="gradient_boosting")
    final_forecaster.fit(
        feature_matrix,
        feature_matrix,
        feature_cols=feature_groups["combined"],
        horizons=horizons
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / "horizon_forecaster_combined.joblib"
    joblib.dump(final_forecaster, model_save_path)
    logger.info(f"Stage 7: Final forecaster model persisted to {model_save_path}")

    # Generate one-page outlook
    last_row = feature_matrix.iloc[[-1]]
    latest_preds = final_forecaster.predict(last_row, feature_cols=feature_groups["combined"], horizons=horizons)
    latest_close = float(last_row["close"].values[0])

    social_summary = {
        "date_range": [str(feature_matrix["timestamp"].min()), str(feature_matrix["timestamp"].max())],
        "total_posts": int(feature_matrix["post_count"].sum()),
        "field_availability_rates": {"likes": 1.0, "views": 0.0, "author_followers": 0.99}
    }

    outlook_md = generate_one_page_sentiment_outlook(
        latest_close=latest_close,
        social_summary=social_summary,
        recent_news=[],
        n0_forecast=float(latest_preds[0][0]),
        n5_forecast=float(latest_preds[5][0]),
        model_name="GradientBoostingRegressor"
    )
    outlook_path = reports_dir / "one_page_outlook.md"
    with open(outlook_path, "w", encoding="utf-8") as f:
        f.write(outlook_md)
    logger.info(f"Stage 7: One-page outlook saved to {outlook_path}")

    # Save JSON summary of metrics
    metrics_summary_path = reports_dir / "metrics_summary.json"
    summary_data = {
        "combined_risk": wf_combined.get("risk_metrics", {}),
        "price_only_risk": wf_price_only.get("risk_metrics", {}),
        "sentiment_only_risk": wf_sentiment_only.get("risk_metrics", {}),
        "horizon_metrics_combined": wf_combined.get("horizon_metrics", {}),
        "horizon_metrics_price_only": wf_price_only.get("horizon_metrics", {}),
        "latest_forecasts": {f"n+{h}": float(latest_preds[h][0]) for h in horizons}
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Stage 7: Metrics summary JSON saved to {metrics_summary_path}")

    return {
        "comparisons": comparisons,
        "latest_close": latest_close,
        "latest_forecasts": summary_data["latest_forecasts"]
    }


def execute_full_stages():
    """Runs Stage 5, Stage 6, and Stage 7 end-to-end with 0 API tokens."""
    cfg = get_config()
    raw_x_file = cfg.raw_x_dir / "btc_x_historical_180d.parquet"
    raw_price_file = cfg.raw_prices_dir / "btc_usd_1h.csv"

    proc_x_parquet = cfg.processed_dir / "x_features_1h.parquet"
    proc_x_csv = cfg.processed_dir / "x_features_1h.csv"
    matrix_parquet = cfg.processed_dir / "feature_matrix_180d.parquet"
    matrix_csv = cfg.processed_dir / "feature_matrix_180d.csv"

    print("==================================================")
    print("  RUNNING STAGES 5, 6, & 7 (LOCAL / ZERO CREDITS)")
    print("==================================================")

    # Stage 5
    _, df_agg, s5_meta = run_stage_5_social_sentiment(raw_x_file, proc_x_parquet, proc_x_csv)
    print(f"[STAGE 5 COMPLETE] Enriched & aggregated {s5_meta['hourly_bars']} hourly sentiment bars.")

    # Stage 6
    feat_matrix, feat_groups = run_stage_6_feature_engineering(raw_price_file, df_agg, matrix_parquet, matrix_csv)
    print(f"[STAGE 6 COMPLETE] Built feature matrix with {len(feat_matrix)} rows and {len(feat_groups['combined'])} features.")

    # Stage 7
    results = run_stage_7_model_evaluation(feat_matrix, feat_groups, cfg.reports_dir, cfg.models_dir)
    print(f"[STAGE 7 COMPLETE] Walk-forward evaluation and deliverables generated successfully.")
    print("Latest Close:", f"${results['latest_close']:,.2f}")
    print("Latest Horizon Forecasts:", results["latest_forecasts"])


if __name__ == "__main__":
    execute_full_stages()

