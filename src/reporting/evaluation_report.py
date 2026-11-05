"""
Reporting Module:
- One-page BTC current sentiment outlook generator.
- Comprehensive technical evaluation report generator with risk metrics (Sharpe, Sortino, Max Drawdown).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from config.settings import AppConfig, get_config


def generate_one_page_sentiment_outlook(
    latest_close: float,
    social_summary: Dict[str, Any],
    recent_news: List[Dict[str, Any]],
    n0_forecast: float,
    n5_forecast: float,
    model_name: str = "GradientBoostingRegressor"
) -> str:
    """Produces the structured markdown one-page sentiment outlook."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    news_lines = []
    for item in recent_news[:5]:
        news_lines.append(f"- **{item.get('title')}** (*{item.get('source')}*, {item.get('published_date', 'recent')})")
    news_text = "\n".join(news_lines) if news_lines else "No recent external news fetched."

    outlook = f"""# CrowdWisdomTrading: Bitcoin (BTC) Current Sentiment Outlook
**Generated At:** `{now_str}`  
**Asset:** Bitcoin (`BTC-USD`) | **Latest Spot Close:** `${latest_close:,.2f}`  
**Pipeline Run Type:** Research / Assessment Mode

---

## 1. Social Sentiment Dynamics (X / Twitter)
- **Data Coverage Window:** {social_summary.get('date_range', ['N/A', 'N/A'])[0]} to {social_summary.get('date_range', ['N/A', 'N/A'])[1]}
- **Total Posts Ingested:** `{social_summary.get('total_posts', 0):,}`
- **Field Completeness Rates:**
  - Likes availability: `{social_summary.get('field_availability_rates', {}).get('likes', 0.0) * 100:.1f}%`
  - Views availability: `{social_summary.get('field_availability_rates', {}).get('views', 0.0) * 100:.1f}%`
  - Follower counts availability: `{social_summary.get('field_availability_rates', {}).get('author_followers', 0.0) * 100:.1f}%`

## 2. Multi-Horizon Nowcast & Prediction (n+0 to n+5)
- **Current Nowcast (n+0):** `${n0_forecast:,.2f}`
- **Subsequent Horizon (n+5):** `${n5_forecast:,.2f}`
- **Forecasting Engine:** `{model_name}` (Leakage-safe walk-forward calibrated)

## 3. Contextual News Summary (Tavily / Exa Aux Context)
> *Note: External news items are provided for qualitative background and are strictly excluded from the predictive ML feature matrix.*
{news_text}

## 4. Caveats & Methodological Disclosures
- Social sentiment exhibits high variance and temporal regime shifts.
- Past social momentum is not financial advice and does not guarantee price trajectory.
- Models exclude future lookahead information and are benchmarked against no-change persistence.
"""
    return outlook


def generate_evaluation_report(
    eval_results: Dict[str, Any],
    feature_group_comparisons: Dict[str, Dict[str, Any]],
    output_path: Path
) -> str:
    """Generates the technical evaluation markdown report detailing Sharpe, Sortino, Drawdown, and Horizon MAE/RMSE."""
    risk = eval_results.get("risk_metrics", {})
    horizon_metrics = eval_results.get("horizon_metrics", {})
    
    rows = []
    for h, m in horizon_metrics.items():
        rows.append(f"| n+{h} | ${m.get('mae', 0):,.2f} | ${m.get('rmse', 0):,.2f} | {m.get('smape', 0):.2f}% | {m.get('r2', 0):.4f} | {m.get('directional_accuracy', 0) * 100:.1f}% |")
    horizon_table = "\n".join(rows)

    comp_rows = []
    for fset, res in feature_group_comparisons.items():
        h1 = res.get("horizon_metrics", {}).get(1, {})
        r = res.get("risk_metrics", {})
        comp_rows.append(f"| **{fset.capitalize()}** | ${h1.get('mae', 0):,.2f} | {h1.get('directional_accuracy', 0)*100:.1f}% | {r.get('sharpe_ratio', 0):.2f} | {r.get('max_drawdown', 0):.2f}% |")
    comp_table = "\n".join(comp_rows)

    report_content = f"""# CrowdWisdomTrading: Technical Model Evaluation Report
**Asset:** Bitcoin (`BTC-USD`)  
**Validation Methodology:** Expanding-Window Walk-Forward Validation (Chronological, No Shuffling)

---

## 1. Out-of-Sample Financial Risk & Strategy Metrics
*(Evaluated with 10 bps transaction fee + slippage assumptions)*

- **Sharpe Ratio (Annualized):** `{risk.get('sharpe_ratio', 'N/A')}`
- **Sortino Ratio (Annualized):** `{risk.get('sortino_ratio', 'N/A')}`
- **Maximum Drawdown:** `{risk.get('max_drawdown', 'N/A')}%`
- **Cumulative Strategy Return:** `{risk.get('cumulative_return', 'N/A')}%`
- **Total Evaluated Trades:** `{risk.get('total_trades', 0)}`

## 2. Multi-Horizon Forecast Metrics (n+0 to n+5)

| Horizon | MAE ($) | RMSE ($) | sMAPE (%) | R² | Directional Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
{horizon_table}

## 3. Incremental Value of Sentiment (Feature Comparison)

| Feature Set | Horizon n+1 MAE | Directional Accuracy | Out-of-Sample Sharpe | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
{comp_table}

## 4. Key Limitations & Risk Disclosures
1. **API Rate Limits & Ingestion Gaps:** Social post volume is subject to API quotas and sampling filters.
2. **Sentiment Model Error:** Lexicon and transformer classifiers may misclassify sarcasm, memes, or crypto slang.
3. **Regime Shifts:** Crypto volatility regimes and liquidity shocks can degrade historical learned relationships.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content

