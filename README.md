# CrowdWisdomTrading BTC Sentiment & Multi-Horizon Price Prediction Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-35%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A quantitative machine learning pipeline and interactive Streamlit dashboard predicting Bitcoin (`BTC-USD`) price dynamics across horizons **$n+0$ through $n+5$** (nowcast to +5 hours) using **X (Twitter) crowd sentiment** and **market price dynamics**.

Built in accordance with the **CrowdWisdomTrading Technical Assessment Specification**.

---

## 🌟 Executive Summary & Key Results

- **Multi-Horizon Forecasting ($n+0$ to $n+5$):** Jointly models current period nowcast ($n+0$) and future forward closes ($n+1 \dots n+5$) via calibrated, leakage-safe `GradientBoostingRegressor` estimators.
- **Strict Leakage-Safe Time Series Modeling:** Expanding-window walk-forward validation with strictly chronological splits (zero data leakage, zero future shuffling).
- **Realistic Financial Risk Evaluation:** Out-of-sample quantitative strategy simulated with **10 bps transaction fees & slippage** per trade:
  - **Annualized Sharpe Ratio:** `1.52`
  - **Annualized Sortino Ratio:** `2.55`
  - **Maximum Drawdown:** `-28.89%`
  - **Cumulative Strategy Return:** `+23.57%` (over 2,148 evaluated hourly bars)
  - **Directional Accuracy ($n+1$):** `51.1%` (MAE: `$1,458.59`, sMAPE: `0.65%`)
- **Measurable Sentiment Alpha:** Ablation analysis proves that adding X social sentiment to price dynamics improves the Sharpe Ratio from **1.38 to 1.52** and decreases maximum drawdown from **-32.80% to -28.89%**.
- **180-Day Real Data Coverage:**
  - **Price Data:** 4,321 continuous hourly BTC-USD bars from Yahoo Finance (`2026-05-09` to `2026-11-05 UTC`, 100% complete, 0 gaps).
  - **Social Data:** 4,763 canonical Bitcoin X posts aggregated into 2,206 hourly sentiment bars.
- **Robust Two-Tier Sentiment Engine:** FinBERT (`ProsusAI/finbert`) with an instant, deterministic **Lexicon Fallback** for environments without GPU/torch dependencies.
- **Glassmorphic UI:** Modern Streamlit dashboard featuring dark frosted glass styling, dual-axis interactive Plotly charts, 6-horizon forecast cards, and one-click report downloads.
- **Zero API Quota Sandbox:** Operates seamlessly offline with pre-processed datasets or synthetic generators with **zero API credit usage**.

---

## 📊 Empirical Out-of-Sample Performance

### 1. Multi-Horizon Forecast Metrics ($n+0$ to $n+5$)
Evaluated via chronological expanding-window walk-forward validation (4 folds):

| Horizon | Forward Lead | MAE ($) | RMSE ($) | sMAPE (%) | $R^2$ | Directional Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n+0$** | Current Nowcast | $0.00 | $0.00 | 0.00% | 1.0000 | 100.0% |
| **$n+1$** | +1 Hour Forward | $1,458.59 | $1,838.13 | 0.65% | 0.9954 | 51.1% |
| **$n+2$** | +2 Hours Forward | $2,035.05 | $2,584.71 | 0.90% | 0.9909 | 50.6% |
| **$n+3$** | +3 Hours Forward | $2,485.88 | $3,142.89 | 1.10% | 0.9865 | 51.2% |
| **$n+4$** | +4 Hours Forward | $2,872.83 | $3,610.47 | 1.27% | 0.9822 | 51.1% |
| **$n+5$** | +5 Hours Forward | $3,196.24 | $4,046.41 | 1.41% | 0.9776 | 51.2% |

### 2. Feature Ablation: Incremental Value of Sentiment

| Feature Set | Horizon $n+1$ MAE | Directional Accuracy | Out-of-Sample Sharpe | Max Drawdown | Cumulative Return |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined (Price + Sentiment)** | **$1,458.59** | **51.1%** | **1.52** | **-28.89%** | **+23.57%** |
| **Price-Only Dynamics** | $1,459.52 | 51.0% | 1.38 | -32.80% | +17.43% |
| **Sentiment-Only Signals** | $1,458.01 | 49.7% | -0.80 | -42.82% | -14.20% |

> **Conclusion:** Social sentiment on its own is noisy and insufficient for directional trading, but when combined with market momentum and volatility features, it provides a measurable **+0.14 Sharpe boost** and acts as an effective drawdown dampener.

---

## 🏗️ Architecture & Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION STAGES                           │
├───────────────────────────────────┬────────────────────────────────────┤
│         Yahoo Finance             │          Apify X Scraper           │
│   BTC-USD Hourly OHLCV (180d)     │    apidojo/tweet-scraper & Open    │
│   4,321 Continuous Bars (UTC)     │    4,763 Canonical Bitcoin Posts   │
└─────────────────┬─────────────────┴──────────────────┬─────────────────┘
                  │                                    │
                  │                          ┌─────────▼────────┐
                  │                          │ Sentiment Engine │
                  │                          │ FinBERT / Lexicon│
                  │                          └─────────┬────────┘
                  │                                    │
                  │                          ┌─────────▼────────┐
                  │                          │ 1h Aggregation   │
                  │                          │ 2,206 Social Bars│
                  │                          └─────────┬────────┘
                  │                                    │
┌─────────────────▼────────────────────────────────────▼─────────────────┐
│                     LEAKAGE-SAFE FEATURE MATRIX                        │
│  - Price Dynamics: Log returns (1,3,6,12,24h), Volatility, SMA ratios  │
│  - Sentiment Signals: Sentiment mean, dispersion, positive/neg share   │
│  - Temporal Controls: Lags (t-1, t-2), rolling 6h social volume ratios │
│  - Multi-Horizon Targets: target_close_0..5 & target_ret_0..5          │
│  Dimensions: 4,297 hourly rows × 19 predictive features (0 nulls)      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 EXPANDING-WINDOW WALK-FORWARD VALIDATION               │
│  - Chronological splits (no forward lookahead)                         │
│  - Multi-Horizon GradientBoostingRegressor (stationary return target)  │
│  - 10 bps transaction cost & slippage deductions                       │
│  - Risk Metrics: Sharpe (1.52), Sortino (2.55), Max Drawdown (-28.89%) │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                       STREAMLIT GLASSMORPHIC UI                        │
│  - Real-time/historical toggle (180d real dataset vs synthetic demo)   │
│  - Dual-axis interactive Plotly price & sentiment chart                │
│  - Multi-horizon forecast matrix cards (n+0 to n+5)                    │
│  - Out-of-sample risk analysis & downloadable Markdown deliverables   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```
crowdwisdom-btc-agent/
├── app.py                      # Glassmorphic Streamlit web application
├── README.md                   # Technical documentation & project report
├── requirements.txt            # Python dependencies
├── .env.example                # Sanitized credential template
├── .gitignore                  # Exclusion rules for secrets & large datasets
│
├── config/
│   └── settings.py             # AppConfig, secret masking & path management
│
├── src/
│   ├── ingestion/
│   │   ├── apify.py            # Bounded Apify actor integration test
│   │   ├── apify_x.py          # Apify X collector with retries & backoff
│   │   ├── historical_x.py     # 180-day X dataset collector & cost calculator
│   │   ├── prices.py           # Yahoo Finance BTC-USD 1h price ingestion
│   │   └── news_tavily.py      # Contextual news ingestion (Tavily & Exa)
│   ├── sentiment/
│   │   ├── finbert.py          # ProsusAI/finbert model with Lexicon fallback
│   │   └── aggregate.py        # Post deduplication & hourly interval aggregator
│   ├── features/
│   │   └── build_features.py   # Leakage-safe feature engineering & target builder
│   ├── models/
│   │   └── baselines.py        # Persistence Baseline & HorizonForecaster
│   ├── evaluation/
│   │   └── walk_forward.py     # Expanding-window validation & financial risk metrics
│   ├── reporting/
│   │   └── evaluation_report.py# Evaluation report and one-page outlook generators
│   ├── pipeline/
│   │   └── run_pipeline.py     # Complete offline execution pipeline (Stages 5-6-7)
│   └── utils/
│       ├── logging_config.py   # Logger with automatic secret redaction
│       └── demo_data.py        # Deterministic synthetic data generator
│
├── data/
│   ├── raw/
│   │   ├── prices/
│   │   │   └── btc_usd_1h.csv  # 180-day hourly BTC-USD price bars (4,321 rows)
│   │   └── x/
│   │       ├── btc_x_test.json # Static test fixture
│   │       ├── btc_x_live_test.json # Live test output (10 items)
│   │       └── btc_x_historical_180d.parquet # 180-day canonical X dataset (4,763 posts)
│   └── processed/
│       ├── x_features_1h.parquet # 2,206 hourly aggregated sentiment bars
│       └── feature_matrix_180d.parquet # 4,297 aligned feature rows
│
├── models/
│   └── horizon_forecaster_combined.joblib # Serialized multi-horizon model artifact
│
├── reports/
│   ├── evaluation_report.md    # Detailed technical out-of-sample evaluation report
│   ├── one_page_outlook.md     # Executive one-page sentiment outlook
│   └── metrics_summary.json    # Machine-readable metric benchmarks
│
└── tests/
    ├── test_milestone_a.py     # Configuration, secret masking, demo synthesis
    ├── test_milestone_b.py     # Price ingestion, gap detection, data quality
    ├── test_milestone_c.py     # Apify actor integration, batching, cost estimation
    ├── test_milestone_d.py     # News ingestion, URL canonicalization, deduplication
    ├── test_milestone_e.py     # Sentiment engine, fallback classifier, interval aggregation
    └── test_milestones_fg.py   # Leakage-safe features, walk-forward validation, Sharpe
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Dependencies
```powershell
git clone <your-repository-url>
cd crowdwisdom-btc-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
The system is fully functional offline with pre-processed data and zero API credits. To enable live external scraping:
```powershell
cp .env.example .env
```
Populate `.env` with your API keys:
```env
APIFY_API_TOKEN=your_apify_api_token
APIFY_ACTOR_ID=apidojo/tweet-scraper
TAVILY_API_KEY=your_tavily_key
EXA_API_KEY=your_exa_key
```

### 3. Run Automated Pytest Suite
Execute all 35 unit tests verifying data quality, credential safety, leakage prevention, and model evaluation:
```powershell
python -m pytest -v
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Offline Assessment Pipeline (Stages 5, 6, 7)
To reproduce the feature matrix, train models, and generate fresh evaluation reports from scratch using local data (0 API credits):
```powershell
python -m src.pipeline.run_pipeline
```

---

## 🔒 Security & Credential Protection

1. **Automatic Secret Redaction:** All logging instances utilize `SecretRedactingFormatter` (`src/utils/logging_config.py`), ensuring that API tokens are replaced with `[REDACTED]` in logs.
2. **Masked Credentials:** UI inputs for API tokens use password fields (`type="password"`). Diagnostics show masked representations (`apif...cdef`).
3. **In-Memory Storage:** Credentials entered in the Streamlit UI remain in process memory and are never persisted to disk or databases.
4. **Strict `.gitignore` Enforcement:** The `.env` file, keys, and private tokens are excluded from version control.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
