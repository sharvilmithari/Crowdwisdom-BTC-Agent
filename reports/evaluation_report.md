# CrowdWisdomTrading: Technical Model Evaluation Report
**Asset:** Bitcoin (`BTC-USD`)  
**Validation Methodology:** Expanding-Window Walk-Forward Validation (Chronological, No Shuffling)

---

## 1. Out-of-Sample Financial Risk & Strategy Metrics
*(Evaluated with 10 bps transaction fee + slippage assumptions)*

- **Sharpe Ratio (Annualized):** `1.52`
- **Sortino Ratio (Annualized):** `2.55`
- **Maximum Drawdown:** `-28.89%`
- **Cumulative Strategy Return:** `23.57%`
- **Total Evaluated Trades:** `2148`

## 2. Multi-Horizon Forecast Metrics (n+0 to n+5)

| Horizon | MAE ($) | RMSE ($) | sMAPE (%) | R² | Directional Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| n+0 | $0.00 | $0.00 | 0.00% | 1.0000 | 100.0% |
| n+1 | $1,458.59 | $1,838.13 | 0.65% | 0.9954 | 51.1% |
| n+2 | $2,035.05 | $2,584.71 | 0.90% | 0.9909 | 50.6% |
| n+3 | $2,485.88 | $3,142.89 | 1.10% | 0.9865 | 51.2% |
| n+4 | $2,872.83 | $3,610.47 | 1.27% | 0.9822 | 51.1% |
| n+5 | $3,196.24 | $4,046.41 | 1.41% | 0.9776 | 51.2% |

## 3. Incremental Value of Sentiment (Feature Comparison)

| Feature Set | Horizon n+1 MAE | Directional Accuracy | Out-of-Sample Sharpe | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
| **Combined** | $1,458.59 | 51.1% | 1.52 | -28.89% |
| **Price_only** | $1,459.52 | 51.0% | 1.38 | -32.80% |
| **Sentiment_only** | $1,458.01 | 49.7% | -0.80 | -42.82% |

## 4. Key Limitations & Risk Disclosures
1. **API Rate Limits & Ingestion Gaps:** Social post volume is subject to API quotas and sampling filters.
2. **Sentiment Model Error:** Lexicon and transformer classifiers may misclassify sarcasm, memes, or crypto slang.
3. **Regime Shifts:** Crypto volatility regimes and liquidity shocks can degrade historical learned relationships.
