from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.baselines import HorizonForecaster
from src.utils.logging_config import setup_logger

logger = setup_logger("evaluation")


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates symmetric Mean Absolute Percentage Error (sMAPE) safely."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    mask = denom > 1e-6
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask] / denom[mask]) * 100.0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, current_close: np.ndarray) -> Dict[str, float]:
    """Calculates regression and directional metrics."""
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    cc = current_close[valid_mask]

    if len(yt) == 0:
        return {"mae": 0.0, "rmse": 0.0, "smape": 0.0, "r2": 0.0, "directional_accuracy": 0.5}

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    smape = calculate_smape(yt, yp)
    r2 = float(r2_score(yt, yp)) if len(yt) > 2 else 0.0

    # Directional accuracy: sign(pred - current_close) == sign(actual - current_close)
    pred_dir = np.sign(yp - cc)
    act_dir = np.sign(yt - cc)
    correct_dir = (pred_dir == act_dir) | ((pred_dir == 0) & (act_dir == 0))
    dir_acc = float(np.mean(correct_dir))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "smape": round(smape, 2),
        "r2": round(r2, 4),
        "directional_accuracy": round(dir_acc, 4)
    }


def compute_financial_risk_metrics(
    strategy_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 24 * 365  # Hourly
) -> Dict[str, float]:
    """Computes out-of-sample financial metrics: Sharpe, Sortino, Max Drawdown."""
    rets = strategy_returns[~np.isnan(strategy_returns)]
    if len(rets) < 10 or np.all(rets == 0):
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "cumulative_return": 0.0,
            "total_trades": 0
        }

    mean_ret = np.mean(rets)
    std_ret = np.std(rets)

    # Sharpe Ratio
    sharpe = float((mean_ret - risk_free_rate) / (std_ret + 1e-8) * np.sqrt(annualization_factor))

    # Downside deviation for Sortino Ratio
    downside_returns = rets[rets < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    sortino = float((mean_ret - risk_free_rate) / (downside_std + 1e-8) * np.sqrt(annualization_factor))

    # Cumulative equity curve & Max Drawdown
    equity_curve = np.cumprod(1.0 + rets)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd * 100.0, 2),  # Percentage
        "cumulative_return": round((equity_curve[-1] - 1.0) * 100.0, 2),
        "total_trades": int(np.sum(rets != 0))
    }


def run_walk_forward_validation(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    horizons: List[int] = [0, 1, 2, 3, 4, 5],
    n_splits: int = 4,
    min_train_ratio: float = 0.5,
    fee_bps: float = 10.0, # 10 bps transaction fee + slippage
    model_type: str = "gradient_boosting"
) -> Dict[str, Any]:
    """Chronological expanding-window walk forward validation."""
    n = len(df_features)
    start_train_size = int(n * min_train_ratio)
    test_step = (n - start_train_size) // n_splits

    fold_metrics = []
    all_oos_preds = {h: [] for h in horizons}
    all_oos_actuals = {h: [] for h in horizons}
    all_oos_closes = []
    strategy_returns = []

    for fold in range(n_splits):
        train_end = start_train_size + fold * test_step
        test_end = min(n, train_end + test_step)

        df_train = df_features.iloc[:train_end].copy()
        df_test = df_features.iloc[train_end:test_end].copy()

        if len(df_test) == 0:
            continue

        forecaster = HorizonForecaster(model_type=model_type)
        forecaster.fit(df_train, df_train, feature_cols=feature_cols, horizons=horizons)
        preds = forecaster.predict(df_test, feature_cols=feature_cols, horizons=horizons)

        test_closes = df_test["close"].values
        all_oos_closes.extend(test_closes)

        # Record predictions and actuals
        for h in horizons:
            all_oos_preds[h].extend(preds[h])
            all_oos_actuals[h].extend(df_test[f"target_close_{h}"].values)

        # Strategy simulation using n+1 horizon forecast
        # Position: Long if predicted close > current close, Short if predicted close < current close
        pred_h1 = preds.get(1, preds.get(0))
        signals = np.sign(pred_h1 - test_closes)
        actual_h1_returns = df_test["target_ret_1"].fillna(0.0).values
        
        # Calculate raw returns and subtract trading fees on signal change
        trades = np.abs(np.diff(signals, prepend=0)) > 0
        fee_cost = trades * (fee_bps / 10000.0)
        fold_strat_returns = (signals * actual_h1_returns) - fee_cost
        strategy_returns.extend(fold_strat_returns)

    # Global out-of-sample metrics per horizon
    overall_metrics = {}
    closes_arr = np.array(all_oos_closes)
    for h in horizons:
        yt = np.array(all_oos_actuals[h])
        yp = np.array(all_oos_preds[h])
        overall_metrics[h] = calculate_metrics(yt, yp, closes_arr)

    # Risk metrics on strategy
    risk_metrics = compute_financial_risk_metrics(np.array(strategy_returns))

    return {
        "horizon_metrics": overall_metrics,
        "risk_metrics": risk_metrics,
        "n_folds": n_splits,
        "total_oos_bars": len(all_oos_closes),
        "model_type": model_type
    }

