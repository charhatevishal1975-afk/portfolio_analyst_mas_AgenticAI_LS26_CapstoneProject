"""Quant Risk Agent.

Responsibility: turn a raw return series into risk metrics a quant desk
would actually look at. Uses the same toolkit as time-series econometrics
coursework (ADF test for stationarity) plus standard risk measures.
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller

TRADING_DAYS = 252
RISK_FREE_RATE = 0.07  # approx Indian T-bill rate, used for Sharpe


def compute_risk_metrics(market_data: dict) -> dict:
    if "error" in market_data:
        return {"error": market_data["error"]}

    returns = np.array(market_data["daily_returns"])
    closes = np.array(market_data["closes"])

    if len(returns) < 10:
        return {"error": "Not enough data points to compute reliable risk metrics."}

    # Annualized volatility
    daily_vol = returns.std(ddof=1)
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)

    # Annualized return + Sharpe ratio
    mean_daily_return = returns.mean()
    annual_return = mean_daily_return * TRADING_DAYS
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0.0

    # Max drawdown
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Historical 95% VaR (1-day, parametric-free)
    var_95 = float(np.percentile(returns, 5))

    # ADF test for stationarity of returns (not prices -- prices are
    # almost never stationary, returns usually are)
    adf_stat, adf_pvalue, *_ = adfuller(returns)
    is_stationary = adf_pvalue < 0.05

    return {
        "annual_volatility": float(annual_vol),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "var_95_daily": var_95,
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "returns_stationary": bool(is_stationary),
    }
