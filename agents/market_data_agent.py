"""Market Data Agent.

Responsibility: pull raw price history for the ticker and compute basic
descriptive stats. Deliberately dumb -- no interpretation, no judgment.
That's left to the Risk Agent and Sentiment Agent downstream.
"""

import numpy as np
import yfinance as yf


def fetch_market_data(ticker: str, period: str = "6mo") -> dict:
    hist = yf.Ticker(ticker).history(period=period)

    if hist.empty:
        return {
            "error": (
                f"No market data returned for ticker '{ticker}'. "
                "The ticker may be invalid, or Yahoo Finance may be "
                "temporarily unavailable or rate-limiting requests."
            )
        }

    closes = hist["Close"].dropna()
    returns = closes.pct_change().dropna()

    last_price = float(closes.iloc[-1])
    period_return = float((closes.iloc[-1] / closes.iloc[0]) - 1)

    return {
        "ticker": ticker,
        "period": period,
        "last_price": round(last_price, 2),
        "period_return": period_return,
        "daily_returns": returns.tolist(),
        "closes": closes.tolist(),
        "num_trading_days": len(closes),
    }