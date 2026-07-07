"""Shared state definition and small helpers used across agents."""

from typing import TypedDict, Optional, List, Dict, Any


class AnalystState(TypedDict, total=False):
    ticker: str
    company_name: str

    # filled by market_data_agent
    market_data: Optional[Dict[str, Any]]
    market_data_done: bool

    # filled by risk_agent
    risk_metrics: Optional[Dict[str, Any]]
    risk_done: bool

    # filled by sentiment_agent
    sentiment: Optional[Dict[str, Any]]
    sentiment_done: bool

    # filled by report_agent
    report: Optional[str]
    report_done: bool

    # supervisor bookkeeping
    next_step: str
    steps_taken: List[str]


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_num(x: float, decimals: int = 3) -> str:
    return f"{x:.{decimals}f}"
