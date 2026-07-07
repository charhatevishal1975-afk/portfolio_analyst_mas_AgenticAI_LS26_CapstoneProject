"""Report Agent.

Responsibility: the only agent that talks to the user. Takes the
structured outputs of the other three agents and synthesizes a single
coherent research memo. Does not re-derive any numbers -- it only
narrates what it's given, to avoid the LLM hallucinating stats.
"""

from langchain_core.messages import HumanMessage
from utils import fmt_pct, fmt_num


def generate_report(state: dict, llm) -> str:
    ticker = state["ticker"]
    market = state.get("market_data", {})
    risk = state.get("risk_metrics", {})
    sentiment = state.get("sentiment", {})

    if "error" in market:
        return f"# Report for {ticker}\n\nCould not generate report: {market['error']}"

    facts = f"""
Ticker: {ticker}
Last price: {market.get('last_price')}
6-month price return: {fmt_pct(market.get('period_return', 0))}

Risk metrics:
- Annualized volatility: {fmt_pct(risk.get('annual_volatility', 0))}
- Annualized return: {fmt_pct(risk.get('annual_return', 0))}
- Sharpe ratio: {fmt_num(risk.get('sharpe_ratio', 0))}
- Max drawdown: {fmt_pct(risk.get('max_drawdown', 0))}
- 1-day 95% VaR: {fmt_pct(risk.get('var_95_daily', 0))}
- Returns stationary (ADF p={fmt_num(risk.get('adf_pvalue', 0), 4)}): {risk.get('returns_stationary')}

News sentiment: {sentiment.get('sentiment', 'unknown')}
Sentiment reasoning: {sentiment.get('reasoning', 'N/A')}
Recent headlines:
{chr(10).join(f"- {h}" for h in sentiment.get('headlines', []))}
"""

    prompt = f"""You are a buy-side research associate. Write a concise
markdown research memo for {ticker} using ONLY the facts below. Do not
invent numbers. Structure it with headers: Summary, Risk Profile,
Sentiment, and a final one-line Verdict (not financial advice, just an
observation). Keep it under 300 words.

FACTS:
{facts}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
