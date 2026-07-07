"""Sentiment Agent.

Responsibility: pull recent headlines about the company (free Google News
RSS, no API key needed) and use the LLM to make a single bullish / bearish
/ neutral judgment call with a short justification. This is the only agent
whose job is genuinely to interpret unstructured text, which is why it
gets the LLM instead of the Risk Agent.
"""

import feedparser
from urllib.parse import quote

from langchain_core.messages import HumanMessage


def fetch_headlines(company_name: str, max_items: int = 8) -> list:
    query = quote(f"{company_name} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:max_items]]


def analyze_sentiment(company_name: str, llm) -> dict:
    headlines = fetch_headlines(company_name)

    if not headlines:
        return {
            "headlines": [],
            "sentiment": "unknown",
            "reasoning": "No recent headlines found.",
        }

    prompt = f"""You are a markets sentiment analyst. Here are recent headlines
about {company_name}:

{chr(10).join(f"- {h}" for h in headlines)}

In 2-3 sentences, judge the overall sentiment as one of: bullish, bearish,
or neutral. Start your response with the single word (bullish/bearish/neutral)
followed by a colon, then your reasoning."""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    sentiment_label = "neutral"
    for label in ("bullish", "bearish", "neutral"):
        if text.lower().startswith(label):
            sentiment_label = label
            break

    return {
        "headlines": headlines,
        "sentiment": sentiment_label,
        "reasoning": text,
    }
