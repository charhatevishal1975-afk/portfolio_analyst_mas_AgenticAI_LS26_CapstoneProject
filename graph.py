"""LangGraph orchestration.

Pattern: SUPERVISOR.

A dedicated supervisor node inspects the shared state (what's been done
so far) and decides which worker agent runs next. Workers never talk to
each other directly and never decide their own successor -- that
authority belongs only to the supervisor. This is what distinguishes it
from a Pipeline: the routing decision is centralized and re-evaluated
after every step, not hardcoded as a fixed sequence.
"""

import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from utils import AnalystState
from agents.market_data_agent import fetch_market_data
from agents.risk_agent import compute_risk_metrics
from agents.sentiment_agent import analyze_sentiment
from agents.report_agent import generate_report

VALID_STEPS = ["market_data", "risk_analysis", "sentiment_analysis", "report", "FINISH"]


def build_graph(llm):

    def supervisor_node(state: AnalystState) -> AnalystState:
        """Decide the next worker to invoke based on what's already done."""
        status = {
            "market_data_done": state.get("market_data_done", False),
            "risk_done": state.get("risk_done", False),
            "sentiment_done": state.get("sentiment_done", False),
            "report_done": state.get("report_done", False),
        }

        # Deterministic fallback covers the common path without spending
        # an LLM call on every hop; the LLM is used only if state doesn't
        # fit the expected flow (e.g. market data fetch failed).
        if not status["market_data_done"]:
            next_step = "market_data"
        elif not status["risk_done"] and "error" not in (state.get("market_data") or {}):
            next_step = "risk_analysis"
        elif not status["sentiment_done"]:
            next_step = "sentiment_analysis"
        elif not status["report_done"]:
            next_step = "report"
        else:
            next_step = "FINISH"

        # If market data errored out, skip straight to report so the
        # user gets a clean error message instead of a crash.
        if "error" in (state.get("market_data") or {}) and not status["report_done"]:
            next_step = "report"

        state["next_step"] = next_step
        state.setdefault("steps_taken", []).append(next_step)
        return state

    def market_data_node(state: AnalystState) -> AnalystState:
        state["market_data"] = fetch_market_data(state["ticker"])
        state["market_data_done"] = True
        return state

    def risk_node(state: AnalystState) -> AnalystState:
        state["risk_metrics"] = compute_risk_metrics(state["market_data"])
        state["risk_done"] = True
        return state

    def sentiment_node(state: AnalystState) -> AnalystState:
        state["sentiment"] = analyze_sentiment(state["company_name"], llm)
        state["sentiment_done"] = True
        return state

    def report_node(state: AnalystState) -> AnalystState:
        state["report"] = generate_report(state, llm)
        state["report_done"] = True
        return state

    def route(state: AnalystState) -> str:
        return state["next_step"]

    graph = StateGraph(AnalystState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("market_data_node", market_data_node)
    graph.add_node("risk_analysis", risk_node)
    graph.add_node("sentiment_analysis", sentiment_node)
    graph.add_node("report_node", report_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "market_data": "market_data_node",
            "risk_analysis": "risk_analysis",
            "sentiment_analysis": "sentiment_analysis",
            "report": "report_node",
            "FINISH": END,
        },
    )

    # every worker reports back to the supervisor for the next decision
    graph.add_edge("market_data_node", "supervisor")
    graph.add_edge("risk_analysis", "supervisor")
    graph.add_edge("sentiment_analysis", "supervisor")
    graph.add_edge("report_node", "supervisor")

    return graph.compile()