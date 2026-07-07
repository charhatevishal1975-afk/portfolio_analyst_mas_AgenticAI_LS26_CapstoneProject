"""CLI entry point.

Usage:
    python main.py TCS.NS "Tata Consultancy Services"
    python main.py AAPL "Apple Inc"
"""

import sys
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from graph import build_graph
import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
)


load_dotenv()


def main():
    if len(sys.argv) < 3:
        print('Usage: python main.py <TICKER> "<Company Name>"')
        print('Example: python main.py TCS.NS "Tata Consultancy Services"')
        sys.exit(1)

    ticker = sys.argv[1]
    company_name = sys.argv[2]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GOOGLE_API_KEY in your .env file. See .env.example.")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

    app = build_graph(llm)

    initial_state = {
        "ticker": ticker,
        "company_name": company_name,
        "steps_taken": [],
    }

    print(f"Running multi-agent analysis for {ticker} ({company_name})...\n")
    final_state = app.invoke(initial_state)

    print("Orchestration trace (supervisor decisions):")
    print(" -> ".join(final_state["steps_taken"]))
    print()

    report = final_state.get("report", "No report generated.")
    print(report)

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/report_{ticker.replace('.', '_')}.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
