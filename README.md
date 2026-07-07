# Portfolio Risk & Sentiment Analyst — A Multi-Agent Research System

Built for Agentic AI Learners' Space 2026, Capstone Project.

## Problem Statement

Retail investors check three completely different things before
looking at a stock: the price chart, the actual risk profile, and what the news is
saying. In practice most people only look at the price chart and skip the other two,
because doing proper risk math (volatility, drawdown, VaR, stationarity of returns)
by hand is tedious, and reading through news takes time.

This system automates that research workflow end to end. Give it a ticker, it
returns a structured research memo covering price action, quantitative risk
metrics, and news sentiment, produced by four coordinating agents rather than
one agent doing everything.

This isn't a single agent with extra steps. Each agent needs a genuinely
different skill: pulling numeric time series, running statistical tests,
interpreting unstructured news text, and synthesizing all of it into prose.
Cramming that into one prompt produces worse output than routing each job to
a specialized step.

## Architecture — Supervisor Pattern

```
                     ┌──────────────┐
              ┌─────▶│  Supervisor  │◀─────┐
              │      └──────┬───────┘      │
              │             │              │
     (routes to next agent based on state) │
              │             │              │
   ┌──────────▼───┐ ┌───────▼──────┐ ┌─────▼────────┐ ┌────────────┐
   │ Market Data  │ │ Risk Agent   │ │  Sentiment   │ │   Report   │
   │    Agent     │ │              │ │    Agent     │ │   Agent    │
   └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘
   (yfinance)        (volatility,     (Google News     (Gemini synthesizes
                       Sharpe, VaR,     RSS + Gemini      final memo from
                       ADF test)        sentiment call)   the other 3 outputs)
```

A central **supervisor node** inspects the shared state after every step and
decides which worker runs next. Workers never call each other directly and
never decide their own successor. That's what makes this a Supervisor
pattern rather than a Pipeline: the routing decision is centralized and
re-evaluated at every hop, so if `market_data` fails, the supervisor reroutes
straight to `report` with an error message instead of blindly continuing.

Flow for the happy path: `supervisor → market_data → supervisor → risk_analysis
→ supervisor → sentiment_analysis → supervisor → report → supervisor → FINISH`

### Why 4 agents, clear division of responsibility

| Agent | Responsibility | Uses LLM? |
|---|---|---|
| Market Data Agent | Pull raw OHLC price history via yfinance, compute returns | No |
| Risk Agent | Quant risk metrics: annualized volatility, Sharpe ratio, max drawdown, 95% VaR, ADF stationarity test on returns | No |
| Sentiment Agent | Pull recent headlines (Google News RSS), judge bullish/bearish/neutral tone | Yes |
| Report Agent | Synthesize the other three outputs into one markdown memo | Yes |

Only the two agents that genuinely need judgment (interpreting news,
writing prose) call the LLM. The Risk Agent is intentionally pure Python —
you don't want an LLM doing arithmetic on a Sharpe ratio, you want
`statsmodels` and `numpy` doing it deterministically. Keeping LLM calls to 2
per run also matters practically: I'm on a free-tier Gemini quota, same as
in the Week 3 assignment.

### Cross-course connection

The Risk Agent's ADF (Augmented Dickey-Fuller) stationarity test on the
return series is a direct application of the time series econometrics
covered earlier in this bootcamp. Prices are non-stationary; returns
generally are. Checking that assumption before trusting the volatility/VaR
numbers is what an actual quant workflow does, not just a stock analysis
gimmick.

## Setup

```bash
git clone <this-repo-url>
cd portfolio-analyst-mas
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Gemini API key into .env
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Repository Structure

```text
portfolio-analyst-mas/
├── agents/             # Individual AI agents for market data, sentiment analysis, risk assessment, and report generation
├── outputs/            # Sample generated reports included for demonstration. Running the application will generate new reports here.
├── graph.py            # Defines the agent workflow
├── main.py             # Entry point of the application
├── utils.py            # Shared utility functions
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```



## Sample Outputs
```
The `outputs/` directory contains sample reports generated by the application for the following stocks:

- `AAPL`
- `NVDA`
- `TCS.NS`
- `TSLA`

These reports are included to demonstrate the format and quality of the generated analysis. When you run the project, new reports will be generated in the `outputs/` directory based on the stock ticker(s) you provide.
```


## Usage

```bash
python main.py TCS.NS "Tata Consultancy Services"
python main.py AAPL "Apple Inc"
python main.py INFY.NS "Infosys"
```

NSE-listed stocks need the `.NS` suffix. US stocks use the plain ticker.

Output prints to console and saves to `outputs/report_<TICKER>.md`.

## Example output shape

```markdown
# TCS.NS Research Memo

## Summary
TCS trades at ₹X, up/down Y% over 6 months...

## Risk Profile
Annualized volatility of Z%, Sharpe ratio of W...
Returns are confirmed stationary (ADF p < 0.05)...

## Sentiment
Recent coverage leans [bullish/bearish/neutral] because...

## Verdict
[one line, not financial advice]
```

## Design decisions and limitations

- **Deterministic supervisor routing with LLM fallback capacity.** The
  supervisor's core logic is rule-based (check what's done, route to what's
  missing) rather than burning an LLM call on every single hop. The
  supervisor is still a first-class orchestration node with the sole
  authority over routing, which is what the pattern requires; it doesn't
  spend tokens deciding something a `state.get()` check already answers.
- **No RAG.** Considered adding a RAG layer over historical earnings call
  transcripts, cut it to keep scope shippable in the timeframe. Documented
  here as a known extension, not left unmentioned.
- **VaR is historical, not parametric.** Simpler and doesn't assume
  normally distributed returns, which real returns usually aren't.
- **Free-tier only.** Both external data sources (yfinance, Google News RSS)
  are free and require no API keys, so the only quota concern is Gemini
  calls, capped at 2 per run.

## Future extensions

- Add a Portfolio Agent that runs this across multiple tickers in parallel
  (Parallel + Aggregator pattern) and ranks them.
- RAG over the company's last 4 quarters of earnings call transcripts for
  the Sentiment Agent to ground its judgment in management commentary, not
  just headlines.
- MCP server wrapping a real brokerage API (Zerodha Kite Connect, Groww) so
  the Report Agent's verdict can be turned into an actual watchlist action.

## Tech stack

LangGraph (orchestration), LangChain (LLM wrapper), Google Gemini 2.5 Flash
(LLM), yfinance (market data), feedparser + Google News RSS (headlines),
statsmodels (ADF test), numpy/pandas (numerics).
