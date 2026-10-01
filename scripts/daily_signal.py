"""
Daily trading signal runner — sends analysis to Telegram and records to dashboard.
Usage:
  python scripts/daily_signal.py --tickers BTC-USD ETH-USD --date 2026-09-30
  ANALYSIS_DATE is optional; defaults to yesterday.
"""
import argparse
import html
import os
import sys
import textwrap
from datetime import date, timedelta

import requests


def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
    resp.raise_for_status()


def get_close_price(ticker: str, analysis_date: str) -> float | None:
    try:
        import yfinance as yf
        d = date.fromisoformat(analysis_date)
        hist = yf.Ticker(ticker).history(start=analysis_date, end=str(d + timedelta(days=4)))
        if not hist.empty:
            return float(hist["Close"].iloc[0])
    except Exception as exc:
        print(f"  Price fetch failed for {ticker}: {exc}", file=sys.stderr)
    return None


def post_to_dashboard(url: str, secret: str, ticker: str, action: str,
                      analysis_date: str, price: float | None, summary: str) -> None:
    try:
        requests.post(
            f"{url}/api/signal",
            json={"ticker": ticker, "action": action, "analysis_date": analysis_date,
                  "price_at_signal": price, "summary": summary},
            headers={"X-API-Secret": secret},
            timeout=10,
        )
    except Exception as exc:
        print(f"  Dashboard post failed: {exc}", file=sys.stderr)


def run_analysis(ticker: str, analysis_date: str, provider: str, debug: bool) -> dict:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["checkpoint_enabled"] = False

    if provider == "anthropic":
        config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "claude-sonnet-4-6")
        config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "claude-haiku-4-5-20251001")
    elif provider == "openai":
        config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "gpt-4o")
        config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "gpt-4o-mini")

    ta = TradingAgentsGraph(debug=debug, config=config)
    state, decision = ta.propagate(ticker, analysis_date)
    return {"decision": decision, "state": state}


def format_message(results: list[dict], analysis_date: str) -> str:
    lines = [f"<b>📊 Trading Signals — {analysis_date}</b>\n"]
    for r in results:
        ticker = r["ticker"]
        decision = r.get("decision", {})
        action = decision.get("action", "—").upper() if isinstance(decision, dict) else str(decision).upper()
        summary = decision.get("summary", "") if isinstance(decision, dict) else ""

        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "SHORT": "🔻"}.get(action, "⚪")

        lines.append(f"{emoji} <b>{ticker}</b>: {action}")
        if summary:
            wrapped = textwrap.shorten(summary, width=280, placeholder="…")
            lines.append(f"  {html.escape(wrapped)}")
        if r.get("error"):
            lines.append(f"  ⚠️ {r['error']}")
        lines.append("")

    lines.append("<i>🤖 TradingAgents · Claude · yfinance</i>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["BTC-USD", "ETH-USD", "GC=F", "SI=F"])
    parser.add_argument("--date", default=str(date.today() - timedelta(days=1)))
    parser.add_argument("--provider", default=os.getenv("TRADINGAGENTS_LLM_PROVIDER", "anthropic"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set", file=sys.stderr)
        sys.exit(1)

    dashboard_url = os.environ.get("DASHBOARD_URL", "")
    dashboard_secret = os.environ.get("DASHBOARD_SECRET", "")

    results = []
    for ticker in args.tickers:
        print(f"Analysing {ticker} for {args.date}…", flush=True)
        try:
            r = run_analysis(ticker, args.date, args.provider, args.debug)
            results.append({"ticker": ticker, **r})
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            results.append({"ticker": ticker, "error": str(exc)})

    # Record signals to dashboard
    if dashboard_url and dashboard_secret:
        for r in results:
            if r.get("error"):
                continue
            decision = r.get("decision", {})
            action = decision.get("action", "").upper() if isinstance(decision, dict) else str(decision).upper()
            summary = decision.get("summary", "") if isinstance(decision, dict) else ""
            price = get_close_price(r["ticker"], args.date)
            post_to_dashboard(dashboard_url, dashboard_secret, r["ticker"], action, args.date, price, summary)
            print(f"  Dashboard ✓ {r['ticker']} {action} @ {price}", flush=True)

    msg = format_message(results, args.date)
    print(msg)
    send_telegram(token, chat_id, msg)
    print("Sent to Telegram ✓")


if __name__ == "__main__":
    main()
