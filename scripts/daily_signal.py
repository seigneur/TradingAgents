"""
Daily trading signal runner — sends analysis to Telegram and records to dashboard.
Usage:
  python scripts/daily_signal.py --tickers BTC-USD ETH-USD --date 2026-09-30
  --date is optional. Default depends on when it runs (UTC):
    before 12:00 UTC (morning SGT run) -> yesterday, i.e. the last full daily candle
    from 12:00 UTC (evening SGT run)   -> today, using intraday data so far
"""
import argparse
import html
import os
import re
import sys
import textwrap
from datetime import date, datetime, timedelta, timezone

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


def get_price_for_date(ticker: str, target_date: date) -> float | None:
    try:
        import yfinance as yf
        today = date.today()
        if target_date >= today:
            hist = yf.Ticker(ticker).history(period="1d")
        else:
            hist = yf.Ticker(ticker).history(start=str(target_date), end=str(target_date + timedelta(days=4)))
        if not hist.empty:
            return float(hist["Close"].iloc[0])
    except Exception:
        pass
    return None


def get_signal_return(ticker: str, signal: dict, days: int) -> float | None:
    entry = signal.get("price_at_signal")
    if not entry:
        return None
    analysis_dt = date.fromisoformat(signal["analysis_date"])
    exit_price = get_price_for_date(ticker, analysis_dt + timedelta(days=days))
    if exit_price is None:
        return None
    return (exit_price - entry) / entry * 100


def fetch_memory(dashboard_url: str, ticker: str, limit: int = 10) -> list[dict]:
    try:
        resp = requests.get(f"{dashboard_url}/api/signals", params={"ticker": ticker, "limit": limit}, timeout=10)
        return resp.json()
    except Exception:
        return []


def format_memory_context(signals: list[dict]) -> str:
    if not signals:
        return ""
    lines = ["Previous signals (newest first):"]
    for s in signals:
        price_str = f" @ ${s['price_at_signal']:,.2f}" if s.get("price_at_signal") else ""
        lines.append(f"  {s['analysis_date']}: {s['action']}{price_str}")
    return "\n".join(lines)


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
    return {"decision": decision, "state": state, "llm": ta.quick_thinking_llm}


def compute_ta(ticker: str, analysis_date: str) -> dict | None:
    """RSI14, MACD(12,26,9) histogram, price vs SMA20/50, 7d change, as of analysis_date."""
    try:
        import yfinance as yf
        d = date.fromisoformat(analysis_date)
        hist = yf.Ticker(ticker).history(start=str(d - timedelta(days=150)), end=str(d + timedelta(days=1)))
        close = hist["Close"].dropna()
        if len(close) < 50:
            return None
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = float((100 - 100 / (1 + gain / loss)).iloc[-1])
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_hist = float((macd - macd.ewm(span=9, adjust=False).mean()).iloc[-1])
        last = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        lookback = 7 if ticker.endswith("-USD") else 5  # crypto trades daily, ETFs on weekdays
        chg7 = (last / float(close.iloc[-1 - lookback]) - 1) * 100
        score = sum([last > sma20, last > sma50, macd_hist > 0, rsi > 50])
        return {"rsi": rsi, "macd_hist": macd_hist, "above20": last > sma20,
                "above50": last > sma50, "chg7": chg7, "score": score}
    except Exception as exc:
        print(f"  TA failed for {ticker}: {exc}", file=sys.stderr)
        return None


def format_ta(t: dict) -> list[str]:
    """Two plain-text lines (no <code>, so they wrap cleanly on a phone)."""
    score = t["score"]
    label = {4: "strong bullish", 3: "bullish", 2: "mixed", 1: "bearish", 0: "strong bearish"}[score]
    rsi_note = " (overbought)" if t["rsi"] >= 70 else " (oversold)" if t["rsi"] <= 30 else ""
    if t["above20"] and t["above50"]:
        sma = "above SMA20/50"
    elif not t["above20"] and not t["above50"]:
        sma = "below SMA20/50"
    else:
        sma = f"{'above' if t['above20'] else 'below'} SMA20, {'above' if t['above50'] else 'below'} SMA50"
    macd = "MACD up" if t["macd_hist"] > 0 else "MACD down"
    return [
        f"<b>Momentum {score}/4</b> · {label}\n",
        f"RSI {t['rsi']:.0f}{rsi_note} · {macd} · {sma} · 7d {t['chg7']:+.1f}%",
    ]


# Emoji / pictographs the model may add to the debate summary.
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")


def debate_blurb(llm, state: dict) -> dict:
    """Condense the bull/bear debate and the final decision into three short lines."""
    debate = state.get("investment_debate_state") or {}
    bull = (debate.get("bull_history") or "")[-4000:]
    bear = (debate.get("bear_history") or "")[-4000:]
    final = (state.get("final_trade_decision") or "")[-3000:]
    if not (bull or bear):
        return {}
    prompt = (
        "Condense this trading debate for a phone notification. Reply with exactly three lines and "
        "nothing else. Plain text only: no markdown, no asterisks, no quotes. Each line at most 90 "
        "characters; use terse fragments, not sentences.\n"
        "BULL: <strongest bull argument>\n"
        "BEAR: <strongest bear argument>\n"
        "WHY: <why the final decision went the way it did>\n\n"
        f"=== BULL ===\n{bull}\n\n=== BEAR ===\n{bear}\n\n=== FINAL DECISION ===\n{final}"
    )
    try:
        text = llm.invoke(prompt).content
        if isinstance(text, list):  # content blocks
            text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
        out = {}
        for line in text.splitlines():
            key, _, val = line.partition(":")
            key = key.strip().strip("*").upper()
            val = EMOJI_RE.sub("", val.replace("*", "").replace("`", "")).strip(" _\"'")
            if key in ("BULL", "BEAR", "WHY") and val:
                out[key] = textwrap.shorten(val, width=110, placeholder="…")
        return out
    except Exception as exc:
        print(f"  Debate blurb failed: {exc}", file=sys.stderr)
        return {}


def default_analysis_date(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    d = now.date() if now.hour >= 12 else now.date() - timedelta(days=1)
    return str(d)


# Sizing for the suggested Coinbase bot commands (bot caps orders at $300).
TRADE_MAX_USD = 300.0
TRADE_USD_STRONG = float(os.getenv("TRADE_USD_STRONG", "100"))  # BUY / SELL
TRADE_USD_LEAN = float(os.getenv("TRADE_USD_LEAN", "50"))       # OVERWEIGHT / UNDERWEIGHT
TRADE_LEVERAGE = int(os.getenv("TRADE_LEVERAGE", "2"))


def trade_command(ticker: str, action: str) -> str | None:
    """Coinbase bot command for a signal, e.g. '/long BTC 100 2'. Crypto (-USD) only."""
    if not ticker.endswith("-USD"):
        return None
    side = {"BUY": "long", "OVERWEIGHT": "long",
            "SELL": "short", "SHORT": "short", "UNDERWEIGHT": "short"}.get(action)
    if not side:
        return None
    usd = TRADE_USD_STRONG if action in ("BUY", "SELL", "SHORT") else TRADE_USD_LEAN
    usd = min(usd, TRADE_MAX_USD)
    return f"/{side} {ticker[:-4]} {usd:g} {TRADE_LEVERAGE}"


def format_return(ret: float | None, action: str) -> str:
    if ret is None:
        return "—"
    bullish = action in ("BUY", "OVERWEIGHT")
    bearish = action in ("SELL", "SHORT", "UNDERWEIGHT")
    correct = (bullish and ret > 0) or (bearish and ret < 0)
    sign = "+" if ret >= 0 else ""
    mark = "hit" if correct else "miss"
    return f"{sign}{ret:.1f}% {mark}"


ACTION_LABEL = {
    "OVERWEIGHT": "OVERWEIGHT · long bias",
    "UNDERWEIGHT": "UNDERWEIGHT · short bias",
}


def format_blocks(results: list[dict], analysis_date: str) -> list[str]:
    """Message sections: header, one per ticker, footer."""
    analysis_dt = date.fromisoformat(analysis_date)
    session = "AM" if datetime.now(timezone.utc).hour < 12 else "PM"
    blocks = [f"<b>Trading Signals</b>\n{analysis_dt:%a %d %b %Y} · {session}"]

    for r in results:
        ticker = r["ticker"]
        name = ticker[:-4] if ticker.endswith("-USD") else ticker
        decision = r.get("decision", {})
        action = decision.get("action", "—").upper() if isinstance(decision, dict) else str(decision).upper()
        summary = decision.get("summary", "") if isinstance(decision, dict) else ""
        label = "analysis failed" if r.get("error") else ACTION_LABEL.get(action, action)

        lines = [f"<b>{name}</b> — {label}"]
        cmd = trade_command(ticker, action)
        if cmd:
            lines.append(f"<code>{cmd}</code>")
        if r.get("ta"):
            lines.extend(format_ta(r["ta"]))

        # Debate + summary collapsed by default; tap to expand.
        quote = []
        blurb = r.get("blurb") or {}
        for key, title in (("BULL", "Bull"), ("BEAR", "Bear"), ("WHY", "Why")):
            if blurb.get(key):
                quote.append(f"<b>{title}:</b> {html.escape(blurb[key])}")
        if summary:
            quote.append(html.escape(textwrap.shorten(summary, width=280, placeholder="…")))
        if quote:
            lines.append("<blockquote expandable>" + "\n".join(quote) + "</blockquote>")

        # Track record: returns of past signals + recent streak
        memory = r.get("memory", [])
        memory_by_date = {m["analysis_date"]: m for m in memory}
        sig_1d = memory_by_date.get(str(analysis_dt - timedelta(days=1)))
        sig_5d = memory_by_date.get(str(analysis_dt - timedelta(days=5)))
        ret1 = get_signal_return(ticker, sig_1d, 1) if sig_1d else None
        ret5 = get_signal_return(ticker, sig_5d, 5) if sig_5d else None
        record = []
        if ret1 is not None:
            record.append(f"1d {format_return(ret1, sig_1d['action'].upper())}")
        if ret5 is not None:
            record.append(f"5d {format_return(ret5, sig_5d['action'].upper())}")
        if memory:
            recent = [m["action"].upper() for m in memory[:3]]
            if len(set(recent)) == 1:
                record.append(f"{len(recent)}x {recent[0].lower()} in a row")
            else:
                record.append("prev " + " → ".join(a.lower() for a in reversed(recent)))
        if record:
            lines.append(f"<i>{html.escape(' · '.join(record))}</i>")

        if r.get("error"):
            lines.append(f"<i>{html.escape(textwrap.shorten(str(r['error']), width=200, placeholder='…'))}</i>")
        blocks.append("\n".join(lines))

    blocks.append("<i>TradingAgents · not financial advice</i>")
    return blocks


def get_live_price(ticker: str) -> float | None:
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        print(f"  Live price failed for {ticker}: {exc}", file=sys.stderr)
    return None


def store_latest_signals(table_name: str, results: list[dict], analysis_date: str) -> None:
    """Save each ticker's newest prediction to DynamoDB for the hourly watcher (scripts/signal_watch.py).

    Entry price is the live price when the signal is sent, since that's when a trade would be placed.
    """
    import boto3
    from decimal import Decimal

    table = boto3.resource("dynamodb").Table(table_name)
    sent_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in results:
        if r.get("error"):
            continue
        decision = r.get("decision", {})
        action = decision.get("action", "").upper() if isinstance(decision, dict) else str(decision).upper()
        price = get_live_price(r["ticker"])
        if not price:
            continue
        table.put_item(Item={
            "ticker": r["ticker"],
            "action": action,
            "entry_price": Decimal(str(round(price, 6))),
            "analysis_date": analysis_date,
            "sent_at": sent_at,
            "alerted_step": 0,  # adverse-move alerts already sent for this prediction
        })
        print(f"  Stored latest {r['ticker']} {action} @ {price}", flush=True)


def split_message(blocks: list[str], limit: int = 4000) -> list[str]:
    """Pack whole sections into messages under Telegram's 4096-char limit."""
    chunks, cur = [], ""
    for block in blocks:
        if cur and len(cur) + len(block) + 2 > limit:
            chunks.append(cur)
            cur = block
        else:
            cur = f"{cur}\n\n{block}" if cur else block
    chunks.append(cur)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["BTC-USD", "ETH-USD", "XRP-USD", "LINK-USD", "GLD"])
    parser.add_argument("--date", default=default_analysis_date())
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

        memory = []
        if dashboard_url:
            memory = fetch_memory(dashboard_url, ticker, limit=10)
            if memory:
                print(format_memory_context(memory), flush=True)

        try:
            r = run_analysis(ticker, args.date, args.provider, args.debug)
            llm = r.pop("llm")
            r["blurb"] = debate_blurb(llm, r["state"])
            results.append({"ticker": ticker, "memory": memory, **r})
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            results.append({"ticker": ticker, "memory": memory, "error": str(exc)})
        results[-1]["ta"] = compute_ta(ticker, args.date)

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

    # Track the newest prediction for the hourly adverse-move watcher. Skip backfills of old dates.
    signals_table = os.environ.get("SIGNALS_TABLE", "")
    if signals_table and args.date >= str(date.today() - timedelta(days=1)):
        try:
            store_latest_signals(signals_table, results, args.date)
        except Exception as exc:
            print(f"  Storing latest signals failed: {exc}", file=sys.stderr)

    blocks = format_blocks(results, args.date)
    print("\n\n".join(blocks))
    for chunk in split_message(blocks):
        send_telegram(token, chat_id, chunk)
    print("Sent to Telegram ✓")


if __name__ == "__main__":
    main()
