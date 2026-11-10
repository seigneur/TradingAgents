"""
Hourly watcher: alerts Telegram when the market moves against the latest prediction.

Reads each ticker's newest prediction from DynamoDB (written by daily_signal.py), compares
the live price with the entry price, and sends an alert each time the adverse move crosses
another ALERT_STEP_PCT (default 3%: alerts at -3%, -6%, -9%, ...). Each step alerts once
per prediction; a new prediction resets it.

Usage (cron, hourly):
  python scripts/signal_watch.py            # check and alert
  python scripts/signal_watch.py --dry-run  # print what it would send

Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SIGNALS_TABLE (default tradingagents-signals),
     ALERT_STEP_PCT (default 3), AWS credentials/region as usual for boto3.
"""
import argparse
import contextlib
import html
import os
import sys
from datetime import UTC, datetime

import boto3
import requests
import yfinance as yf

LONG = {"BUY", "OVERWEIGHT"}
SHORT = {"SELL", "SHORT", "UNDERWEIGHT"}


def live_price(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        print(f"{ticker}: price fetch failed: {exc}", file=sys.stderr)
    return None


def send_telegram(token: str, chat_id: str, text: str) -> None:
    resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                         json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
    resp.raise_for_status()


def fmt_price(p: float) -> str:
    return f"{p:,.0f}" if p >= 1000 else f"{p:,.2f}" if p >= 1 else f"{p:.4f}"


def format_alert(item: dict, price: float, move: float, adverse: float) -> str:
    ticker = item["ticker"]
    name = ticker[:-4] if ticker.endswith("-USD") else ticker
    action = item["action"]
    side = "long" if action in LONG else "short"
    entry = float(item["entry_price"])
    sent = datetime.fromisoformat(item["sent_at"])
    hours = (datetime.now(UTC) - sent).total_seconds() / 3600
    lines = [
        f"<b>Signal alert: {name} moving against {action.lower()}</b>",
        f"{adverse:.1f}% against the {side} call in {hours:.0f}h",
        f"Entry {fmt_price(entry)} · now {fmt_price(price)} ({move:+.1f}%)",
        f"<i>Signal from {item['analysis_date']} run, sent {sent:%d %b %H:%M} UTC</i>",
    ]
    if ticker.endswith("-USD"):
        lines.append(f"If you took it: <code>/close {html.escape(name)}</code>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    step_pct = float(os.getenv("ALERT_STEP_PCT", "3"))
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not args.dry_run and not (token and chat_id):
        sys.exit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    table = boto3.resource("dynamodb").Table(os.getenv("SIGNALS_TABLE", "tradingagents-signals"))
    for item in table.scan()["Items"]:
        ticker, action = item["ticker"], item["action"]
        if action not in LONG | SHORT:
            continue  # HOLD / unparsed: no direction to go against
        price = live_price(ticker)
        if price is None:
            continue
        move = (price / float(item["entry_price"]) - 1) * 100
        adverse = -move if action in LONG else move
        step = int(adverse // step_pct) if adverse > 0 else 0
        already = int(item.get("alerted_step", 0))
        print(f"{ticker} {action}: {move:+.2f}% since signal, adverse step {step} (alerted {already})")
        if step <= already:
            continue

        msg = format_alert(item, price, move, adverse)
        if args.dry_run:
            print(msg)
            continue
        send_telegram(token, chat_id, msg)
        # Only bump if it's still the same prediction (a new signal may have landed meanwhile).
        with contextlib.suppress(table.meta.client.exceptions.ConditionalCheckFailedException):
            table.update_item(
                Key={"ticker": ticker},
                UpdateExpression="SET alerted_step = :s",
                ConditionExpression="sent_at = :t",
                ExpressionAttributeValues={":s": step, ":t": item["sent_at"]},
            )
        print(f"  alert sent for {ticker}")


if __name__ == "__main__":
    main()
