"""
Daily Technical Analysis Bot
Fetches price data, computes TA indicators, sends Telegram report.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────

PORTFOLIO = os.environ.get("PORTFOLIO", "AAPL,TSLA,2330.TW,2317.TW").split(",")
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # optional AI summary

# ── Indicator helpers ─────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)

def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    # Crossover detection
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        crossover = "Bullish crossover"
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        crossover = "Bearish crossover"
    elif macd_line.iloc[-1] > signal_line.iloc[-1]:
        crossover = "Bullish"
    else:
        crossover = "Bearish"
    return crossover, round(float(macd_line.iloc[-1]), 4), round(float(signal_line.iloc[-1]), 4)

def compute_bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    price = series.iloc[-1]
    u, l, m = float(upper.iloc[-1]), float(lower.iloc[-1]), float(sma.iloc[-1])
    pct = (price - l) / (u - l) * 100 if (u - l) != 0 else 50
    if pct >= 80:
        pos = f"Near upper band ({pct:.0f}%)"
    elif pct <= 20:
        pos = f"Near lower band ({pct:.0f}%)"
    else:
        pos = f"Middle ({pct:.0f}%)"
    return pos, round(u, 2), round(l, 2)

def compute_emas(series: pd.Series):
    ema20 = series.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = series.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = series.ewm(span=200, adjust=False).mean().iloc[-1]
    price = series.iloc[-1]
    return price > ema20, price > ema50, price > ema200

def volume_status(vol_series: pd.Series) -> str:
    avg = vol_series.rolling(20).mean().iloc[-1]
    last = vol_series.iloc[-1]
    ratio = last / avg if avg else 1
    if ratio >= 1.5:
        return f"High ({ratio:.1f}x avg)"
    elif ratio <= 0.7:
        return f"Low ({ratio:.1f}x avg)"
    else:
        return f"Normal ({ratio:.1f}x avg)"

def overall_signal(rsi, macd_str, ema20_above, ema50_above, bb_pos):
    bull = 0
    bear = 0
    if rsi < 35:
        bull += 2
    elif rsi > 65:
        bear += 2
    if "Bullish" in macd_str:
        bull += 2
    elif "Bearish" in macd_str:
        bear += 2
    if ema20_above:
        bull += 1
    else:
        bear += 1
    if ema50_above:
        bull += 1
    else:
        bear += 1
    if "lower" in bb_pos.lower():
        bull += 1
    elif "upper" in bb_pos.lower():
        bear += 1
    if bull >= 4:
        return "BUY"
    elif bear >= 4:
        return "SELL"
    else:
        return "HOLD"

# ── Fetch & analyse ───────────────────────────────────────────────────────────

def analyse_ticker(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if hist.empty or len(hist) < 50:
            return {"ticker": ticker, "error": "Insufficient data"}

        close = hist["Close"]
        volume = hist["Volume"]

        price = round(float(close.iloc[-1]), 2)
        prev = round(float(close.iloc[-2]), 2)
        change_pct = round((price - prev) / prev * 100, 2)
        change_str = f"+{change_pct}%" if change_pct >= 0 else f"{change_pct}%"

        rsi = compute_rsi(close)
        macd_str, macd_val, signal_val = compute_macd(close)
        bb_pos, bb_upper, bb_lower = compute_bollinger(close)
        ema20_above, ema50_above, ema200_above = compute_emas(close)
        vol_status = volume_status(volume)
        signal = overall_signal(rsi, macd_str, ema20_above, ema50_above, bb_pos)

        rsi_label = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"

        return {
            "ticker": ticker,
            "price": price,
            "change": change_str,
            "direction": "up" if change_pct >= 0 else "down",
            "rsi": rsi,
            "rsi_label": rsi_label,
            "macd": macd_str,
            "ema20_above": ema20_above,
            "ema50_above": ema50_above,
            "ema200_above": ema200_above,
            "bb_position": bb_pos,
            "volume": vol_status,
            "signal": signal,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ── AI narrative (optional) ───────────────────────────────────────────────────

def get_ai_summary(results: list) -> str:
    if not ANTHROPIC_API_KEY:
        return ""
    try:
        data_str = json.dumps(results, indent=2)
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 400,
                "messages": [{
                    "role": "user",
                    "content": f"""You are a concise technical analyst. Given this portfolio TA data, write a 3-4 sentence market overview for today. Highlight any notable signals, risks, or opportunities. Be direct and professional. No bullet points.

Data:
{data_str}"""
                }]
            },
            timeout=30
        )
        body = resp.json()
        return body["content"][0]["text"].strip()
    except Exception:
        return ""

# ── Telegram sender ───────────────────────────────────────────────────────────

SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
DIR_EMOJI = {"up": "▲", "down": "▼"}

def build_message(results: list, ai_summary: str) -> str:
    today = date.today().strftime("%b %d, %Y")
    lines = [f"📊 *Daily TA Report — {today}*\n"]

    if ai_summary:
        lines.append(f"_{ai_summary}_\n")

    for r in results:
        t = r["ticker"]
        if "error" in r:
            lines.append(f"*{t}* — ⚠️ {r['error']}\n")
            continue

        sig = r["signal"]
        emoji = SIGNAL_EMOJI.get(sig, "⚪")
        dir_e = DIR_EMOJI.get(r["direction"], "")
        ema_str = " ".join([
            "EMA20✓" if r["ema20_above"] else "EMA20✗",
            "EMA50✓" if r["ema50_above"] else "EMA50✗",
            "EMA200✓" if r["ema200_above"] else "EMA200✗",
        ])

        lines.append(
            f"{emoji} *{t}* — {sig}\n"
            f"  Price: `{r['price']}` {dir_e} {r['change']}\n"
            f"  RSI: `{r['rsi']}` ({r['rsi_label']})\n"
            f"  MACD: {r['macd']}\n"
            f"  BB: {r['bb_position']}\n"
            f"  Vol: {r['volume']}\n"
            f"  {ema_str}\n"
        )

    lines.append("_Signals: 🟢 BUY  🟡 HOLD  🔴 SELL_")
    return "\n".join(lines)

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }, timeout=15)
    resp.raise_for_status()
    print(f"Telegram sent: {resp.status_code}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Analysing portfolio: {PORTFOLIO}")
    results = [analyse_ticker(t.strip()) for t in PORTFOLIO]

    for r in results:
        print(json.dumps(r, indent=2, default=str))

    ai_summary = get_ai_summary([r for r in results if "error" not in r])
    message = build_message(results, ai_summary)
    print("\n── Telegram message ──")
    print(message)
    send_telegram(message)
    print("Done.")

if __name__ == "__main__":
    main()
