import pandas as pd
import numpy as np

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close):
    fast = ema(close, 12)
    slow = ema(close, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist

def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def analyze(df):
    df = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["high", "low", "close", "volume"]).sort_index()

    if len(df) < 60:
        raise ValueError("ÙØ­ØªØ§Ø¬ Ø¨ÙØ§ÙØ§Øª ÙÙÙÙØ© ÙØ§ÙÙØ© (ÙÙØ¶Ù 60 Ø´ÙØ¹Ø© Ø£Ù Ø£ÙØ«Ø±) ÙÙØªØ­ÙÙÙ.")

    close = df["close"]
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    rs = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)
    atr_v = atr(df, 14)

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    vol_avg = df["volume"].rolling(20).mean()

    price = float(close.iloc[-1])
    rsi_now = float(rs.iloc[-1])
    atr_now = float(atr_v.iloc[-1])
    atr_pct = (atr_now / price) * 100 if price else 0

    score = 50
    reasons = []

    # Trend
    if price > e20.iloc[-1]:
        score += 7
        reasons.append("Ø§ÙØ³Ø¹Ø± ÙÙÙ EMA20")
    else:
        score -= 7
        reasons.append("Ø§ÙØ³Ø¹Ø± ØªØ­Øª EMA20")

    if e20.iloc[-1] > e50.iloc[-1]:
        score += 10
        reasons.append("EMA20 ÙÙÙ EMA50")
    else:
        score -= 10
        reasons.append("EMA20 ØªØ­Øª EMA50")

    if price > e200.iloc[-1]:
        score += 8
        reasons.append("Ø§ÙØ³Ø¹Ø± ÙÙÙ EMA200")
    else:
        score -= 8
        reasons.append("Ø§ÙØ³Ø¹Ø± ØªØ­Øª EMA200")

    # RSI
    if 52 <= rsi_now <= 68:
        score += 8
        reasons.append("RSI ÙÙ ÙØ·Ø§Ù Ø²Ø®Ù Ø¥ÙØ¬Ø§Ø¨Ù")
    elif rsi_now < 42:
        score -= 8
        reasons.append("RSI Ø¶Ø¹ÙÙ")
    elif rsi_now > 75:
        score -= 6
        reasons.append("RSI ÙØ±ØªÙØ¹ ÙÙØ¯ ÙØ¹ÙÙ ØªØ´Ø¨Ø¹ÙØ§")

    # MACD
    if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_hist.iloc[-1] > 0:
        score += 8
        reasons.append("MACD Ø¥ÙØ¬Ø§Ø¨Ù")
    elif macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_hist.iloc[-1] < 0:
        score -= 8
        reasons.append("MACD Ø³ÙØ¨Ù")

    # Bollinger position
    if price > upper.iloc[-1]:
        score -= 3
        reasons.append("Ø§ÙØ³Ø¹Ø± Ø£Ø¹ÙÙ ÙÙ Ø§ÙØ­Ø¯ Ø§ÙØ¹ÙÙÙ ÙÙØ¨ÙÙÙÙØ¬Ø±")
    elif price < lower.iloc[-1]:
        score += 2
        reasons.append("Ø§ÙØ³Ø¹Ø± ÙØ±Ø¨ Ø§ÙØ­Ø¯ Ø§ÙØ³ÙÙÙ ÙÙØ¨ÙÙÙÙØ¬Ø±")

    # Volume
    avg_vol = vol_avg.iloc[-1]
    if pd.notna(avg_vol) and avg_vol > 0 and df["volume"].iloc[-1] > avg_vol * 1.3:
        score += 7
        reasons.append("Ø§ÙØ­Ø¬Ù Ø£Ø¹ÙÙ ÙÙ ÙØªÙØ³Ø· 20 ÙÙÙ")

    score = int(max(0, min(100, round(score))))

    # Signal is intentionally conservative: neutral zone is wide.
    if score >= 68:
        signal = "ØµØ¹ÙØ¯ ÙØ­ØªÙÙ"
        entry = price
        target = price + 2.5 * atr_now
        stop = price - 1.5 * atr_now
    elif score <= 32:
        signal = "ÙØ¨ÙØ· ÙØ­ØªÙÙ"
        entry = price
        target = price - 2.5 * atr_now
        stop = price + 1.5 * atr_now
    else:
        signal = "ÙØ­Ø§ÙØ¯"
        entry = price
        target = price + 1.5 * atr_now if score >= 50 else price - 1.5 * atr_now
        stop = price - atr_now if score >= 50 else price + atr_now

    trend = "ØµØ§Ø¹Ø¯" if e20.iloc[-1] > e50.iloc[-1] else "ÙØ§Ø¨Ø·"

    return {
        "score": score,
        "signal": signal,
        "price": round(price, 2),
        "entry": round(entry, 2),
        "target": round(target, 2),
        "stop": round(stop, 2),
        "rsi": round(rsi_now, 1),
        "atr_pct": round(atr_pct, 2),
        "trend": trend,
        "reasons": " | ".join(reasons),
    }

def scan_symbols(data):
    out = []
    for symbol, df in data.items():
        x = analyze(df)
        x["symbol"] = symbol
        out.append(x)
    return sorted(out, key=lambda x: x["score"], reverse=True)
