import pandas as pd
import numpy as np

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def analyze(df):
    df = df.copy()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close'])
    if len(df) < 30:
        raise ValueError('نحتاج 30 شمعة يومية على الأقل.')
    close = df.close
    e20, e50, rs = ema(close,20), ema(close,50), rsi(close,14)
    score, reasons = 50, []
    if close.iloc[-1] > e20.iloc[-1]: score += 10; reasons.append('السعر فوق EMA20')
    else: score -= 10; reasons.append('السعر تحت EMA20')
    if e20.iloc[-1] > e50.iloc[-1]: score += 15; reasons.append('EMA20 فوق EMA50')
    else: score -= 15; reasons.append('EMA20 تحت EMA50')
    if 55 <= rs.iloc[-1] <= 70: score += 10; reasons.append('RSI يدعم الزخم')
    elif rs.iloc[-1] < 45: score -= 10; reasons.append('RSI ضعيف')
    elif rs.iloc[-1] > 75: score -= 5; reasons.append('RSI مرتفع جدًا')
    av = df.volume.rolling(20).mean().iloc[-1]
    if av and df.volume.iloc[-1] > av * 1.3: score += 10; reasons.append('حجم أعلى من متوسط 20 يوم')
    score = int(max(0,min(100,score)))
    price = float(close.iloc[-1])
    if score >= 65:
        signal='صعود محتمل'; entry=price; target=price*1.04; stop=price*0.97
    elif score <= 35:
        signal='هبوط محتمل'; entry=price; target=price*0.96; stop=price*1.03
    else:
        signal='محايد'; entry=target=stop=price
    return {'score':score,'signal':signal,'price':round(price,2),'entry':round(entry,2),
            'target':round(target,2),'stop':round(stop,2),'reasons':' | '.join(reasons)}

def scan_symbols(data):
    out=[]
    for symbol, df in data.items():
        x=analyze(df); x['symbol']=symbol; out.append(x)
    return sorted(out,key=lambda x:x['score'],reverse=True)
