"""NAMA AI Trader v0.5.0 — educational market scanner for US and Saudi equities.
Not investment advice. Market data coverage/latency depends on Yahoo Finance.
"""
from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_VERSION = "0.5.0"
US_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA", "NFLX", "AVGO", "PLTR", "INTC", "MU", "QCOM", "CRM", "ADBE", "ORCL", "UBER", "SHOP", "ARM"]
SA_SYMBOLS = {
    "2222.SR": "أرامكو السعودية", "1120.SR": "مصرف الراجحي", "1180.SR": "البنك الأهلي السعودي",
    "2010.SR": "سابك", "1211.SR": "معادن", "7010.SR": "stc", "1010.SR": "الرياض", "1150.SR": "الإنماء",
    "2380.SR": "بترو رابغ", "4003.SR": "إكسترا", "4190.SR": "جرير", "7203.SR": "علم", "2082.SR": "أكوا باور",
    "2223.SR": "لوبريف", "8010.SR": "التعاونية", "4001.SR": "أسواق العثيم", "4300.SR": "دار الأركان", "5110.SR": "كهرباء السعودية",
}

st.set_page_config(page_title="NAMA AI Trader", page_icon="📈", layout="wide")
st.title("📈 NAMA AI Trader")
st.caption(f"الإصدار {APP_VERSION} · أداة تحليل تعليمية للأسهم الأمريكية والسعودية، وليست توصية استثمارية")

@st.cache_data(ttl=900, show_spinner=False)
def load_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    try:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns={c: str(c).title() for c in data.columns})
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
        data = data[needed].copy()
        for c in needed:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        return data.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_info_news(symbol: str) -> Tuple[dict, list]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info() or {}
    except Exception:
        info = {}
    try:
        raw_news = yf.Ticker(symbol).news or []
    except Exception:
        raw_news = []
    news = []
    for item in raw_news[:8]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title") or "خبر بدون عنوان"
        url = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else item.get("link")
        provider = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher", "")
        pub = content.get("pubDate") or item.get("providerPublishTime")
        news.append({"title": title, "url": url, "publisher": provider or "", "date": str(pub or "")})
    return info, news

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["Close"]
    x["EMA20"] = c.ewm(span=20, adjust=False).mean()
    x["EMA50"] = c.ewm(span=50, adjust=False).mean()
    x["EMA200"] = c.ewm(span=200, adjust=False).mean()
    x["RSI"] = rsi(c)
    x["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    x["MACD_Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    prev = x["Close"].shift(1)
    tr = pd.concat([(x["High"]-x["Low"]), (x["High"]-prev).abs(), (x["Low"]-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    x["ATR_Pct"] = x["ATR"] / c * 100
    x["VolAvg20"] = x["Volume"].rolling(20).mean() if "Volume" in x else np.nan
    return x

def analyze(symbol: str, df0: pd.DataFrame) -> Optional[dict]:
    if df0.empty or len(df0) < 60:
        return None
    df = indicators(df0).dropna(subset=["Close"])
    last = df.iloc[-1]
    price = float(last["Close"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.025
    if not np.isfinite(atr) or atr <= 0: atr = price * 0.025
    score, reasons = 0, []
    if price > last["EMA20"]: score += 12; reasons.append("السعر فوق متوسط 20")
    if last["EMA20"] > last["EMA50"]: score += 15; reasons.append("متوسط 20 أعلى من 50")
    if price > last["EMA200"]: score += 15; reasons.append("السعر فوق متوسط 200")
    if last["MACD"] > last["MACD_Signal"]: score += 15; reasons.append("زخم MACD إيجابي")
    if 48 <= last["RSI"] <= 68: score += 15; reasons.append("RSI ضمن نطاق زخم معتدل")
    elif 40 <= last["RSI"] < 48: score += 7; reasons.append("RSI يتحسن لكنه دون 48")
    if pd.notna(last.get("VolAvg20")) and last["VolAvg20"] > 0 and last["Volume"] > 1.2 * last["VolAvg20"]:
        score += 10; reasons.append("حجم تداول أعلى من متوسط 20 يومًا")
    change20 = price / float(df["Close"].iloc[-21]) - 1 if len(df) >= 21 else 0
    if change20 > 0.03: score += 10; reasons.append("أداء 20 جلسة إيجابي")
    if last["RSI"] > 75: score -= 15; reasons.append("خصم بسبب تشبع شرائي محتمل")
    if last["RSI"] < 30: score -= 5; reasons.append("RSI منخفض؛ احتمال تقلب مرتفع")
    score = int(max(0, min(100, score)))
    signal = "فرصة محتملة" if score >= 70 else "مراقبة" if score >= 52 else "ضعيفة حاليًا"
    return {"الرمز":symbol, "السعر":price, "التقييم":score, "الإشارة":signal, "RSI":float(last["RSI"]), "ATR %":float(last["ATR_Pct"]) if pd.notna(last["ATR_Pct"]) else np.nan,
            "EMA20":float(last["EMA20"]), "EMA50":float(last["EMA50"]), "EMA200":float(last["EMA200"]), "دخول مرجعي":price,
            "هدف مرجعي":price + 2*atr, "وقف مرجعي":price - 1.5*atr, "أسباب التقييم":"، ".join(reasons), "_df":df}

def backtest(df0: pd.DataFrame, threshold: int, horizon: int, fee_pct: float, slippage_pct: float) -> Tuple[pd.DataFrame, dict]:
    """Long-only signal test. Signal at close, enter next open; conservative stop-first if both hit."""
    df = indicators(df0).dropna(subset=["EMA50", "ATR", "RSI"]).copy()
    rows = []
    if len(df) < 80: return pd.DataFrame(), {"عدد الصفقات":0}
    costs = (fee_pct + slippage_pct) / 100
    # Keep trades non-overlapping to reduce dependence between simultaneous positions.
    i = 60
    while i < len(df) - 2:
        hist = df.iloc[:i+1]
        fake = analyze_from_row(hist)
        if fake is None or fake["score"] < threshold:
            i += 1; continue
        entry_i = i + 1
        entry = float(df["Open"].iloc[entry_i]) if pd.notna(df["Open"].iloc[entry_i]) else float(df["Close"].iloc[entry_i])
        atr = float(df["ATR"].iloc[i])
        if not np.isfinite(atr) or atr <= 0: i += 1; continue
        stop, target = entry - 1.5*atr, entry + 2*atr
        end_i = min(entry_i + horizon - 1, len(df)-1)
        exit_price, exit_i, reason = float(df["Close"].iloc[end_i]), end_i, "انتهاء المدة"
        for j in range(entry_i, end_i+1):
            lo, hi = float(df["Low"].iloc[j]), float(df["High"].iloc[j])
            if lo <= stop:
                exit_price, exit_i, reason = stop, j, "وقف الخسارة"; break
            if hi >= target:
                exit_price, exit_i, reason = target, j, "تحقق الهدف"; break
        ret = (exit_price / entry - 1) - costs
        rows.append({"تاريخ الدخول":str(df.index[entry_i].date()), "تاريخ الخروج":str(df.index[exit_i].date()), "الدخول":entry, "الخروج":exit_price, "العائد %":ret*100, "النتيجة":reason, "مدة الاحتفاظ":exit_i-entry_i+1})
        i = exit_i + 1
    trades = pd.DataFrame(rows)
    if trades.empty: return trades, {"عدد الصفقات":0,"نسبة الفوز %":0.0,"متوسط العائد %":0.0,"العائد التراكمي %":0.0,"أقصى تراجع %":0.0}
    rets = trades["العائد %"].to_numpy()/100
    equity = np.cumprod(1+rets)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    dd = equity/peak-1
    stats = {"عدد الصفقات":len(trades), "نسبة الفوز %":round(float((rets>0).mean()*100),2), "متوسط العائد %":round(float(trades["العائد %"].mean()),2),
             "العائد التراكمي %":round(float((equity[-1]-1)*100),2), "أقصى تراجع %":round(float(dd.min()*100),2)}
    return trades, stats

def analyze_from_row(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 60: return None
    r = df.iloc[-1]; price=float(r["Close"]); score=0
    if price > r["EMA20"]: score += 12
    if r["EMA20"] > r["EMA50"]: score += 15
    if price > r["EMA200"]: score += 15
    if r["MACD"] > r["MACD_Signal"]: score += 15
    if 48 <= r["RSI"] <= 68: score += 15
    elif 40 <= r["RSI"] < 48: score += 7
    if pd.notna(r.get("VolAvg20")) and r["VolAvg20"] > 0 and r["Volume"] > 1.2*r["VolAvg20"]: score += 10
    if len(df)>=21 and price/float(df["Close"].iloc[-21])-1 > .03: score += 10
    if r["RSI"]>75: score-=15
    if r["RSI"]<30: score-=5
    return {"score":max(0,min(100,int(score)))}

with st.sidebar:
    st.header("إعدادات الفحص")
    market = st.radio("السوق", ["الأمريكي", "السعودي"], horizontal=True)
    default_symbols = US_SYMBOLS if market == "الأمريكي" else list(SA_SYMBOLS.keys())
    symbol_text = st.text_area("الرموز (رمز في كل سطر)", value="\n".join(default_symbols[:10]), height=180, help="للسوق السعودي استخدم رمز Yahoo مثل 2222.SR")
    period = st.selectbox("الفترة التاريخية", ["6mo", "1y", "2y", "5y"], index=2)
    min_score = st.slider("الحد الأدنى لتقييم الفرصة", 0, 100, 68, 1)
    scan_clicked = st.button("🔎 فحص الفرص", type="primary", use_container_width=True)
    st.divider()
    st.caption("البيانات مقدمة عبر Yahoo Finance وقد تكون متأخرة أو غير متاحة لبعض الرموز.")

symbols = list(dict.fromkeys([s.strip().upper() for s in symbol_text.replace(",", "\n").splitlines() if s.strip()]))[:40]
if "alert_seen" not in st.session_state: st.session_state.alert_seen = {}
if scan_clicked:
    results=[]; frames={}
    progress = st.progress(0, text="جارٍ تحميل البيانات...")
    for idx, sym in enumerate(symbols):
        hist = load_history(sym, period=period)
        result = analyze(sym, hist)
        if result:
            frames[sym] = result.pop("_df")
            results.append(result)
        progress.progress((idx+1)/max(1,len(symbols)), text=f"فحص {sym} ({idx+1}/{len(symbols)})")
    progress.empty()
    results.sort(key=lambda z:z["التقييم"], reverse=True)
    st.session_state.scan_results = results
    st.session_state.scan_frames = frames
    st.session_state.scan_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

results = st.session_state.get("scan_results", [])
frames = st.session_state.get("scan_frames", {})
if results:
    st.subheader("نتائج اكتشاف الفرص")
    st.caption("آخر فحص: " + st.session_state.get("scan_time", ""))
    # In-site alerts: generated on scan only, never sent outside the website.
    new_alerts=[]
    for r in results:
        if r["التقييم"] >= min_score:
            key=f'{r["الرمز"]}:{r["التقييم"]//5}'
            if key not in st.session_state.alert_seen:
                new_alerts.append(r); st.session_state.alert_seen[key]=True
    if new_alerts:
        for r in new_alerts:
            st.info(f'🔔 تنبيه داخل الموقع: {r["الرمز"]} — التقييم {r["التقييم"]}/100 ({r["الإشارة"]})')
    table = pd.DataFrame([{k:v for k,v in r.items() if k!="أسباب التقييم"} for r in results])
    st.dataframe(table, use_container_width=True, hide_index=True)
    high = [r for r in results if r["التقييم"] >= min_score]
    st.metric("فرص تجاوزت الحد", len(high))
    st.download_button("تنزيل نتائج الفحص CSV", table.to_csv(index=False).encode("utf-8-sig"), file_name="nama_scan_results.csv", mime="text/csv")
    st.subheader("تفاصيل السهم")
    chosen = st.selectbox("اختر رمزًا لعرض الرسم والتحليل", [r["الرمز"] for r in results])
    selected = next(r for r in results if r["الرمز"] == chosen)
    detail_df = frames.get(chosen, pd.DataFrame())
    left,right = st.columns(2)
    with left:
        st.metric("السعر المعدل", f'{selected["السعر"]:,.2f}', f'التقييم {selected["التقييم"]}/100')
        st.write("**الإشارة:**", selected["الإشارة"])
        st.write("**أسباب التقييم:**", selected["أسباب التقييم"] or "لا توجد أسباب كافية")
        st.write(f'**دخول مرجعي:** {selected["دخول مرجعي"]:,.2f} · **هدف مرجعي:** {selected["هدف مرجعي"]:,.2f} · **وقف مرجعي:** {selected["وقف مرجعي"]:,.2f}')
    with right:
        if not detail_df.empty:
            st.line_chart(detail_df[["Close","EMA20","EMA50"]].tail(180), use_container_width=True)
            st.caption("المستويات المرجعية حسابية وتعليمية، وليست توصية شراء أو بيع.")
    st.subheader("الأخبار والبيانات الأساسية")
    with st.spinner("جارٍ جلب الأخبار والبيانات الأساسية..."):
        info, news = load_info_news(chosen)
    cols = st.columns(4)
    metrics = [("القيمة السوقية", info.get("marketCap")), ("مكرر الربحية", info.get("trailingPE")), ("مكرر الربحية المتوقع", info.get("forwardPE")), ("نمو الإيرادات", info.get("revenueGrowth"))]
    for col,(label,val) in zip(cols,metrics):
        if isinstance(val,(int,float)) and label=="القيمة السوقية": val=f"{val:,.0f}"
        elif isinstance(val,(int,float)) and "نمو" in label: val=f"{val*100:.1f}%"
        elif isinstance(val,(int,float)): val=f"{val:.2f}"
        col.metric(label, str(val) if val is not None else "غير متاح")
    st.write("**ملخص النشاط:**", info.get("longBusinessSummary", "لا يتوفر وصف أساسي من المصدر.") or "لا يتوفر وصف أساسي من المصدر.")
    if news:
        for item in news:
            if item["url"]: st.markdown(f'- [{item["title"]}]({item["url"]}) — {item["publisher"]}')
            else: st.markdown(f'- {item["title"]} — {item["publisher"]}')
    else: st.info("لم تُرجع خدمة البيانات أخبارًا لهذا الرمز حاليًا.")
    st.subheader("الاختبار التاريخي")
    c1,c2,c3,c4 = st.columns(4)
    with c1: horizon = st.selectbox("مدة الصفقة (جلسات)", [5,10,15,20], index=1)
    with c2: bt_threshold = st.slider("حد التقييم للاختبار", 0,100, min_score, 1)
    with c3: fee = st.number_input("العمولة الإجمالية لكل دخول/خروج %", min_value=0.0, max_value=2.0, value=0.15, step=0.05)
    with c4: slip = st.number_input("انزلاق سعري لكل صفقة %", min_value=0.0, max_value=2.0, value=0.10, step=0.05)
    if st.button("تشغيل الاختبار التاريخي", use_container_width=True):
        hist = load_history(chosen, period="5y")
        trades, stats = backtest(hist, bt_threshold, horizon, fee, slip)
        st.session_state.backtest_trades = trades
        st.session_state.backtest_stats = stats
        st.session_state.backtest_symbol = chosen
    if st.session_state.get("backtest_symbol")==chosen and "backtest_stats" in st.session_state:
        stats=st.session_state.backtest_stats
        cols=st.columns(5)
        for col,key in zip(cols,["عدد الصفقات","نسبة الفوز %","متوسط العائد %","العائد التراكمي %","أقصى تراجع %"]):
            col.metric(key, str(stats.get(key,"—")))
        trades=st.session_state.get("backtest_trades",pd.DataFrame())
        if not trades.empty:
            st.dataframe(trades,use_container_width=True,hide_index=True)
            st.download_button("تنزيل صفقات الاختبار CSV", trades.to_csv(index=False).encode("utf-8-sig"), file_name=f"nama_backtest_{chosen.replace('.','_')}.csv", mime="text/csv")
        else: st.warning("لم ينتج الاختبار صفقات كافية بالشروط المحددة؛ جرّب خفض الحد أو تغيير الفترة.")
else:
    st.info("اختر السوق والرموز من القائمة الجانبية ثم اضغط «فحص الفرص». يمكنك إدخال رمز واحد أو عدة رموز.")

with st.expander("كيف تقرأ النتائج وحدود الأداة"):
    st.markdown("""
- **التقييم** مزيج قواعد فنية معلنة (الاتجاه، المتوسطات، RSI، MACD، الحجم والزخم)، وليس نموذجًا يتنبأ بالسعر أو ضمانًا للربح.
- **الاختبار التاريخي** يقيّم إشارة عند إغلاق الجلسة ويفترض الدخول في افتتاح الجلسة التالية؛ يخرج عند الوقف أو الهدف أو نهاية المدة. إذا لمس السعر الوقف والهدف في الشمعة نفسها، يُحتسب الوقف أولًا بصورة محافظة.
- تُخصم نسبة العمولة والانزلاق التي تدخلها، لكن لا تُحاكى الضرائب أو جميع تكاليف التنفيذ أو السيولة أو تأثير حجم الصفقة. الأداء التاريخي لا يضمن المستقبل، وقد تتأثر النتائج بانحياز الاختيار.
- **التنبيهات داخل الموقع** تظهر عند تشغيل الفحص فقط، وتُحفظ حالة التكرار خلال جلسة المتصفح. لا توجد إشعارات خارج الموقع ولا مراقبة دائمة عندما يكون التطبيق مغلقًا.
- الأخبار والمؤشرات الأساسية تعتمد على توفر بيانات Yahoo Finance؛ قد تكون ناقصة أو غير محدثة، خصوصًا لبعض الأسهم السعودية. تحقّق من المصدر الرسمي قبل اتخاذ أي قرار.
    """)
st.caption("NAMA AI Trader — للاستخدام التعليمي والبحثي فقط. لا يقدم نصيحة مالية شخصية.")
