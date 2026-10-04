import os, time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from services.scanner import analyze
from services.data_alpha_vantage import get_daily_data

load_dotenv()

st.set_page_config(page_title="NAMA AI Trader", page_icon="📈", layout="wide")
st.title("📈 NAMA AI Trader")
st.caption("v0.3 — تحليل فني + اختبار تاريخي أولي. النتائج احتمالية وليست توصية استثمارية أو ضمانًا للربح.")

try:
    API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
except Exception:
    API_KEY = ""
API_KEY = API_KEY or os.getenv("ALPHA_VANTAGE_API_KEY", "")

@st.cache_data(ttl=900, show_spinner=False)
def cached_daily_data(symbol, api_key):
    return get_daily_data(symbol, api_key)


def repair_text(value):
    if not isinstance(value, str):
        return value
    if any(marker in value for marker in ("Ã", "Â", "Ø", "Ù", "Ú", "Û")):
        try:
            return value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
    return value


def format_result(r):
    return {
        "الرمز": r["symbol"],
        "الدرجة": r["score"],
        "الإشارة": repair_text(r["signal"]),
        "السعر": r["price"],
        "الدخول المقترح": r["entry"],
        "الهدف": r["target"],
        "وقف الخسارة": r["stop"],
        "RSI": r["rsi"],
        "ATR %": r["atr_pct"],
        "الاتجاه": repair_text(r["trend"]),
        "الأسباب": repair_text(r["reasons"]),
    }


def backtest_symbol(df, horizon=10, min_score=68):
    """Walk-forward test using only information available at each signal day.
    Long signals are evaluated against the next `horizon` trading days.
    If target and stop are both touched on the same candle, outcome is ambiguous.
    """
    df = df.copy().sort_index()
    if len(df) < 70:
        raise ValueError("نحتاج بيانات تاريخية كافية للاختبار (يفضل 70 شمعة أو أكثر).")

    trades = []
    start = 60
    end = len(df) - horizon - 1

    for i in range(start, end + 1):
        hist = df.iloc[: i + 1]
        try:
            sig = analyze(hist)
        except Exception:
            continue

        if sig["signal"] != "صعود محتمل" or sig["score"] < min_score:
            continue

        entry = float(sig["entry"])
        target = float(sig["target"])
        stop = float(sig["stop"])
        future = df.iloc[i + 1 : i + 1 + horizon]

        outcome = "لم يتحقق"
        exit_price = float(future["close"].iloc[-1])
        exit_date = future.index[-1]
        bars_to_exit = horizon

        for j, (idx, row) in enumerate(future.iterrows(), start=1):
            hit_target = float(row["high"]) >= target
            hit_stop = float(row["low"]) <= stop
            if hit_target and hit_stop:
                outcome = "ملتبس (الهدف والوقف في نفس الشمعة)"
                exit_price = stop
                exit_date = idx
                bars_to_exit = j
                break
            if hit_target:
                outcome = "هدف"
                exit_price = target
                exit_date = idx
                bars_to_exit = j
                break
            if hit_stop:
                outcome = "وقف"
                exit_price = stop
                exit_date = idx
                bars_to_exit = j
                break

        ret_pct = ((exit_price - entry) / entry) * 100 if entry else 0
        trades.append({
            "تاريخ الإشارة": hist.index[-1].date(),
            "الدرجة": sig["score"],
            "الدخول": round(entry, 2),
            "الهدف": round(target, 2),
            "الوقف": round(stop, 2),
            "النتيجة": outcome,
            "العائد %": round(ret_pct, 2),
            "أيام الصفقة": bars_to_exit,
            "تاريخ الخروج": exit_date.date(),
        })

    result_df = pd.DataFrame(trades)
    if result_df.empty:
        return result_df, {
            "signals": 0,
            "targets": 0,
            "stops": 0,
            "unresolved": 0,
            "ambiguous": 0,
            "win_rate": None,
            "avg_return": None,
            "total_return": None,
        }

    targets = int((result_df["النتيجة"] == "هدف").sum())
    stops = int((result_df["النتيجة"] == "وقف").sum())
    ambiguous = int(result_df["النتيجة"].str.startswith("ملتبس").sum())
    unresolved = int((result_df["النتيجة"] == "لم يتحقق").sum())
    decided = targets + stops
    win_rate = (targets / decided * 100) if decided else None

    stats = {
        "signals": len(result_df),
        "targets": targets,
        "stops": stops,
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "win_rate": round(win_rate, 1) if win_rate is not None else None,
        "avg_return": round(float(result_df["العائد %"].mean()), 2),
        "total_return": round(float(result_df["العائد %"].sum()), 2),
    }
    return result_df, stats


with st.sidebar:
    st.header("⚙️ الإعداد")
    symbols_text = st.text_area("رموز الأسهم الأمريكية", "AAPL,NVDA,MSFT")
    symbols = list(dict.fromkeys(x.strip().upper() for x in symbols_text.split(",") if x.strip()))
    st.info("الحساب المجاني لـ Alpha Vantage محدود. استخدم 1–3 أسهم ولا تكرر الفحص بسرعة.")

    horizon = st.selectbox("مدة اختبار الصفقة", [5, 10, 15], index=1)
    min_score = st.slider("أقل درجة لاختبار الإشارة", 68, 90, 68)

    run_scan = st.button("🔎 فحص الفرص", type="primary", use_container_width=True)
    run_backtest = st.button("🧪 تشغيل الاختبار التاريخي", use_container_width=True)

if run_scan:
    if not API_KEY:
        st.error("مفتاح Alpha Vantage غير موجود. أضفه في Streamlit Secrets باسم ALPHA_VANTAGE_API_KEY.")
        st.stop()
    if not symbols:
        st.warning("أدخل رمز سهم واحد على الأقل.")
        st.stop()
    if len(symbols) > 3:
        st.warning("تم تقليص القائمة إلى أول 3 أسهم لتقليل استهلاك API.")
        symbols = symbols[:3]

    rows = []
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        try:
            if i > 0:
                time.sleep(1.2)
            df = cached_daily_data(symbol, API_KEY)
            result = analyze(df)
            result["symbol"] = symbol
            rows.append(format_result(result))
        except Exception as e:
            rows.append({
                "الرمز": symbol, "الدرجة": 0, "الإشارة": "لم يتم التحليل",
                "السعر": None, "الدخول المقترح": None, "الهدف": None,
                "وقف الخسارة": None, "RSI": None, "ATR %": None,
                "الاتجاه": "-", "الأسباب": repair_text(str(e))
            })
        progress.progress((i + 1) / len(symbols))
    st.session_state["results"] = pd.DataFrame(rows)

if run_backtest:
    if not API_KEY:
        st.error("مفتاح Alpha Vantage غير موجود. أضفه في Streamlit Secrets باسم ALPHA_VANTAGE_API_KEY.")
        st.stop()
    if not symbols:
        st.warning("أدخل رمز سهم واحد على الأقل.")
        st.stop()
    if len(symbols) > 3:
        st.warning("تم تقليص القائمة إلى أول 3 أسهم لتقليل استهلاك API.")
        symbols = symbols[:3]

    st.subheader("🧪 الاختبار التاريخي الأولي")
    st.caption(
        f"نختبر إشارات صعود بدرجة {min_score}+ ونرى ماذا حدث خلال {horizon} جلسات لاحقة. "
        "الاختبار أولي ومحدود بتاريخ البيانات المتاح من المصدر، وليس ضمانًا للأداء المستقبلي."
    )

    summary_rows = []
    all_trades = []
    progress = st.progress(0)

    for i, symbol in enumerate(symbols):
        try:
            if i > 0:
                time.sleep(1.2)
            df = cached_daily_data(symbol, API_KEY)
            trades, stats = backtest_symbol(df, horizon=horizon, min_score=min_score)
            summary_rows.append({
                "الرمز": symbol,
                "الإشارات": stats["signals"],
                "الهدف": stats["targets"],
                "الوقف": stats["stops"],
                "لم يتحقق": stats["unresolved"],
                "ملتبس": stats["ambiguous"],
                "نسبة النجاح %": stats["win_rate"],
                "متوسط العائد %": stats["avg_return"],
                "مجموع العوائد %": stats["total_return"],
            })
            if not trades.empty:
                trades.insert(0, "الرمز", symbol)
                all_trades.append(trades)
        except Exception as e:
            summary_rows.append({
                "الرمز": symbol, "الإشارات": 0, "الهدف": 0, "الوقف": 0,
                "لم يتحقق": 0, "ملتبس": 0, "نسبة النجاح %": None,
                "متوسط العائد %": None, "مجموع العوائد %": None,
            })
            st.warning(f"{symbol}: {repair_text(str(e))}")
        progress.progress((i + 1) / len(symbols))

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if all_trades:
        st.subheader("📋 تفاصيل الإشارات التي تم اختبارها")
        trades_df = pd.concat(all_trades, ignore_index=True)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("لم تظهر إشارات صعود بالدرجة المحددة ضمن البيانات المتاحة. جرّب درجة أقل أو سهمًا آخر.")

    st.warning(
        "مهم: هذا Backtest أولي. لا يشمل عمولات التنفيذ، الانزلاق السعري، فجوات الافتتاح، "
        "ولا يمثل اختبارًا استثماريًا احترافيًا. قبل استخدام النظام تجاريًا سنبني محرك اختبار أدق."
    )

if "results" in st.session_state and not run_backtest:
    st.subheader("📊 نتائج الفحص")
    st.dataframe(st.session_state["results"], use_container_width=True, hide_index=True)
    st.caption("البيانات ليست لحظية؛ تعتمد على البيانات اليومية المتاحة من Alpha Vantage.")
    st.subheader("🧠 كيف حُسبت الإشارة؟")
    st.write(
        "الدرجة تجمع الاتجاه عبر EMA20/50/200، الزخم عبر RSI وMACD، التذبذب عبر ATR، "
        "البولينجر والحجم. الهدف ووقف الخسارة حسابات آلية للتجربة وليست توصية."
    )

if not run_scan and not run_backtest and "results" not in st.session_state:
    st.markdown("""
### 🚀 ابدأ
1. اكتب رموز الأسهم مثل `AAPL,NVDA,MSFT`.
2. اضغط **فحص الفرص** لرؤية الإشارة الحالية.
3. اضغط **تشغيل الاختبار التاريخي** لمعرفة كيف كان أداء الإشارات تاريخيًا ضمن البيانات المتاحة.

**المرحلة التالية:** تحسين الاختبار، ثم الأخبار، السوق السعودي، والعقود/الخيارات.
""")
