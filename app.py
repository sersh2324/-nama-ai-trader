import os, time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from services.scanner import analyze
from services.data_alpha_vantage import get_daily_data

load_dotenv()

st.set_page_config(page_title="NAMA AI Trader", page_icon="📈", layout="wide")
st.title("📈 NAMA AI Trader")
st.caption("v0.4 — تحليل فني + Backtest متقدم + جودة تاريخية. النتائج احتمالية وليست توصية استثمارية أو ضمانًا للربح.")

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


def empty_stats():
    return {
        "signals": 0, "targets": 0, "stops": 0, "unresolved": 0,
        "ambiguous": 0, "win_rate": None, "avg_return": None,
        "total_return": None, "avg_win": None, "avg_loss": None,
        "profit_factor": None, "expectancy": None, "quality": None,
        "decided": 0,
    }


def calculate_quality(result_df):
    if result_df.empty:
        return None

    decided_df = result_df[result_df["النتيجة"].isin(["هدف", "وقف"])].copy()
    if decided_df.empty:
        return None

    wins = int((decided_df["النتيجة"] == "هدف").sum())
    losses = int((decided_df["النتيجة"] == "وقف").sum())
    decided = wins + losses
    win_rate = wins / decided * 100

    wins_returns = decided_df.loc[decided_df["النتيجة"] == "هدف", "العائد %"]
    loss_returns = decided_df.loc[decided_df["النتيجة"] == "وقف", "العائد %"]
    avg_win = float(wins_returns.mean()) if not wins_returns.empty else 0.0
    avg_loss = float(loss_returns.mean()) if not loss_returns.empty else 0.0

    gross_profit = float(wins_returns.sum())
    gross_loss = abs(float(loss_returns.sum()))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    expectancy = ((wins / decided) * avg_win) + ((losses / decided) * avg_loss)

    # Conservative historical quality score: small samples are shrunk toward 50%.
    adjusted_win = ((wins + 10) / (decided + 20)) * 100
    pf_component = 50.0 if profit_factor is None else min(100.0, (profit_factor / (profit_factor + 1)) * 100)
    exp_component = min(100.0, max(0.0, 50 + expectancy * 12))
    quality = round(0.45 * adjusted_win + 0.30 * pf_component + 0.25 * exp_component)
    quality = int(max(0, min(100, quality)))

    return {
        "decided": decided,
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "expectancy": round(expectancy, 2),
        "quality": quality,
    }


def backtest_symbol(df, horizon=10, min_score=68):
    """Walk-forward test using only information available on each signal day.
    Target/stop are checked on future candles. If both are touched on the same
    candle, the trade is marked ambiguous and excluded from win-rate metrics.
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

        if sig["score"] < min_score or sig["signal"] != "صعود محتمل":
            continue

        entry = float(sig["entry"])
        target = float(sig["target"])
        stop = float(sig["stop"])
        future = df.iloc[i + 1 : i + 1 + horizon]
        if future.empty:
            continue

        outcome = "لم يتحقق"
        exit_price = float(future["close"].iloc[-1])
        exit_date = future.index[-1]
        bars_to_exit = horizon

        for j, (idx, row) in enumerate(future.iterrows(), start=1):
            hit_target = float(row["high"]) >= target
            hit_stop = float(row["low"]) <= stop
            if hit_target and hit_stop:
                outcome = "ملتبس"
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
        return result_df, empty_stats()

    targets = int((result_df["النتيجة"] == "هدف").sum())
    stops = int((result_df["النتيجة"] == "وقف").sum())
    ambiguous = int((result_df["النتيجة"] == "ملتبس").sum())
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
        "decided": decided,
        "avg_win": None,
        "avg_loss": None,
        "profit_factor": None,
        "expectancy": None,
        "quality": None,
    }

    quality = calculate_quality(result_df)
    if quality:
        stats.update(quality)
    return result_df, stats


def bucket_quality(all_trades):
    if not all_trades:
        return pd.DataFrame()
    df = pd.concat(all_trades, ignore_index=True)
    bins = [49, 59, 69, 79, 89, 100]
    labels = ["50–59", "60–69", "70–79", "80–89", "90–100"]
    df["فئة الدرجة"] = pd.cut(df["الدرجة"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for label in labels:
        x = df[df["فئة الدرجة"] == label]
        if x.empty:
            continue
        decided = x[x["النتيجة"].isin(["هدف", "وقف"])]
        wins = int((decided["النتيجة"] == "هدف").sum())
        losses = int((decided["النتيجة"] == "وقف").sum())
        n = wins + losses
        avg_return = float(x["العائد %"].mean())
        pf = None
        if losses:
            gp = float(decided.loc[decided["النتيجة"] == "هدف", "العائد %"].sum())
            gl = abs(float(decided.loc[decided["النتيجة"] == "وقف", "العائد %"].sum()))
            if gl > 0:
                pf = gp / gl
        rows.append({
            "فئة الدرجة": label,
            "الإشارات": len(x),
            "المحسومة": n,
            "الأهداف": wins,
            "الوقف": losses,
            "نسبة النجاح %": round(wins / n * 100, 1) if n else None,
            "متوسط العائد %": round(avg_return, 2),
            "Profit Factor": round(pf, 2) if pf is not None else None,
        })
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("⚙️ الإعداد")
    symbols_text = st.text_area("رموز الأسهم الأمريكية", "AAPL,NVDA,MSFT")
    symbols = list(dict.fromkeys(x.strip().upper() for x in symbols_text.split(",") if x.strip()))
    st.info("الحساب المجاني لـ Alpha Vantage محدود. استخدم 1–3 أسهم ولا تكرر الفحص بسرعة.")
    horizon = st.selectbox("مدة اختبار الصفقة", [5, 10, 15], index=1)
    min_score = st.slider("أقل درجة لاختبار صفقة شراء", 50, 90, 68)
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


# عرض نتائج الفحص الحالي بعد الضغط على زر فحص الفرص
if "results" in st.session_state and isinstance(st.session_state["results"], pd.DataFrame):
    results_df = st.session_state["results"]
    if not results_df.empty:
        st.subheader("📊 نتائج الفحص")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

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

    st.subheader("🧪 الاختبار التاريخي المتقدم")
    st.caption(
        f"نختبر فرص شراء بدرجة {min_score}+ ونرى ماذا حدث خلال {horizon} جلسات لاحقة. "
        "الهدف من v0.4 هو قياس جودة الإشارة، وليس إثبات الربحية المستقبلية."
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
                "المحسومة": stats["decided"],
                "الهدف": stats["targets"],
                "الوقف": stats["stops"],
                "لم يتحقق": stats["unresolved"],
                "ملتبس": stats["ambiguous"],
                "نسبة النجاح %": stats["win_rate"],
                "متوسط الربح %": stats["avg_win"],
                "متوسط الخسارة %": stats["avg_loss"],
                "Profit Factor": stats["profit_factor"],
                "التوقع لكل صفقة %": stats["expectancy"],
                "جودة تاريخية /100": stats["quality"],
                "متوسط العائد %": stats["avg_return"],
                "مجموع العوائد %": stats["total_return"],
            })
            if not trades.empty:
                trades.insert(0, "الرمز", symbol)
                all_trades.append(trades)
        except Exception as e:
            summary_rows.append({
                "الرمز": symbol, "الإشارات": 0, "المحسومة": 0, "الهدف": 0, "الوقف": 0,
                "لم يتحقق": 0, "ملتبس": 0, "نسبة النجاح %": None,
                "متوسط الربح %": None, "متوسط الخسارة %": None, "Profit Factor": None,
                "التوقع لكل صفقة %": None, "جودة تاريخية /100": None,
                "متوسط العائد %": None, "مجموع العوائد %": None,
            })
            st.warning(f"{symbol}: {repair_text(str(e))}")
        progress.progress((i + 1) / len(symbols))

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if all_trades:
        bucket_df = bucket_quality(all_trades)
        st.subheader("📊 جودة الإشارة حسب الدرجة")
        if not bucket_df.empty:
            st.dataframe(bucket_df, use_container_width=True, hide_index=True)
            st.info("هذه المقارنة تساعدنا على معرفة هل الدرجات الأعلى كانت أفضل تاريخيًا داخل العينة المتاحة. لا تعني أن النتيجة ستتكرر مستقبلًا.")

        st.subheader("📋 تفاصيل الإشارات التي تم اختبارها")
        trades_df = pd.concat(all_trades, ignore_index=True)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("لم تظهر فرص بالدرجة المحددة ضمن البيانات المتاحة. جرّب درجة أقل أو سهمًا آخر.")

    st.warning(
        "مهم: هذا Backtest أولي. لا يشمل عمولات التنفيذ أو الانزلاق السعري أو تفاصيل تنفيذ الأوامر، "
        "والصفقات قد تتداخل زمنيًا. كما أن العينة محدودة بتاريخ البيانات المتاح من المصدر. "
        "لا تستخدم جودة تاريخية كضمان للربح."
    )
