import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from services.scanner import analyze
from services.data_alpha_vantage import get_daily_data

load_dotenv()

st.set_page_config(page_title="NAMA AI Trader", page_icon="📈", layout="wide")

st.title("📈 NAMA AI Trader")
st.caption("v0.2 — تحليل فني تجريبي للأسهم الأمريكية. النتائج احتمالية وليست توصية استثمارية أو ضمانًا للربح.")

# Secrets first, then local .env for local testing
try:
    API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
except Exception:
    API_KEY = ""
API_KEY = API_KEY or os.getenv("ALPHA_VANTAGE_API_KEY", "")

with st.sidebar:
    st.header("⚙️ إعداد الفحص")
    symbols_text = st.text_area(
        "رموز الأسهم الأمريكية",
        "AAPL,NVDA,MSFT",
        help="للحساب المجاني في Alpha Vantage ابدأ بـ 1–3 أسهم فقط."
    )
    symbols = list(dict.fromkeys(
        x.strip().upper() for x in symbols_text.split(",") if x.strip()
    ))
    st.info("Alpha Vantage المجاني محدود بعدد الطلبات اليومية. لا تضغط الفحص بشكل متكرر.")
    run = st.button("🔎 فحص الفرص", type="primary", use_container_width=True)

@st.cache_data(ttl=900, show_spinner=False)
def cached_daily_data(symbol, api_key):
    return get_daily_data(symbol, api_key)

def format_result(r):
    return {
        "الرمز": r["symbol"],
        "الدرجة": r["score"],
        "الإشارة": r["signal"],
        "السعر": r["price"],
        "الدخول المقترح": r["entry"],
        "الهدف": r["target"],
        "وقف الخسارة": r["stop"],
        "RSI": r["rsi"],
        "ATR %": r["atr_pct"],
        "الاتجاه": r["trend"],
        "الأسباب": r["reasons"],
    }

if run:
    if not API_KEY:
        st.error("مفتاح Alpha Vantage غير موجود. أضفه في Streamlit Secrets باسم ALPHA_VANTAGE_API_KEY.")
        st.stop()

    if not symbols:
        st.warning("أدخل رمز سهم واحد على الأقل.")
        st.stop()

    if len(symbols) > 3:
        st.warning("في النسخة المجانية نوصي بفحص 3 أسهم أو أقل في كل مرة لتقليل استهلاك الطلبات.")
        symbols = symbols[:3]

    rows = []
    progress = st.progress(0)

    for i, symbol in enumerate(symbols):
        try:
            # Space requests to respect the provider's rate guidance.
            if i > 0:
                time.sleep(1.2)

            df = cached_daily_data(symbol, API_KEY)
            result = analyze(df)
            result["symbol"] = symbol
            rows.append(format_result(result))

        except Exception as e:
            rows.append({
                "الرمز": symbol,
                "الدرجة": 0,
                "الإشارة": "لم يتم التحليل",
                "السعر": None,
                "الدخول المقترح": None,
                "الهدف": None,
                "وقف الخسارة": None,
                "RSI": None,
                "ATR %": None,
                "الاتجاه": "-",
                "الأسباب": str(e),
            })
        progress.progress((i + 1) / len(symbols))

    st.session_state["results"] = pd.DataFrame(rows)

if "results" in st.session_state:
    df_results = st.session_state["results"]

    st.subheader("📊 نتائج الفحص")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.caption(
        "ملاحظة: البيانات تعتمد على مصدر Alpha Vantage المتاح لحسابك. "
        "النسخة الحالية ليست بيانات سوق لحظية."
    )

    st.subheader("🧠 كيف حُسبت الإشارة؟")
    st.write(
        "الدرجة تجمع عدة عوامل فنية: الاتجاه عبر EMA20/50/200، "
        "الزخم عبر RSI وMACD، التذبذب عبر ATR، "
        "البولينجر، والحجم. الهدف ووقف الخسارة حسابات آلية للتجربة وليست توصية."
    )
else:
    st.markdown("""
### ابدأ بهذه الخطوة
1. اكتب رموز الأسهم مثل `AAPL, NVDA, MSFT`.
2. اضغط **فحص الفرص**.
3. راجع الدرجة والإشارة والدخول والهدف ووقف الخسارة والأسباب.

**المرحلة القادمة:** إضافة فلترة أقوى للفرص، دعم السوق السعودي، ثم بناء وحدة للعقود/الخيارات ومقارنة النتائج بالاختبارات التاريخية.
""")
