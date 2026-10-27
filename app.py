from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

VERSION = "0.6.1"
US = ["AAPL","MSFT","NVDA","AMD","META","GOOGL","AMZN","TSLA","NFLX","AVGO","PLTR","INTC","MU","QCOM","CRM","ADBE","ORCL","UBER","SHOP","ARM"]
US_NAMES = {
    "AAPL":"Apple — آبل", "MSFT":"Microsoft — مايكروسوفت", "NVDA":"NVIDIA — إنفيديا",
    "AMD":"Advanced Micro Devices — إيه إم دي", "META":"Meta Platforms — ميتا",
    "GOOGL":"Alphabet (Google) — ألفابت/جوجل", "AMZN":"Amazon — أمازون", "TSLA":"Tesla — تسلا",
    "NFLX":"Netflix — نتفلكس", "AVGO":"Broadcom — برودكوم", "PLTR":"Palantir — بالانتير",
    "INTC":"Intel — إنتل", "MU":"Micron Technology — ميكرون", "QCOM":"Qualcomm — كوالكوم",
    "CRM":"Salesforce — سيلزفورس", "ADBE":"Adobe — أدوبي", "ORCL":"Oracle — أوراكل",
    "UBER":"Uber — أوبر", "SHOP":"Shopify — شوبيفاي", "ARM":"Arm Holdings — آرم"
}
SA = {"2222.SR":"أرامكو السعودية","1120.SR":"مصرف الراجحي","1180.SR":"الأهلي السعودي","2010.SR":"سابك","1211.SR":"معادن","7010.SR":"stc","1010.SR":"بنك الرياض","1150.SR":"مصرف الإنماء","4190.SR":"جرير","7203.SR":"علم","2082.SR":"أكوا باور","8010.SR":"التعاونية","4001.SR":"أسواق العثيم","5110.SR":"السعودية للكهرباء"}

def company_name(symbol):
    return SA.get(symbol, US_NAMES.get(symbol, symbol))

def display_symbol(symbol):
    return f"{company_name(symbol)} ({symbol})"

st.set_page_config(page_title="NAMA AI Trader", page_icon="📈", layout="wide")
st.title("📈 NAMA AI Trader")
st.caption(f"الإصدار {VERSION} — تحليل تعليمي للسوقين الأمريكي والسعودي، وليس توصية استثمارية")

@st.cache_data(ttl=900, show_spinner=False)
def history(symbol, period="2y"):
    try:
        d = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if d is None or d.empty: return pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d = d.rename(columns={str(c):str(c).title() for c in d.columns})
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in d: d[c] = np.nan
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return d[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"]).sort_index()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals_news(symbol):
    try: info = yf.Ticker(symbol).get_info() or {}
    except Exception: info = {}
    try: raw = yf.Ticker(symbol).news or []
    except Exception: raw = []
    news=[]
    for n in raw[:10]:
        c=n.get("content",n) if isinstance(n,dict) else {}
        if not isinstance(c,dict): c=n
        u=c.get("canonicalUrl")
        url=u.get("url") if isinstance(u,dict) else n.get("link")
        p=c.get("provider")
        provider=p.get("displayName","") if isinstance(p,dict) else n.get("publisher","")
        date=c.get("pubDate") or n.get("providerPublishTime") or ""
        if isinstance(date,(int,float)):
            try: date=datetime.fromtimestamp(date,tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception: pass
        news.append({"العنوان":c.get("title") or n.get("title") or "خبر بدون عنوان","الرابط":url,"المصدر":provider,"التاريخ":str(date)})
    return info,news

def indicators(d):
    x=d.copy(); c=x.Close
    x["EMA20"]=c.ewm(span=20,adjust=False).mean()
    x["EMA50"]=c.ewm(span=50,adjust=False).mean()
    x["EMA200"]=c.ewm(span=200,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    x["RSI"]=(100-100/(1+gain/loss.replace(0,np.nan))).fillna(50)
    x["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    x["MACDsig"]=x.MACD.ewm(span=9,adjust=False).mean()
    prev=c.shift(1)
    tr=pd.concat([x.High-x.Low,(x.High-prev).abs(),(x.Low-prev).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14,min_periods=14).mean()
    x["ATR%"]=x.ATR/c.replace(0,np.nan)*100
    x["VolAvg20"]=x.Volume.rolling(20,min_periods=20).mean()
    x["Ret20"]=c.pct_change(20)*100
    x["High20"]=x.High.rolling(20,min_periods=20).max().shift(1)
    return x

def score(r, price):
    s=0; reasons=[]
    rules=[(price>r.EMA20,10,"السعر فوق متوسط 20"),(r.EMA20>r.EMA50,12,"الاتجاه القصير إيجابي"),
           (price>r.EMA200,12,"السعر فوق متوسط 200"),(r.MACD>r.MACDsig,12,"زخم MACD إيجابي"),
           (48<=r.RSI<=68,12,"RSI ضمن نطاق معتدل"),
           (pd.notna(r.Ret20) and r.Ret20>3,10,"عائد 20 جلسة إيجابي"),
           (pd.notna(r.High20) and price>r.High20,12,"اختراق أعلى 20 جلسة")]
    for ok,pts,why in rules:
        if ok: s+=pts; reasons.append(why)
    if 40<=r.RSI<48: s+=6; reasons.append("RSI منخفض نسبيًا")
    if pd.notna(r.VolAvg20) and r.VolAvg20>0 and pd.notna(r.Volume) and r.Volume>1.2*r.VolAvg20:
        s+=10; reasons.append("حجم تداول أعلى من المتوسط")
    if r.RSI>75: s-=15; reasons.append("تحذير تشبع شرائي")
    if r.RSI<30: s-=5; reasons.append("تحذير RSI منخفض")
    if pd.notna(r["ATR%"]) and r["ATR%"]>8: s-=8; reasons.append("تذبذب مرتفع")
    s=int(np.clip(s,0,100))
    setup="اختراق محتمل" if pd.notna(r.High20) and price>r.High20 and s>=65 else ("اتجاه صاعد" if price>r.EMA20>r.EMA50 else "متابعة فنية")
    return s,reasons,setup

def analyze(sym, raw, style):
    if raw.empty or len(raw)<60: return None
    d=indicators(raw); r=d.iloc[-1]; p=float(r.Close)
    if not np.isfinite(p) or not np.isfinite(r.EMA50): return None
    atr=float(r.ATR) if pd.notna(r.ATR) and r.ATR>0 else p*.025
    sm,tm=(1,1.4) if style=="مضاربة خلال الجلسة" else ((2.5,4) if style=="استثمار طويل الأجل" else (1.5,2))
    stop=max(0,p-sm*atr); target=p+tm*atr
    s,why,setup=score(r,p)
    return {"الرمز":sym,"السعر":p,"التقييم":s,"الإشارة":"فرصة محتملة" if s>=70 else ("مراقبة" if s>=52 else "ضعيفة حاليًا"),
            "نوع الفرصة":setup,"RSI":float(r.RSI),"ATR %":float(r["ATR%"]) if pd.notna(r["ATR%"]) else np.nan,
            "EMA20":float(r.EMA20),"EMA50":float(r.EMA50),"EMA200":float(r.EMA200),
            "دخول مرجعي":p,"هدف مرجعي":target,"وقف مرجعي":stop,"العائد/المخاطرة":(target-p)/(p-stop) if p>stop else np.nan,
            "أسباب التقييم":"، ".join(why),"_df":d}

def backtest(raw, threshold, horizon, fees, slip, sm, tm):
    empty={"عدد الصفقات":0,"نسبة الفوز %":0,"متوسط العائد %":0,"العائد التراكمي %":0,"أقصى تراجع %":0,"عامل الربح":0,"متوسط مدة الصفقة":0}
    if raw.empty or len(raw)<100: return pd.DataFrame(),empty
    d=indicators(raw).dropna(subset=["EMA50","ATR","RSI","Open","High","Low","Close"])
    rows=[]; i=60
    while i<len(d)-2:
        r=d.iloc[i]; s,why,setup=score(r,float(r.Close))
        if s<threshold: i+=1; continue
        ei=i+1; entry=float(d.Open.iloc[ei]); atr=float(d.ATR.iloc[i])
        if not np.isfinite(entry) or entry<=0 or not np.isfinite(atr) or atr<=0: i+=1; continue
        stop=max(0,entry-sm*atr); target=entry+tm*atr
        end=min(ei+horizon-1,len(d)-1); exitp=float(d.Close.iloc[end]); xi=end; reason="انتهاء المدة"
        for j in range(ei,end+1):
            op,lo,hi=float(d.Open.iloc[j]),float(d.Low.iloc[j]),float(d.High.iloc[j])
            if op<=stop: exitp,xi,reason=op,j,"وقف بعد فجوة"; break
            if op>=target: exitp,xi,reason=target,j,"هدف بعد فجوة"; break
            if lo<=stop: exitp,xi,reason=stop,j,"وقف الخسارة"; break
            if hi>=target: exitp,xi,reason=target,j,"تحقق الهدف"; break
        ret=(exitp/entry-1)-(fees+slip)/100
        rows.append({"تاريخ الإشارة":str(d.index[i].date()),"تاريخ الدخول":str(d.index[ei].date()),
                     "تاريخ الخروج":str(d.index[xi].date()),"التقييم":s,"نوع الفرصة":setup,
                     "الدخول":entry,"الخروج":exitp,"العائد الصافي %":ret*100,"سبب الخروج":reason,"مدة الاحتفاظ":xi-ei+1})
        i=xi+1
    t=pd.DataFrame(rows)
    if t.empty: return t,empty
    rets=t["العائد الصافي %"].to_numpy()/100; eq=np.cumprod(1+rets); curve=np.r_[1,eq]
    dd=curve/np.maximum.accumulate(curve)-1; wins=rets[rets>0]; losses=rets[rets<0]
    pf=wins.sum()/abs(losses.sum()) if len(losses) and abs(losses.sum()) else (float("inf") if len(wins) else 0)
    stats={"عدد الصفقات":len(t),"نسبة الفوز %":round(float((rets>0).mean()*100),2),
           "متوسط العائد %":round(float(rets.mean()*100),2),"العائد التراكمي %":round(float((eq[-1]-1)*100),2),
           "أقصى تراجع %":round(float(dd.min()*100),2),"عامل الربح":round(float(pf),2) if np.isfinite(pf) else "∞",
           "متوسط مدة الصفقة":round(float(t["مدة الاحتفاظ"].mean()),1)}
    return t,stats

with st.sidebar:
    st.header("إعدادات التحليل")
    market=st.radio("السوق",["الأمريكي","السعودي","السوقان معًا"])
    style=st.selectbox("أسلوب التداول",["مضاربة خلال الجلسة","سوينغ","استثمار طويل الأجل"],index=1)
    period=st.selectbox("الفترة",["6mo","1y","2y","5y"],index=2)
    base=US if market=="الأمريكي" else list(SA) if market=="السعودي" else US[:10]+list(SA)[:10]
    st.caption("ابحث باسم الشركة أو رمز السهم، ثم اختر الأسهم التي تريد فحصها.")
    chosen_symbols=st.multiselect(
        "البحث والاختيار باسم الشركة أو الرمز",
        options=base,
        default=base[:20],
        format_func=display_symbol,
        max_selections=50,
        help="ابدأ بكتابة اسم الشركة بالعربية أو الإنجليزية أو رمز السهم مثل AAPL أو 2222.SR."
    )
    symbols_text=st.text_area("إضافة رموز يدويًا (اختياري، رمز في كل سطر)","",height=90,
                              help="يمكنك إدخال أي رموز أخرى غير موجودة في القائمة، مثل رمز سهم جديد.")
    min_score=st.slider("الحد الأدنى للتقييم",0,100,68)
    scan=st.button("🔎 فحص الفرص",type="primary",use_container_width=True)

manual_symbols=[x.strip().upper() for x in symbols_text.replace(",","\n").splitlines() if x.strip()]
symbols=list(dict.fromkeys(chosen_symbols+manual_symbols))[:50]
for key,default in [("results",[]),("frames",{}),("watch",[]),("alerts",[]),
                    ("portfolio",pd.DataFrame(columns=["التاريخ","السوق","الرمز","النوع","الكمية","سعر العملية","سعر الخروج","ملاحظات"]))]:
    if key not in st.session_state: st.session_state[key]=default
if scan:
    rs=[]; fm={}; prog=st.progress(0)
    for i,sym in enumerate(symbols):
        a=analyze(sym,history(sym,period),style)
        if a:
            fm[sym]=a.pop("_df")
            a["اسم الشركة"]=company_name(sym)
            rs.append(a)
        prog.progress((i+1)/max(1,len(symbols)))
    prog.empty(); rs.sort(key=lambda x:x["التقييم"],reverse=True)
    st.session_state.results=rs; st.session_state.frames=fm
    st.session_state.scan_time=datetime.now().strftime("%Y-%m-%d %H:%M")
    for r in rs:
        if r["التقييم"]>=min_score:
            k=(r["الرمز"],r["التقييم"]//5,r["نوع الفرصة"])
            if not any(x["key"]==k for x in st.session_state.alerts):
                st.session_state.alerts.insert(0,{"الوقت":st.session_state.scan_time,"الرمز":r["الرمز"],"التقييم":r["التقييم"],"النوع":r["نوع الفرصة"],"key":k})
    st.session_state.alerts=st.session_state.alerts[:100]
results,frames=st.session_state.results,st.session_state.frames
tabs=st.tabs(["🔎 اكتشاف الفرص","🎯 خطة الصفقة","💼 المحفظة والسجل","🧪 الاختبار التاريخي","📰 الأخبار والأساسيات","🔔 التنبيهات"])
with tabs[0]:
    if results:
        st.caption("آخر فحص: "+st.session_state.get("scan_time",""))
        for r in results:
            if r["التقييم"]>=min_score: st.info(f'🔔 {company_name(r["الرمز"])} ({r["الرمز"]}): {r["التقييم"]}/100 — {r["نوع الفرصة"]}')
        table=pd.DataFrame([{k:v for k,v in r.items() if k!="أسباب التقييم"} for r in results])
        st.dataframe(table,use_container_width=True,hide_index=True)
        c=st.columns(3); c[0].metric("الأسهم المفحوصة",len(results)); c[1].metric("تجاوزت الحد",sum(x["التقييم"]>=min_score for x in results)); c[2].metric("متوسط التقييم",round(np.mean([x["التقييم"] for x in results]),1))
        st.download_button("تنزيل نتائج الفحص CSV",table.to_csv(index=False).encode("utf-8-sig"),"nama_scan_results.csv","text/csv")
        chosen=st.selectbox("تفاصيل السهم",[x["الرمز"] for x in results],format_func=display_symbol); r=next(x for x in results if x["الرمز"]==chosen)
        st.write("**الإشارة:**",r["الإشارة"],"· **الأسباب:**",r["أسباب التقييم"] or "لا توجد")
        st.line_chart(frames[chosen][["Close","EMA20","EMA50","EMA200"]].tail(180))
        if st.button("➕ أضف إلى قائمة المراقبة"):
            if chosen not in st.session_state.watch: st.session_state.watch.append(chosen)
            st.success("أضيفت قائمة المراقبة.")
    else: st.info("اختر السوق والأسلوب ثم اضغط فحص الفرص.")
with tabs[1]:
    st.subheader("خطة الصفقة وحجم المركز")
    sym=st.selectbox("السهم",[x["الرمز"] for x in results] if results else symbols,key="plan",format_func=display_symbol)
    r=next((x for x in results if x["الرمز"]==sym),None)
    e0,s0,t0=(r["دخول مرجعي"],r["وقف مرجعي"],r["هدف مرجعي"]) if r else (100,95,110)
    a,b,c=st.columns(3)
    capital=a.number_input("رأس المال",min_value=0.0,value=10000.0,step=500.0)
    riskpct=a.number_input("المخاطرة %",min_value=.1,max_value=10.0,value=1.0,step=.1)
    entry=b.number_input("الدخول",min_value=.0001,value=max(float(e0),.0001),format="%.4f")
    stop=b.number_input("وقف الخسارة",min_value=0.0,value=max(float(s0),0.0),format="%.4f")
    target=c.number_input("الهدف",min_value=0.0,value=max(float(t0),0.0),format="%.4f")
    alloc=c.number_input("أقصى تخصيص %",min_value=1.0,max_value=100.0,value=20.0)
    if entry>stop and capital>0:
        qty=max(0,min(math.floor(capital*riskpct/100/(entry-stop)),math.floor(capital*alloc/100/entry)))
        m=st.columns(4); m[0].metric("الكمية التقديرية",f"{qty:,}"); m[1].metric("قيمة الصفقة",f"{qty*entry:,.2f}")
        m[2].metric("الخسارة عند الوقف",f"{qty*(entry-stop):,.2f}"); m[3].metric("الربح عند الهدف",f"{qty*max(0,target-entry):,.2f}")
        st.metric("العائد/المخاطرة",f"{(target-entry)/(entry-stop):.2f}" if target>entry else "غير مناسب")
    else: st.warning("تحقق من أن الدخول أعلى من الوقف ورأس المال أكبر من صفر.")
    st.caption("الحساب تقديري ولا يشمل جميع الرسوم أو فجوات الأسعار.")
with tabs[2]:
    st.subheader("سجل المحفظة والصفقات")
    st.caption("السجل محفوظ في جلسة التطبيق الحالية فقط؛ نزّل CSV للاحتفاظ به.")
    with st.form("portfolio_form",clear_on_submit=True):
        a,b,c=st.columns(3)
        date=a.date_input("التاريخ",value=datetime.now().date()); mkt=a.selectbox("السوق",["الأمريكي","السعودي"]); sym=a.text_input("رمز السهم").strip().upper()
        typ=b.selectbox("نوع العملية",["شراء / فتح","بيع / إغلاق","توزيعات","ملاحظة"]); qty=b.number_input("الكمية",min_value=0.0,value=0.0,step=1.0)
        price=c.number_input("سعر العملية",min_value=0.0,value=0.0,step=.1); exitp=c.number_input("سعر الخروج اختياري",min_value=0.0,value=0.0,step=.1); note=st.text_input("ملاحظات")
        submit=st.form_submit_button("إضافة إلى السجل")
        if submit:
            if not sym: st.error("أدخل رمز السهم.")
            else:
                st.session_state.portfolio=pd.concat([st.session_state.portfolio,pd.DataFrame([{"التاريخ":str(date),"السوق":mkt,"الرمز":sym,"النوع":typ,"الكمية":qty,"سعر العملية":price,"سعر الخروج":exitp or np.nan,"ملاحظات":note}])],ignore_index=True)
                st.success("تمت الإضافة لهذه الجلسة.")
    p=st.session_state.portfolio
    if not p.empty:
        st.dataframe(p,use_container_width=True,hide_index=True)
        st.download_button("تنزيل سجل المحفظة CSV",p.to_csv(index=False).encode("utf-8-sig"),"nama_portfolio.csv","text/csv")
        if st.button("مسح السجل"): st.session_state.portfolio=p.iloc[0:0].copy(); st.rerun()
    else: st.info("لا توجد عمليات مسجلة.")
with tabs[3]:
    st.subheader("الاختبار التاريخي")
    sym=st.selectbox("رمز الاختبار",[x["الرمز"] for x in results] if results else symbols,key="bt",format_func=display_symbol)
    a,b,c=st.columns(3); bp=a.selectbox("الفترة",["1y","2y","5y"],index=2); horizon=a.selectbox("أقصى مدة احتفاظ", [5,10,15,20,40],index=1)
    threshold=b.slider("حد التقييم",0,100,min_score); fees=b.number_input("تكاليف الدخول والخروج %",0.0,3.0,.15,.05)
    slip=c.number_input("الانزلاق السعري الإجمالي %",0.0,3.0,.10,.05); bst=c.selectbox("أسلوب الخروج",["سوينغ","مضاربة خلال الجلسة","استثمار طويل الأجل"])
    sm,tm=(1,1.4) if bst=="مضاربة خلال الجلسة" else ((2.5,4) if bst=="استثمار طويل الأجل" else (1.5,2))
    if st.button("تشغيل الاختبار"):
        t,stats=backtest(history(sym,bp),threshold,horizon,fees,slip,sm,tm); st.session_state.bt_result=(sym,t,stats)
    if st.session_state.get("bt_result") and st.session_state.bt_result[0]==sym:
        _,t,stats=st.session_state.bt_result
        m=st.columns(4)
        for col,key in zip(m,["عدد الصفقات","نسبة الفوز %","العائد التراكمي %","أقصى تراجع %"]): col.metric(key,str(stats[key]))
        m=st.columns(3)
        for col,key in zip(m,["متوسط العائد %","عامل الربح","متوسط مدة الصفقة"]): col.metric(key,str(stats[key]))
        if not t.empty:
            st.dataframe(t,use_container_width=True,hide_index=True)
            st.download_button("تنزيل الاختبار CSV",t.to_csv(index=False).encode("utf-8-sig"),f"nama_backtest_{sym.replace('.','_')}.csv","text/csv")
        else: st.warning("لا توجد صفقات وفق هذه الشروط.")
    st.markdown("**المنهجية:** الإشارة عند إغلاق الجلسة والدخول في افتتاح الجلسة التالية؛ لا توجد صفقات متداخلة على الرمز نفسه. إذا لمس الوقف والهدف في شمعة واحدة يُفترض الوقف أولًا. التكاليف تقديرية والنتائج لا تضمن الأداء المستقبلي.")
with tabs[4]:
    sym=st.selectbox("سهم الأخبار والبيانات الأساسية",[x["الرمز"] for x in results] if results else symbols,key="fund",format_func=display_symbol)
    if st.button("تحميل الأخبار والبيانات الأساسية"):
        info,news=fundamentals_news(sym); st.session_state.fund_result=(sym,info,news)
    if st.session_state.get("fund_result") and st.session_state.fund_result[0]==sym:
        _,info,news=st.session_state.fund_result
        metrics=[("القيمة السوقية",info.get("marketCap")),("مكرر الربحية",info.get("trailingPE")),("مكرر متوقع",info.get("forwardPE")),("نمو الإيرادات",info.get("revenueGrowth")),("هامش الربح",info.get("profitMargins")),("عائد التوزيعات",info.get("dividendYield"))]
        cols=st.columns(3)
        for col,(label,val) in zip(cols,metrics):
            if isinstance(val,(int,float)):
                val=f"{val*100:.2f}%" if label in ("نمو الإيرادات","هامش الربح","عائد التوزيعات") else (f"{val:,.0f}" if label=="القيمة السوقية" else f"{val:.2f}")
            col.metric(label,str(val) if val is not None else "غير متاح")
        st.write("**وصف النشاط:**",info.get("longBusinessSummary") or "غير متاح.")
        for n in news:
            st.markdown(f'- [{n["العنوان"]}]({n["الرابط"]}) — {n["المصدر"]} · {n["التاريخ"]}' if n["الرابط"] else f'- {n["العنوان"]} — {n["المصدر"]} · {n["التاريخ"]}')
        if not news: st.info("لا توجد أخبار متاحة من المصدر حاليًا.")
        st.caption("تحقق من الإفصاحات الرسمية؛ قد تكون البيانات متأخرة أو غير متاحة لبعض الأسهم السعودية.")
    else: st.info("اختر سهمًا واضغط تحميل.")
with tabs[5]:
    st.subheader("التنبيهات داخل الموقع")
    st.caption("تظهر التنبيهات عند تشغيل الفحص فقط؛ لا توجد مراقبة في الخلفية عند إغلاق الموقع.")
    alerts=[{k:v for k,v in x.items() if k!="key"} for x in st.session_state.alerts]
    if alerts:
        adf=pd.DataFrame(alerts); st.dataframe(adf,use_container_width=True,hide_index=True)
        st.download_button("تنزيل التنبيهات CSV",adf.to_csv(index=False).encode("utf-8-sig"),"nama_alerts.csv","text/csv")
        if st.button("مسح التنبيهات"): st.session_state.alerts=[]; st.rerun()
    else: st.info("لا توجد تنبيهات بعد.")
with st.expander("قائمة المراقبة وحدود النسخة"):
    st.write("قائمة المراقبة:",", ".join(display_symbol(x) for x in st.session_state.watch) if st.session_state.watch else "فارغة")
    if st.session_state.watch and st.button("مسح قائمة المراقبة"): st.session_state.watch=[]; st.rerun()
    st.markdown("- بيانات الأسعار يومية؛ خيار المضاربة يغير مضاعفات الوقف والهدف فقط ولا يعني بيانات لحظية.\n- سجل المحفظة والمراقبة والتنبيهات محفوظة في الجلسة الحالية فقط؛ نزّل CSV للاحتفاظ بنسخة.\n- تغطية Yahoo Finance تختلف حسب الرمز والسوق، خصوصًا الأخبار والبيانات الأساسية للسوق السعودي.\n- التحليل تعليمي وليس توصية مالية أو ضمانًا للربح.")
st.caption("NAMA AI Trader — للاستخدام التعليمي والبحثي فقط.")
