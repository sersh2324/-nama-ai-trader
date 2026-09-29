import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from services.scanner import scan_symbols
from services.data_alpha_vantage import get_daily_data

load_dotenv()
st.set_page_config(page_title='NAMA AI Trader', page_icon='📈', layout='wide')
st.title('📈 NAMA AI Trader')
st.caption('MVP تجريبي لتحليل الفرص — ليس توصية استثمارية ولا ضمانًا للربح.')

symbols_text = st.sidebar.text_area('الأسهم الأمريكية', 'AAPL,NVDA,TSLA,MSFT,AMZN')
symbols = [x.strip().upper() for x in symbols_text.split(',') if x.strip()]

if st.sidebar.button('🔎 فحص الفرص', type='primary'):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    rows = []
    for symbol in symbols:
        try:
            df = get_daily_data(symbol, api_key)
            result = scan_symbols({symbol: df})[0]
            rows.append(result)
        except Exception as e:
            rows.append({'symbol': symbol, 'score': 0, 'signal': 'خطأ', 'price': None,
                         'entry': None, 'target': None, 'stop': None, 'reasons': str(e)})
    st.session_state['results'] = pd.DataFrame(rows)

if 'results' in st.session_state:
    st.subheader('الفرص المكتشفة')
    st.dataframe(st.session_state['results'], use_container_width=True, hide_index=True)
else:
    st.write('أدخل الرموز ثم اضغط «فحص الفرص».')
