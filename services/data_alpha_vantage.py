import requests
import pandas as pd
BASE_URL='https://www.alphavantage.co/query'

def get_daily_data(symbol, api_key):
    if not api_key: raise ValueError('ضع ALPHA_VANTAGE_API_KEY في ملف .env')
    r=requests.get(BASE_URL,params={'function':'TIME_SERIES_DAILY','symbol':symbol,'outputsize':'compact','apikey':api_key},timeout=20)
    r.raise_for_status(); payload=r.json(); key='Time Series (Daily)'
    if key not in payload: raise ValueError(payload.get('Note') or payload.get('Information') or str(payload))
    df=pd.DataFrame.from_dict(payload[key],orient='index'); df.index=pd.to_datetime(df.index)
    return df.rename(columns={'1. open':'open','2. high':'high','3. low':'low','4. close':'close','5. volume':'volume'}).sort_index()
