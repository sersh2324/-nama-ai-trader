import requests
import pandas as pd

BASE_URL = "https://www.alphavantage.co/query"

def get_daily_data(symbol, api_key):
    if not api_key:
        raise ValueError("مفتاح Alpha Vantage غير موجود.")

    r = requests.get(
        BASE_URL,
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": api_key,
        },
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()

    if "Note" in payload:
        raise ValueError("Alpha Vantage: تم الوصول إلى حد الطلبات. انتظر ثم أعد المحاولة، ولا تكرر الفحص عدة مرات.")
    if "Information" in payload:
        raise ValueError("Alpha Vantage: " + str(payload["Information"]))

    key = "Time Series (Daily)"
    if key not in payload:
        raise ValueError(
            payload.get("Error Message")
            or "لم تصل بيانات يومية لهذا الرمز. تأكد من صحة رمز السهم."
        )

    df = pd.DataFrame.from_dict(payload[key], orient="index")
    df.index = pd.to_datetime(df.index)

    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume",
    }).sort_index()

    return df
