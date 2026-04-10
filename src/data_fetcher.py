import pandas as pd
import functools
import ta
import requests
import yfinance as yf


# -----------------------------
# 🔎 SEARCH
# -----------------------------
@functools.lru_cache(maxsize=32)
def search_ticker(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
    except Exception as e:
        print(f"Search Error: {e}")

    return None


# -----------------------------
# 🧠 PRIMARY DATA SOURCE (FAST)
# -----------------------------
def fetch_from_stooq(ticker):
    try:
        df = pd.read_csv(f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d")

        if df.empty:
            return pd.DataFrame()

        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 🔥 LIMIT DATA (IMPORTANT)
        df = df.tail(90)  # last ~3 months

        return df

    except Exception as e:
        print("Stooq Error:", e)
        return pd.DataFrame()


# -----------------------------
# 🔁 FALLBACK (YFINANCE)
# -----------------------------
def fetch_from_yfinance(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="3mo", interval="1d", auto_adjust=True)

        if df is None or df.empty:
            return pd.DataFrame(), None

        df.columns = [col.lower() for col in df.columns]

        return df.tail(90), ticker_obj  # 🔥 limit rows

    except Exception as e:
        print("YFinance Error:", e)
        return pd.DataFrame(), None


# -----------------------------
# 🚀 MAIN FUNCTION
# -----------------------------
@functools.lru_cache(maxsize=32)
def get_alpha_live_data(ticker):
    resolved_ticker = ticker

    # 1️⃣ Try STOOQ
    df = fetch_from_stooq(ticker)
    ticker_obj = None

    # 2️⃣ Fallback to Yahoo
    if df.empty:
        print("Falling back to yfinance...")
        df, ticker_obj = fetch_from_yfinance(ticker)

        if df.empty:
            found = search_ticker(ticker)
            if found and found != ticker:
                resolved_ticker = found
                df, ticker_obj = fetch_from_yfinance(resolved_ticker)

    if df.empty:
        return pd.DataFrame(), {}, ticker

    # -----------------------------
    # 🧠 FEATURE ENGINEERING (LIGHT)
    # -----------------------------
    try:
        df = df.copy()

        # RSI
        df['RSI_14'] = ta.momentum.RSIIndicator(
            close=df['close'], window=14
        ).rsi()

        # MACD (light version)
        macd = ta.trend.MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        # SMA (only one)
        df['SMA_50'] = ta.trend.SMAIndicator(
            close=df['close'], window=50
        ).sma_indicator()

    except Exception as e:
        print("Feature Error:", e)
        return pd.DataFrame(), {}, ticker

    # -----------------------------
    # 📊 KPI FETCH (SAFE)
    # -----------------------------
    kpis = {}

    if ticker_obj:
        try:
            info = ticker_obj.fast_info  # 🔥 faster than .info
            kpis = {
                "Last Price": info.get("last_price", "N/A"),
                "Market Cap": info.get("market_cap", "N/A")
            }
        except:
            pass

    # Final cleanup
    df = df.dropna()

    return df, kpis, resolved_ticker