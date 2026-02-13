import yfinance as yf
import pandas as pd
import functools
import ta
import requests

@functools.lru_cache(maxsize=32)
def search_ticker(query):
    """Search for a ticker symbol using Yahoo Finance API."""
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

@functools.lru_cache(maxsize=32) # Simple in-memory cache
def get_alpha_live_data(ticker):
    """Fetches real-time price data and calculates 17 technical indicators."""
    
    # Resolve ticker (Handle "Bitcoin" -> "BTC-USD")
    resolved_ticker = ticker
    
    # 1. Fetch deep historical data
    try:
        # Try exact ticker first
        ticker_obj = yf.Ticker(resolved_ticker)
        df = ticker_obj.history(period="max", interval="1d", auto_adjust=True) # Changed from 2y to max
        
        # If empty, try searching
        if df.empty:
            found = search_ticker(ticker)
            if found and found != ticker:
                print(f"Resolving '{ticker}' to '{found}'")
                resolved_ticker = found
                ticker_obj = yf.Ticker(resolved_ticker)
                df = ticker_obj.history(period="max", interval="1d", auto_adjust=True)
                
    except Exception as e:
        print(f"YF Error: {e}")
        return pd.DataFrame(), {}, ticker
    
    if df.empty:
         return pd.DataFrame(), {}, ticker

    # Standardize columns for consistency
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    # 2. Advanced Feature Engineering (The Alpha Factors)
    # Replicating pandas-ta logic using 'ta' library
    
    try:
        # RSI (14)
        df['RSI_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD_12_26_9'] = macd.macd()
        df['MACDh_12_26_9'] = macd.macd_diff()
        df['MACDs_12_26_9'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BBL_20_2.0'] = bb.bollinger_lband()
        df['BBM_20_2.0'] = bb.bollinger_mavg()
        df['BBU_20_2.0'] = bb.bollinger_hband()
        df['BBP_20_2.0'] = bb.bollinger_pband()
        df['BBD_20_2.0'] = bb.bollinger_wband()

        # ATR
        df['ATRr_14'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

        # SMA
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()
        
        # 3. Fetch Fundamental KPIs (AlphaFeed Style)
        info = ticker_obj.info # Use the resolved ticker object!
        kpis = {
            "Market Cap": info.get("marketCap", "N/A"),
            "52W High": info.get("fiftyTwoWeekHigh", "N/A")
        }
    except Exception as e:
        print(f"Feature Eng Error: {e}")
        # Return what we have if possible, but TCN needs specific columns.
        return pd.DataFrame(), {}, ticker

    # Relaxed dropna? or keep strict for model safety?
    # TCN needs all features. If SMA_200 is NaN for first 200 rows, we lose 200 rows.
    # If history < 200, we return empty.
    # Fillna with 0 for SMAs? risky.
    # Let's keep dropna() but rely on "max" period to get enough data.
    
    return df.dropna(), kpis, resolved_ticker