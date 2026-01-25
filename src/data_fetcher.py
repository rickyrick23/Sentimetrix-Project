import yfinance as yf
import pandas as pd
import functools
import ta
import requests

@functools.lru_cache(maxsize=32) # Simple in-memory cache
def get_alpha_live_data(ticker):
    """Fetches real-time price data and calculates 17 technical indicators."""
    
    # 1. Fetch deep historical data
    try:
        # Standard yfinance call
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="2y", interval="1d", auto_adjust=True)
    except Exception as e:
        print(f"YF Error: {e}")
        return pd.DataFrame(), {}
    
    if df.empty:
         return pd.DataFrame(), {}

    # Standardize columns for consistency
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    # 2. Advanced Feature Engineering (The Alpha Factors)
    # Replicating pandas-ta logic using 'ta' library
    
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
    info = yf.Ticker(ticker).info
    kpis = {
        "Market Cap": info.get("marketCap", "N/A"),
        "52W High": info.get("fiftyTwoWeekHigh", "N/A")
    }
    
    return df.dropna(), kpis