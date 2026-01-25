import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests

# ENSURE THIS NAME MATCHES THE IMPORT IN APP.PY
def fetch_ticker_data(ticker, start_date="2020-01-01"):
    """
    Downloads raw stock data and adds technical alpha indicators.
    """
    # Standard yfinance call (v0.2+ handles session internally)
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, auto_adjust=True)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()
    
    # Standardize column names
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    # Feature Engineering using 'ta' library
    
    # RSI (14)
    # pandas-ta default: "RSI_14"
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD (12, 26, 9)
    # pandas-ta names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df['MACDh_12_26_9'] = macd.macd_diff()
    df['MACDs_12_26_9'] = macd.macd_signal()
    
    # Bollinger Bands (20, 2)
    # pandas-ta names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBP_20_2.0, BBD_20_2.0 (Check BBP/BBD support in ta)
    # ta gives bb_mavg, bb_hband, bb_lband, bb_wband, bb_pband
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    df['BBP_20_2.0'] = bb.bollinger_pband() # Percent Band
    # BBD (Bandwidth) -> wband? ta has bollinger_wband()
    df['BBD_20_2.0'] = bb.bollinger_wband()

    # ATR (14)
    # pandas-ta: ATR_14 (approx?)
    # ta requires high, low, close
    df['ATRr_14'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # SMA (50, 200)
    # pandas-ta: SMA_50, SMA_200
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()

    # Target Labeling
    df['returns'] = df['close'].pct_change().shift(-1)
    df['target'] = df['returns'].apply(lambda r: 2 if r > 0.025 else (0 if r < -0.025 else 1))
    
    return df.dropna()