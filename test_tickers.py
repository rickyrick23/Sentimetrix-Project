
from src.data_fetcher import get_alpha_live_data
import pandas as pd

def test_ticker(ticker):
    print(f"Testing {ticker}...")
    try:
        df, kpis = get_alpha_live_data(ticker)
        if df.empty:
            print(f"FAILED: {ticker} returned empty DataFrame.")
            # Check why?
        else:
            print(f"SUCCESS: {ticker} - {len(df)} rows. KPIs: {kpis.keys()}")
    except Exception as e:
        print(f"ERROR: {ticker} - {e}")

tickers = [
    "AAPL",      # US Stock
    "BTC-USD",   # Crypto with suffix
    "BTC",       # Crypto without suffix (Expected to fail or be weird)
    "EURUSD=X",  # Forex standard
    "EURUSD",    # Forex no suffix (Expected to fail)
    "RELIANCE.NS", # Indian stock
    "DJANT",     # Random/Fake
]

for t in tickers:
    test_ticker(t)
