
from src.data_fetcher import get_alpha_live_data
import pandas as pd

def test_ticker(ticker):
    print(f"Testing {ticker}...")
    try:
        # Now returns 3 values
        df, kpis, resolved = get_alpha_live_data(ticker)
        if df.empty:
            print(f"FAILED: {ticker} returned empty DataFrame. Resolved: {resolved}")
        else:
            print(f"SUCCESS: {ticker} -> Resolved: {resolved} - {len(df)} rows.")
    except Exception as e:
        print(f"ERROR: {ticker} - {e}")

tickers = [
    "Bitcoin",   # Should resolve to BTC-USD
    "Apple",     # Should resolve to AAPL
    "Eur",       # Should resolve to something (maybe EURUSD=X or an ETF)
    "Euro",
    "Reliance",  # Should resolve to RELIANCE.NS
]

for t in tickers:
    test_ticker(t)
