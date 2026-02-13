
import requests

def search_yahoo(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers)
        data = r.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            first = data['quotes'][0]
            print(f"Query: '{query}' -> Found: {first['symbol']} ({first.get('longname', 'N/A')}) Type: {first.get('quoteType', 'N/A')}")
            return first['symbol']
        else:
            print(f"Query: '{query}' -> No match.")
    except Exception as e:
        print(f"Error: {e}")

queries = ["Bitcoin", "Apple", "Euro", "Reliance", "BTC", "EURUSD"]
for q in queries:
    search_yahoo(q)
