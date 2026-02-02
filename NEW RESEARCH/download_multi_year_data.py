import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def get_kraken_ohlc_multi_year(pair, interval, years=3):
    url = "https://api.kraken.com/0/public/OHLC"
    all_data = []
    
    # Start from X years ago
    since = int((datetime.now() - timedelta(days=365 * years)).timestamp())
    
    while True:
        params = {"pair": pair, "interval": interval, "since": since}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                break
            data = response.json()
            if data.get("error"):
                print(f"Error for {pair}: {data['error']}")
                break
            
            result = data["result"]
            pair_key = [k for k in result.keys() if k != "last"][0]
            ohlc_data = result[pair_key]
            last_ts = result["last"]
            
            if not ohlc_data:
                break
                
            all_data.extend(ohlc_data)
            
            # If the last timestamp hasn't changed, we've reached the end
            if since == last_ts:
                break
            since = last_ts
            
            print(f"Downloaded {len(all_data)} rows for {pair}...")
            time.sleep(1) # Rate limit
            
            # For 3 years of 15m data, we expect ~105,120 rows.
            if len(all_data) > 150000: # Safety break
                break
                
        except Exception as e:
            print(f"Exception for {pair}: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

if __name__ == "__main__":
    assets = {
        "BTC": "XXBTZUSD",
        "ETH": "XETHZUSD",
        "SOL": "SOLUSD",
        "XRP": "XRPUSD"
    }
    interval = 15 # 15 minutes
    
    for name, pair in assets.items():
        print(f"Starting multi-year download for {name}...")
        df = get_kraken_ohlc_multi_year(pair, interval, years=3)
        if not df.empty:
            path = f"/home/ubuntu/{name.lower()}_multi_year_15m_kraken.csv"
            df.to_csv(path, index=False)
            print(f"Saved {len(df)} rows for {name} to {path}")
