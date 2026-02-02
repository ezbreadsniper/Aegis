import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def discover_unicorns():
    assets = ["btc", "eth", "sol", "xrp"]
    dfs = {}
    
    for asset in assets:
        try:
            df = pd.read_csv(f"/home/ubuntu/{asset}_multi_year_15m_kraken.csv")
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            dfs[asset] = df
        except Exception as e:
            print(f"Error loading {asset}: {e}")
            
    if not dfs:
        print("No data loaded.")
        return

    # 1. Cross-Asset Volatility Clusters (The "Unicorn" Signal)
    # Identify periods where all 4 assets have a volatility spike simultaneously
    print("Searching for Cross-Asset Volatility Clusters...")
    returns_df = pd.DataFrame({asset: dfs[asset]['close'].pct_change() for asset in assets if asset in dfs}).dropna()
    vols_df = returns_df.rolling(window=96).std() # 24h rolling vol
    
    # Normalize volatility to identify relative spikes
    norm_vols = (vols_df - vols_df.mean()) / vols_df.std()
    
    # A "Unicorn" event is when all assets are > 3 sigma in volatility
    unicorn_events = norm_vols[(norm_vols > 3).all(axis=1)]
    print(f"Found {len(unicorn_events)} 'Unicorn' Volatility Clusters.")
    
    # 2. Liquidity-Driven Flash Reversals
    # Identify periods where price moves > 5% in 15m and reverses > 50% in the next 15m
    print("\nSearching for Flash Reversals...")
    for asset in assets:
        ret = returns_df[asset]
        reversals = returns_df[(ret.abs() > 0.05) & (ret.shift(-1) * ret < -0.025)]
        print(f"{asset.upper()} Flash Reversals: {len(reversals)}")
        
    # 3. Long-Term Regime Shift Triggers
    # Use a rolling correlation between BTC and ETH to identify regime shifts
    print("\nAnalyzing BTC-ETH Correlation Regime Shifts...")
    returns_df['btc_eth_corr'] = returns_df['btc'].rolling(window=2880).corr(returns_df['eth']) # 30-day rolling corr
    
    # Identify periods where correlation drops below 0.3 (Rare Regime)
    rare_regimes = returns_df[returns_df['btc_eth_corr'] < 0.3]
    print(f"Rare BTC-ETH Correlation Regimes (<0.3): {len(rare_regimes)}")
    
    # 4. Visualization: Unicorn Volatility Clusters
    if not unicorn_events.empty:
        plt.figure(figsize=(12, 6))
        for asset in assets:
            plt.plot(norm_vols.index, norm_vols[asset], label=f"{asset.upper()}")
        plt.axhline(y=3, color='r', linestyle='--', label='Unicorn Threshold')
        plt.title("Cross-Asset Volatility Clusters (Unicorn Discovery)")
        plt.legend()
        plt.savefig("/home/ubuntu/unicorn_volatility_clusters.png")
        
    return returns_df

if __name__ == "__main__":
    discover_unicorns()
