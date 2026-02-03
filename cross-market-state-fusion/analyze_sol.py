import pandas as pd
import os

# Find latest trades file
log_dir = "logs"
files = [f for f in os.listdir(log_dir) if f.startswith("trades_")]
latest_file = sorted(files)[-1]
file_path = os.path.join(log_dir, latest_file)

print(f"Reading: {file_path}")
df = pd.read_csv(file_path)

# Calculate stats
total_pnl = df['pnl'].sum()
by_asset = df.groupby('asset')['pnl'].sum().sort_values()

print(f"\nTotal PnL: ${total_pnl:.2f}")
print("\nPnL by Asset:")
print(by_asset)

# Ex-SOL calculations
pnl_no_sol = df[df['asset'] != 'SOL']['pnl'].sum()
diff = total_pnl - pnl_no_sol

print(f"\nPnL w/o SOL: ${pnl_no_sol:.2f}")
print(f"Difference: ${diff:.2f} (This is SOL's contribution)")

# Win rates
print("\nWin Rates by Asset:")
win_rates = df.groupby('asset').apply(lambda x: (x['pnl'] > 0).mean() * 100)
print(win_rates)
