#!/usr/bin/env python3
"""Generate PnL progress chart from trades log."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys

# Read the current session trades
trades_file = r"c:\Users\Mamaodu Djigo\Desktop\bot15\cross-market-state-fusion\logs\trades_20260107_045038.csv"

try:
    df = pd.read_csv(trades_file)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Parse timestamps and calculate cumulative PnL
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['cumulative_pnl'] = df['pnl'].cumsum()

# Calculate running win rate
df['is_win'] = df['pnl'] > 0
df['cumulative_wins'] = df['is_win'].cumsum()
df['trade_number'] = range(1, len(df) + 1)
df['running_win_rate'] = df['cumulative_wins'] / df['trade_number'] * 100

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('EarnHFT v7.6 Training Progress - Live Session', fontsize=16, fontweight='bold')

# Plot 1: Cumulative PnL
ax1.fill_between(df['trade_number'], df['cumulative_pnl'], alpha=0.3, color='green')
ax1.plot(df['trade_number'], df['cumulative_pnl'], color='green', linewidth=2, label='Cumulative PnL')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_ylabel('Cumulative PnL ($)', fontsize=12)
ax1.set_title(f'Total PnL: ${df["cumulative_pnl"].iloc[-1]:,.2f} | Trades: {len(df):,}', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Add milestone markers
milestones = [500, 1000, 1500]
for m in milestones:
    if len(df) > m:
        pnl_at_m = df['cumulative_pnl'].iloc[m-1]
        ax1.scatter([m], [pnl_at_m], color='blue', s=100, zorder=5)
        ax1.annotate(f'${pnl_at_m:.0f}', (m, pnl_at_m), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

# Plot 2: Running Win Rate
ax2.plot(df['trade_number'], df['running_win_rate'], color='orange', linewidth=2, label='Win Rate %')
ax2.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random (33%)')
ax2.axhline(y=df['running_win_rate'].iloc[-1], color='orange', linestyle=':', alpha=0.7)
ax2.set_xlabel('Trade Number', fontsize=12)
ax2.set_ylabel('Win Rate (%)', fontsize=12)
ax2.set_title(f'Final Win Rate: {df["running_win_rate"].iloc[-1]:.1f}%', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 50)

# Add summary stats as text
stats_text = f"""Session Stats:
• Duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()/3600:.1f} hours
• Total Trades: {len(df):,}
• Win Rate: {df['running_win_rate'].iloc[-1]:.1f}%
• Avg Win: ${df[df['pnl'] > 0]['pnl'].mean():.2f}
• Avg Loss: ${df[df['pnl'] < 0]['pnl'].mean():.2f}
• Profit Factor: {abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()):.2f}"""

fig.text(0.98, 0.5, stats_text, fontsize=10, verticalalignment='center', 
         horizontalalignment='right', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(right=0.78)

# Save the chart
output_path = r"C:\Users\Mamaodu Djigo\.gemini\antigravity\brain\0cdd6ec1-0ce1-4125-ba33-2fb807e3691c\pnl_progress_chart.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Chart saved to: {output_path}")
print(f"\nFinal Stats:")
print(f"  PnL: ${df['cumulative_pnl'].iloc[-1]:,.2f}")
print(f"  Trades: {len(df):,}")
print(f"  Win Rate: {df['running_win_rate'].iloc[-1]:.1f}%")
