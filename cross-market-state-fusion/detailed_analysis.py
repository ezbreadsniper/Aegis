#!/usr/bin/env python3
"""Generate comprehensive training analysis with multiple charts."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime, timedelta
import sys

# Read the current session trades
trades_file = r"c:\Users\Mamaodu Djigo\Desktop\bot15\cross-market-state-fusion\logs\trades_20260107_045038.csv"

try:
    df = pd.read_csv(trades_file)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Parse timestamps and calculate metrics
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['cumulative_pnl'] = df['pnl'].cumsum()
df['is_win'] = df['pnl'] > 0
df['cumulative_wins'] = df['is_win'].cumsum()
df['trade_number'] = range(1, len(df) + 1)
df['running_win_rate'] = df['cumulative_wins'] / df['trade_number'] * 100

# Calculate rolling metrics
window = 100
df['rolling_pnl'] = df['pnl'].rolling(window=window).mean()
df['rolling_win_rate'] = df['is_win'].rolling(window=window).mean() * 100

# Per-asset analysis
asset_stats = df.groupby('asset').agg({
    'pnl': ['sum', 'mean', 'count'],
    'is_win': 'mean'
}).round(2)
asset_stats.columns = ['Total PnL', 'Avg PnL', 'Trades', 'Win Rate']
asset_stats['Win Rate'] = (asset_stats['Win Rate'] * 100).round(1)

# Time-based analysis
df['hour'] = df['timestamp'].dt.hour
hourly_pnl = df.groupby('hour')['pnl'].sum()

# Calculate drawdown
df['peak'] = df['cumulative_pnl'].cummax()
df['drawdown'] = df['peak'] - df['cumulative_pnl']
max_drawdown = df['drawdown'].max()

# Create comprehensive figure
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Title
duration_hours = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600
fig.suptitle(f'EarnHFT v7.6 Training Analysis\n{duration_hours:.1f} Hours | {len(df):,} Trades | +${df["cumulative_pnl"].iloc[-1]:,.2f} PnL', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Main Equity Curve (large, top left spanning 2 cols)
ax1 = fig.add_subplot(gs[0, :2])
ax1.fill_between(df['trade_number'], df['cumulative_pnl'], alpha=0.3, color='green')
ax1.plot(df['trade_number'], df['cumulative_pnl'], color='green', linewidth=2, label='Cumulative PnL')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_ylabel('Cumulative PnL ($)', fontsize=11)
ax1.set_xlabel('Trade Number', fontsize=11)
ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Add milestone annotations
for milestone in [1000, 2000, 3000, 4000, 5000]:
    if len(df) >= milestone:
        pnl_at = df['cumulative_pnl'].iloc[milestone-1]
        ax1.axvline(x=milestone, color='blue', alpha=0.3, linestyle=':')
        ax1.annotate(f'{milestone}\n${pnl_at:.0f}', (milestone, pnl_at), 
                     textcoords="offset points", xytext=(5,10), fontsize=9)

# 2. Drawdown Chart (top right)
ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(df['trade_number'], df['drawdown'], alpha=0.4, color='red')
ax2.plot(df['trade_number'], df['drawdown'], color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown ($)', fontsize=11)
ax2.set_xlabel('Trade Number', fontsize=11)
ax2.set_title(f'Drawdown (Max: ${max_drawdown:.2f})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Rolling Win Rate (middle left)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df['trade_number'], df['rolling_win_rate'], color='orange', linewidth=1.5)
ax3.axhline(y=33.33, color='gray', linestyle='--', alpha=0.7, label='Random (33%)')
ax3.axhline(y=df['running_win_rate'].iloc[-1], color='orange', linestyle=':', alpha=0.7)
ax3.set_ylabel('Win Rate (%)', fontsize=11)
ax3.set_xlabel('Trade Number', fontsize=11)
ax3.set_title(f'Rolling Win Rate ({window}-trade window)', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 50)
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. PnL Distribution (middle center)
ax4 = fig.add_subplot(gs[1, 1])
wins = df[df['pnl'] > 0]['pnl']
losses = df[df['pnl'] < 0]['pnl']
ax4.hist(wins, bins=50, alpha=0.7, color='green', label=f'Wins (n={len(wins)})')
ax4.hist(losses, bins=50, alpha=0.7, color='red', label=f'Losses (n={len(losses)})')
ax4.axvline(x=wins.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Avg Win: ${wins.mean():.2f}')
ax4.axvline(x=losses.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Avg Loss: ${losses.mean():.2f}')
ax4.set_xlabel('PnL ($)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('PnL Distribution', fontsize=14, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Per-Asset Performance (middle right)
ax5 = fig.add_subplot(gs[1, 2])
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in asset_stats['Total PnL']]
bars = ax5.bar(asset_stats.index, asset_stats['Total PnL'], color=colors, alpha=0.8)
ax5.axhline(y=0, color='black', linewidth=0.5)
ax5.set_ylabel('Total PnL ($)', fontsize=11)
ax5.set_title('PnL by Asset', fontsize=14, fontweight='bold')
for bar, val in zip(bars, asset_stats['Total PnL']):
    ax5.annotate(f'${val:.0f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Rolling PnL per Trade (bottom left)
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(df['trade_number'], df['rolling_pnl'], color='blue', linewidth=1.5)
ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
avg_pnl = df['pnl'].mean()
ax6.axhline(y=avg_pnl, color='green', linestyle=':', alpha=0.7, label=f'Overall Avg: ${avg_pnl:.2f}')
ax6.set_ylabel('Avg PnL per Trade ($)', fontsize=11)
ax6.set_xlabel('Trade Number', fontsize=11)
ax6.set_title(f'Rolling Avg PnL ({window}-trade window)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()

# 7. Hourly Performance (bottom center)
ax7 = fig.add_subplot(gs[2, 1])
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in hourly_pnl.values]
ax7.bar(hourly_pnl.index, hourly_pnl.values, color=colors, alpha=0.8)
ax7.axhline(y=0, color='black', linewidth=0.5)
ax7.set_xlabel('Hour (UTC)', fontsize=11)
ax7.set_ylabel('Total PnL ($)', fontsize=11)
ax7.set_title('PnL by Hour', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Summary Stats Box (bottom right)
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

total_pnl = df['cumulative_pnl'].iloc[-1]
total_trades = len(df)
win_rate = df['running_win_rate'].iloc[-1]
avg_win = wins.mean() if len(wins) > 0 else 0
avg_loss = losses.mean() if len(losses) > 0 else 0
profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')
expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

stats_text = f"""
╔══════════════════════════════════════╗
║       SESSION SUMMARY                ║
╠══════════════════════════════════════╣
║  Duration:      {duration_hours:>6.1f} hours          ║
║  Total Trades:  {total_trades:>6,}               ║
║  Total PnL:     ${total_pnl:>8,.2f}           ║
╠══════════════════════════════════════╣
║       PERFORMANCE METRICS            ║
╠══════════════════════════════════════╣
║  Win Rate:      {win_rate:>6.1f}%              ║
║  Avg Win:       ${avg_win:>8.2f}           ║
║  Avg Loss:      ${avg_loss:>8.2f}           ║
║  Profit Factor: {profit_factor:>8.2f}x           ║
║  Expectancy:    ${expectancy:>8.2f}/trade     ║
║  Max Drawdown:  ${max_drawdown:>8.2f}           ║
╠══════════════════════════════════════╣
║       vs ORIGINAL PHASE 5            ║
╠══════════════════════════════════════╣
║  Their Win Rate:    23.3%            ║
║  Our Win Rate:      {win_rate:.1f}%   {'✓' if win_rate >= 23 else ''}         ║
║  Their PF:          ~2.0x            ║
║  Our PF:            {profit_factor:.2f}x   {'✓' if profit_factor >= 2 else ''}          ║
╚══════════════════════════════════════╝
"""

ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes, fontsize=11,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
output_path = r"C:\Users\Mamaodu Djigo\.gemini\antigravity\brain\0cdd6ec1-0ce1-4125-ba33-2fb807e3691c\detailed_analysis_chart.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Chart saved to: {output_path}")

# Save asset stats for markdown
asset_stats.to_csv(r"C:\Users\Mamaodu Djigo\.gemini\antigravity\brain\0cdd6ec1-0ce1-4125-ba33-2fb807e3691c\asset_stats.csv")

# Print summary
print(f"\n{'='*50}")
print("FINAL STATS")
print('='*50)
print(f"Duration: {duration_hours:.1f} hours")
print(f"Total PnL: ${total_pnl:,.2f}")
print(f"Total Trades: {total_trades:,}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Expectancy: ${expectancy:.2f}/trade")
print(f"Max Drawdown: ${max_drawdown:.2f}")
print(f"\nPer-Asset Performance:")
print(asset_stats.to_string())
