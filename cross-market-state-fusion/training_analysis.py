#!/usr/bin/env python3
"""
Deep Training Analysis with Visualizations
Generates comprehensive charts showing RL training progress
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_data():
    """Load training logs"""
    log_dir = "logs"
    
    # Find most recent session
    updates_files = [f for f in os.listdir(log_dir) if f.startswith("updates_")]
    trades_files = [f for f in os.listdir(log_dir) if f.startswith("trades_")]
    
    if not updates_files or not trades_files:
        print("No training logs found!")
        return None, None
    
    # Use most recent by filename
    updates_file = sorted(updates_files)[-1]
    trades_file = sorted(trades_files)[-1]
    
    print(f"📊 Analyzing: {updates_file}")
    
    updates_df = pd.read_csv(os.path.join(log_dir, updates_file))
    trades_df = pd.read_csv(os.path.join(log_dir, trades_file))
    
    return updates_df, trades_df

def create_training_dashboard(updates_df, trades_df):
    """Create comprehensive training dashboard"""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('🚀 RL Training Deep Analysis Dashboard', fontsize=20, fontweight='bold', color='cyan')
    
    # Create grid layout
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.25)
    
    # ========== ROW 1 ==========
    
    # 1. Cumulative PnL Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(updates_df['update_num'], updates_df['cumulative_pnl'], alpha=0.4, color='lime')
    ax1.plot(updates_df['update_num'], updates_df['cumulative_pnl'], 'lime', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('PPO Update #')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.set_title('💰 Cumulative PnL Growth', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    max_pnl = updates_df['cumulative_pnl'].max()
    max_idx = updates_df['cumulative_pnl'].idxmax()
    ax1.annotate(f'Peak: ${max_pnl:.0f}', xy=(updates_df.loc[max_idx, 'update_num'], max_pnl),
                 xytext=(10, 10), textcoords='offset points', color='yellow', fontweight='bold')
    
    # 2. Policy Loss (Should stay negative and stable)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(updates_df['update_num'], updates_df['policy_loss'], 'cyan', linewidth=2, marker='s', markersize=4)
    ax2.axhline(y=updates_df['policy_loss'].mean(), color='yellow', linestyle='--', alpha=0.7, label=f"Mean: {updates_df['policy_loss'].mean():.4f}")
    ax2.set_xlabel('PPO Update #')
    ax2.set_ylabel('Policy Loss')
    ax2.set_title('📉 Policy Loss (Should Be Negative & Stable)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Value Loss (Should decrease over time)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(updates_df['update_num'], updates_df['value_loss'], color='orange', alpha=0.7, width=0.8)
    ax3.plot(updates_df['update_num'], updates_df['value_loss'].rolling(5).mean(), 'white', linewidth=2, label='5-Update MA')
    ax3.set_xlabel('PPO Update #')
    ax3.set_ylabel('Value Loss')
    ax3.set_title('📊 Value Loss (Critic Learning)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== ROW 2 ==========
    
    # 4. Entropy (Should decrease gradually)
    ax4 = fig.add_subplot(gs[1, 0])
    entropy_color = ['lime' if e > 0.8 else 'yellow' if e > 0.5 else 'red' for e in updates_df['entropy']]
    ax4.scatter(updates_df['update_num'], updates_df['entropy'], c=entropy_color, s=50, zorder=5)
    ax4.plot(updates_df['update_num'], updates_df['entropy'], 'white', linewidth=1, alpha=0.5)
    ax4.axhline(y=1.0, color='lime', linestyle='--', alpha=0.5, label='High (Exploring)')
    ax4.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Optimal Zone')
    ax4.fill_between(updates_df['update_num'], 0.5, 1.0, alpha=0.1, color='green')
    ax4.set_xlabel('PPO Update #')
    ax4.set_ylabel('Entropy')
    ax4.set_title('🎲 Entropy (Exploration vs Exploitation)', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.2)
    
    # 5. Win Rate Over Time
    ax5 = fig.add_subplot(gs[1, 1])
    win_rates = updates_df['cumulative_win_rate'] * 100
    ax5.fill_between(updates_df['update_num'], win_rates, alpha=0.4, color='magenta')
    ax5.plot(updates_df['update_num'], win_rates, 'magenta', linewidth=2, marker='o', markersize=4)
    ax5.axhline(y=50, color='white', linestyle='--', alpha=0.5, label='50% Break-even')
    ax5.set_xlabel('PPO Update #')
    ax5.set_ylabel('Win Rate (%)')
    ax5.set_title('🏆 Cumulative Win Rate', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. KL Divergence (Should stay under target)
    ax6 = fig.add_subplot(gs[1, 2])
    kl = updates_df['approx_kl']
    kl_color = ['lime' if k < 0.02 else 'red' for k in kl]
    ax6.bar(updates_df['update_num'], kl, color=kl_color, alpha=0.7, width=0.8)
    ax6.axhline(y=0.02, color='yellow', linestyle='--', linewidth=2, label='Target KL (0.02)')
    ax6.set_xlabel('PPO Update #')
    ax6.set_ylabel('Approx KL')
    ax6.set_title('📏 KL Divergence (Policy Change)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # ========== ROW 3 ==========
    
    # 7. Trades Per Asset
    ax7 = fig.add_subplot(gs[2, 0])
    asset_counts = trades_df['asset'].value_counts()
    colors = {'BTC': 'gold', 'ETH': 'deepskyblue', 'SOL': 'mediumorchid', 'XRP': 'lime'}
    bars = ax7.bar(asset_counts.index, asset_counts.values, color=[colors.get(a, 'gray') for a in asset_counts.index])
    ax7.set_xlabel('Asset')
    ax7.set_ylabel('Number of Trades')
    ax7.set_title('📈 Trades by Asset', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, asset_counts.values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val), ha='center', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. PnL Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    pnl_data = trades_df['pnl'].dropna()
    wins = pnl_data[pnl_data > 0]
    losses = pnl_data[pnl_data < 0]
    ax8.hist(wins, bins=50, alpha=0.7, color='lime', label=f'Wins ({len(wins)})', edgecolor='white')
    ax8.hist(losses, bins=50, alpha=0.7, color='red', label=f'Losses ({len(losses)})', edgecolor='white')
    ax8.axvline(x=0, color='white', linestyle='--', linewidth=2)
    ax8.axvline(x=pnl_data.mean(), color='yellow', linestyle='--', linewidth=2, label=f'Mean: ${pnl_data.mean():.2f}')
    ax8.set_xlabel('PnL ($)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('💵 PnL Distribution', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. PnL by Asset
    ax9 = fig.add_subplot(gs[2, 2])
    pnl_by_asset = trades_df.groupby('asset')['pnl'].sum().sort_values()
    bar_colors = ['lime' if v > 0 else 'red' for v in pnl_by_asset.values]
    bars = ax9.barh(pnl_by_asset.index, pnl_by_asset.values, color=bar_colors, edgecolor='white')
    ax9.axvline(x=0, color='white', linestyle='--', linewidth=2)
    ax9.set_xlabel('Total PnL ($)')
    ax9.set_ylabel('Asset')
    ax9.set_title('💰 Total PnL by Asset', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, pnl_by_asset.values):
        offset = 5 if val > 0 else -25
        ax9.text(val + offset, bar.get_y() + bar.get_height()/2, f'${val:.0f}', va='center', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='x')
    
    # ========== ROW 4 ==========
    
    # 10. Trade Duration Distribution
    ax10 = fig.add_subplot(gs[3, 0])
    durations = trades_df['duration_sec'].dropna()
    ax10.hist(durations, bins=50, color='cyan', alpha=0.7, edgecolor='white')
    ax10.axvline(x=durations.median(), color='yellow', linestyle='--', linewidth=2, label=f'Median: {durations.median():.1f}s')
    ax10.set_xlabel('Duration (seconds)')
    ax10.set_ylabel('Frequency')
    ax10.set_title('⏱️ Trade Duration Distribution', fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Clip Fraction (How often PPO is clipping)
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.fill_between(updates_df['update_num'], updates_df['clip_fraction'], alpha=0.4, color='orange')
    ax11.plot(updates_df['update_num'], updates_df['clip_fraction'], 'orange', linewidth=2)
    ax11.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High Clipping')
    ax11.axhline(y=0.1, color='yellow', linestyle='--', alpha=0.7, label='Normal Range')
    ax11.set_xlabel('PPO Update #')
    ax11.set_ylabel('Clip Fraction')
    ax11.set_title('✂️ PPO Clip Fraction', fontsize=12, fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Rolling PnL per Trade
    ax12 = fig.add_subplot(gs[3, 2])
    rolling_pnl = trades_df['pnl'].rolling(50).mean()
    ax12.plot(rolling_pnl.index, rolling_pnl, 'lime', linewidth=2, label='50-Trade Rolling Avg')
    ax12.fill_between(rolling_pnl.index, rolling_pnl, alpha=0.3, color='lime')
    ax12.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax12.set_xlabel('Trade #')
    ax12.set_ylabel('Avg PnL ($)')
    ax12.set_title('📈 Rolling PnL per Trade (50-Trade Window)', fontsize=12, fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "training_analysis_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black', edgecolor='none')
    print(f"✅ Dashboard saved to: {output_path}")
    
    return output_path

def print_summary_stats(updates_df, trades_df):
    """Print key statistics"""
    print("\n" + "="*60)
    print("📊 TRAINING ANALYSIS SUMMARY")
    print("="*60)
    
    # PPO Stats
    print("\n🧠 PPO TRAINING METRICS:")
    print(f"  • Total Updates: {len(updates_df)}")
    print(f"  • Total Trades: {len(trades_df):,}")
    print(f"  • Training Duration: ~{len(updates_df)} minutes")
    
    # Entropy Analysis
    start_entropy = updates_df['entropy'].iloc[0]
    end_entropy = updates_df['entropy'].iloc[-1]
    print(f"\n🎲 ENTROPY (Exploration):")
    print(f"  • Start: {start_entropy:.4f}")
    print(f"  • End:   {end_entropy:.4f}")
    print(f"  • Change: {end_entropy - start_entropy:+.4f}")
    if end_entropy < start_entropy:
        print(f"  ✅ Agent becoming more confident!")
    
    # PnL Analysis
    start_pnl = updates_df['cumulative_pnl'].iloc[0]
    end_pnl = updates_df['cumulative_pnl'].iloc[-1]
    max_pnl = updates_df['cumulative_pnl'].max()
    print(f"\n💰 PnL ANALYSIS:")
    print(f"  • Starting PnL: ${start_pnl:.2f}")
    print(f"  • Current PnL:  ${end_pnl:.2f}")
    print(f"  • Peak PnL:     ${max_pnl:.2f}")
    print(f"  • Growth:       ${end_pnl - start_pnl:+.2f}")
    
    # Trade Stats
    wins = (trades_df['pnl'] > 0).sum()
    losses = (trades_df['pnl'] < 0).sum()
    zeros = (trades_df['pnl'] == 0).sum()
    total = len(trades_df)
    
    print(f"\n📈 TRADE STATISTICS:")
    print(f"  • Wins:   {wins} ({100*wins/total:.1f}%)")
    print(f"  • Losses: {losses} ({100*losses/total:.1f}%)")
    print(f"  • Flat:   {zeros} ({100*zeros/total:.1f}%)")
    
    # Best/Worst
    best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
    worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
    print(f"\n🏆 BEST TRADE:")
    print(f"  • {best_trade['asset']} {best_trade['action']}: ${best_trade['pnl']:.2f}")
    print(f"\n💀 WORST TRADE:")
    print(f"  • {worst_trade['asset']} {worst_trade['action']}: ${worst_trade['pnl']:.2f}")
    
    # Asset Breakdown
    print(f"\n📊 PnL BY ASSET:")
    for asset in trades_df['asset'].unique():
        asset_pnl = trades_df[trades_df['asset'] == asset]['pnl'].sum()
        asset_count = len(trades_df[trades_df['asset'] == asset])
        print(f"  • {asset}: ${asset_pnl:.2f} ({asset_count} trades)")
    
    print("\n" + "="*60)

def main():
    print("🚀 Loading training data...")
    updates_df, trades_df = load_data()
    
    if updates_df is None:
        return
    
    print_summary_stats(updates_df, trades_df)
    
    print("\n🎨 Generating dashboard...")
    output_path = create_training_dashboard(updates_df, trades_df)
    
    print(f"\n✅ Analysis complete! Dashboard saved to: {output_path}")

if __name__ == "__main__":
    main()
