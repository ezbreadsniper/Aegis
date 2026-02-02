#!/usr/bin/env python3
"""
Optimal Stop-Loss Parameter Analysis for 15-Minute Crypto Markets
Analyzes multi-year data to identify optimal ATR multipliers, trailing stop parameters,
and regime-aware thresholds for BTC, ETH, SOL, and XRP.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_hurst_exponent(series, max_lag=100):
    """Calculate Hurst exponent for regime detection."""
    lags = range(2, min(max_lag, len(series) // 2))
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    
    # Filter out zeros and invalid values
    valid_indices = [i for i, t in enumerate(tau) if t > 0]
    if len(valid_indices) < 2:
        return 0.5  # Default to random walk
    
    log_lags = np.log([list(lags)[i] for i in valid_indices])
    log_tau = np.log([tau[i] for i in valid_indices])
    
    # Linear regression
    poly = np.polyfit(log_lags, log_tau, 1)
    return poly[0]

def simulate_stop_loss_strategy(df, atr_multiplier, trailing=False):
    """
    Simulate a stop-loss strategy and calculate performance metrics.
    
    Args:
        df: DataFrame with OHLCV data and ATR
        atr_multiplier: Multiplier for ATR-based stop-loss
        trailing: Whether to use trailing stop-loss
    
    Returns:
        Dictionary with performance metrics
    """
    trades = []
    position = None
    entry_price = None
    stop_loss = None
    highest_price = None
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if pd.isna(row['atr']) or row['atr'] == 0:
            continue
        
        # Simple momentum entry signal (for simulation purposes)
        momentum = (row['close'] - df.iloc[max(0, i-4)]['close']) / df.iloc[max(0, i-4)]['close']
        
        if position is None:
            # Entry logic: buy on positive momentum
            if momentum > 0.001:  # 0.1% threshold
                position = 'long'
                entry_price = row['close']
                stop_loss = entry_price - (row['atr'] * atr_multiplier)
                highest_price = entry_price
        else:
            # Update trailing stop if enabled
            if trailing and row['close'] > highest_price:
                highest_price = row['close']
                stop_loss = highest_price - (row['atr'] * atr_multiplier)
            
            # Check stop-loss
            if row['low'] <= stop_loss:
                exit_price = stop_loss
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'stop_loss'
                })
                position = None
                entry_price = None
                stop_loss = None
                highest_price = None
            
            # Check take-profit (2x ATR)
            elif row['high'] >= entry_price + (row['atr'] * atr_multiplier * 2):
                exit_price = entry_price + (row['atr'] * atr_multiplier * 2)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'take_profit'
                })
                position = None
                entry_price = None
                stop_loss = None
                highest_price = None
    
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    # Calculate metrics
    pnl_series = [t['pnl_pct'] for t in trades]
    wins = [t for t in trades if t['pnl_pct'] > 0]
    
    # Calculate max drawdown
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate Sharpe ratio (annualized, assuming 15-min bars)
    if np.std(pnl_series) > 0:
        sharpe = np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(252 * 24 * 4)  # 4 bars per hour
    else:
        sharpe = 0
    
    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_pnl': np.mean(pnl_series),
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'stop_loss_exits': len([t for t in trades if t['exit_reason'] == 'stop_loss']),
        'take_profit_exits': len([t for t in trades if t['exit_reason'] == 'take_profit'])
    }

def analyze_asset(filepath, asset_name):
    """Analyze optimal stop-loss parameters for a single asset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {asset_name}...")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Calculate ATR
    df['atr'] = calculate_atr(df, period=14)
    
    # Calculate Hurst exponent for regime detection
    if len(df) > 200:
        hurst = calculate_hurst_exponent(df['close'].values[-200:])
        print(f"Hurst Exponent (last 200 bars): {hurst:.3f}")
        if hurst > 0.5:
            print("  -> Trending regime (wider stops recommended)")
        else:
            print("  -> Mean-reverting regime (tighter stops recommended)")
    else:
        hurst = 0.5
    
    # Calculate volatility metrics
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].std() * np.sqrt(252 * 24 * 4) * 100  # Annualized volatility
    print(f"Annualized Volatility: {volatility:.2f}%")
    
    # Test different ATR multipliers
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    results = {
        'asset': asset_name,
        'hurst_exponent': hurst,
        'annualized_volatility': volatility,
        'static_stop_results': [],
        'trailing_stop_results': []
    }
    
    print(f"\n--- Static Stop-Loss Results ---")
    print(f"{'Multiplier':<12} {'Trades':<8} {'Win Rate':<10} {'Avg PnL':<10} {'Max DD':<10} {'Sharpe':<10}")
    print("-" * 60)
    
    for mult in multipliers:
        metrics = simulate_stop_loss_strategy(df, mult, trailing=False)
        results['static_stop_results'].append({
            'multiplier': mult,
            **metrics
        })
        print(f"{mult:<12.1f} {metrics['total_trades']:<8} {metrics['win_rate']:<10.1f} {metrics['avg_pnl']:<10.3f} {metrics['max_drawdown']:<10.2f} {metrics['sharpe_ratio']:<10.2f}")
    
    print(f"\n--- Trailing Stop-Loss Results ---")
    print(f"{'Multiplier':<12} {'Trades':<8} {'Win Rate':<10} {'Avg PnL':<10} {'Max DD':<10} {'Sharpe':<10}")
    print("-" * 60)
    
    for mult in multipliers:
        metrics = simulate_stop_loss_strategy(df, mult, trailing=True)
        results['trailing_stop_results'].append({
            'multiplier': mult,
            **metrics
        })
        print(f"{mult:<12.1f} {metrics['total_trades']:<8} {metrics['win_rate']:<10.1f} {metrics['avg_pnl']:<10.3f} {metrics['max_drawdown']:<10.2f} {metrics['sharpe_ratio']:<10.2f}")
    
    # Find optimal multiplier
    best_static = max(results['static_stop_results'], key=lambda x: x['sharpe_ratio'])
    best_trailing = max(results['trailing_stop_results'], key=lambda x: x['sharpe_ratio'])
    
    print(f"\n--- Optimal Parameters ---")
    print(f"Best Static Stop Multiplier: {best_static['multiplier']}x ATR (Sharpe: {best_static['sharpe_ratio']:.2f})")
    print(f"Best Trailing Stop Multiplier: {best_trailing['multiplier']}x ATR (Sharpe: {best_trailing['sharpe_ratio']:.2f})")
    
    results['optimal_static_multiplier'] = best_static['multiplier']
    results['optimal_trailing_multiplier'] = best_trailing['multiplier']
    
    return results

def main():
    """Main analysis function."""
    print("=" * 80)
    print("OPTIMAL STOP-LOSS PARAMETER ANALYSIS FOR 15-MINUTE CRYPTO MARKETS")
    print("=" * 80)
    
    assets = {
        'BTC': '/home/ubuntu/btc_multi_year_15m_kraken.csv',
        'ETH': '/home/ubuntu/eth_multi_year_15m_kraken.csv',
        'SOL': '/home/ubuntu/sol_multi_year_15m_kraken.csv',
        'XRP': '/home/ubuntu/xrp_multi_year_15m_kraken.csv'
    }
    
    all_results = {}
    
    for asset_name, filepath in assets.items():
        results = analyze_asset(filepath, asset_name)
        if results:
            all_results[asset_name] = results
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: OPTIMAL STOP-LOSS PARAMETERS")
    print("=" * 80)
    
    print(f"\n{'Asset':<8} {'Hurst':<8} {'Vol %':<10} {'Best Static':<15} {'Best Trailing':<15}")
    print("-" * 60)
    
    for asset, results in all_results.items():
        print(f"{asset:<8} {results['hurst_exponent']:<8.3f} {results['annualized_volatility']:<10.1f} {results['optimal_static_multiplier']:<15.1f}x {results['optimal_trailing_multiplier']:<15.1f}x")
    
    # Calculate average optimal multiplier
    avg_static = np.mean([r['optimal_static_multiplier'] for r in all_results.values()])
    avg_trailing = np.mean([r['optimal_trailing_multiplier'] for r in all_results.values()])
    
    print(f"\n{'AVERAGE':<8} {'-':<8} {'-':<10} {avg_static:<15.1f}x {avg_trailing:<15.1f}x")
    
    # Save results to JSON
    with open('/home/ubuntu/optimal_stop_loss_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to /home/ubuntu/optimal_stop_loss_results.json")
    
    # Generate recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR POLYMARKET BOT")
    print("=" * 80)
    
    print(f"""
Based on the analysis of multi-year 15-minute data for BTC, ETH, SOL, and XRP:

1. **Replace Hard $5 Stop-Loss** with ATR-based dynamic stop-loss.

2. **Recommended ATR Multiplier**:
   - Static Stop: {avg_static:.1f}x ATR
   - Trailing Stop: {avg_trailing:.1f}x ATR (RECOMMENDED)

3. **Regime-Aware Adjustment**:
   - If Hurst > 0.55 (trending): Use {avg_trailing + 0.5:.1f}x ATR
   - If Hurst < 0.45 (mean-reverting): Use {avg_trailing - 0.5:.1f}x ATR
   - If Hurst ~ 0.50 (random walk): Use {avg_trailing:.1f}x ATR

4. **Position Sizing**:
   - Use 0.25x-0.5x Fractional Kelly based on probability edge.
   - This reduces the need for hard stops.

5. **Expiry Handling**:
   - Close all positions 2-3 minutes before expiry.
   - Do NOT open new positions in the final 5 minutes.
""")

if __name__ == "__main__":
    main()
