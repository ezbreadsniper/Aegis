#!/usr/bin/env python3
"""
Live Training Dashboard Server V2.0
Real-time visualization of RL training metrics with advanced analytics
"""
from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

LOG_DIR = "logs"

class DashboardAnalytics:
    @staticmethod
    def calculate_sharpe(returns, risk_free_rate=0.0):
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        # Annualized Sharpe (assuming ~1 trade per minute for HFT, heavily dependent on freq)
        # For simplicity in this training view, we just use raw mean/std
        return np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def calculate_max_drawdown(cumulative_pnl):
        if len(cumulative_pnl) < 1:
            return 0.0
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl)
        # If peak is 0 or negative (losing from start), drawdown is just absolute loss
        # We return absolute max dollar drawdown here
        return np.max(drawdown) if len(drawdown) > 0 else 0.0

    @staticmethod
    def calculate_profit_factor(pnl_series):
        wins = pnl_series[pnl_series > 0].sum()
        losses = abs(pnl_series[pnl_series < 0].sum())
        if losses == 0:
            return 99.99 if wins > 0 else 0.0
        return wins / losses

    @staticmethod
    def calculate_agent_health(win_rate, entropy, stable_entropy_range=(0.5, 3.0)):
        """
        Heuristic score 0-100 for agent health.
        """
        score = 50.0 # Base score

        # Win Rate contribution (Target > 50%)
        if win_rate > 50:
            score += (win_rate - 50) * 1.5
        else:
            score -= (50 - win_rate) * 2.0
            
        # Entropy stability check (Prevent collapse or total randomness)
        # Assuming discrete/continuous mixed, we want some entropy but not too much or too little
        if stable_entropy_range[0] <= entropy <= stable_entropy_range[1]:
            score += 10
        elif entropy < stable_entropy_range[0]:
            score -= 20 # Collapse warning
        
        return max(0, min(100, score))

def get_latest_session():
    """Find most recent training session"""
    if not os.path.exists(LOG_DIR):
        return None, None
    updates_files = sorted([f for f in os.listdir(LOG_DIR) if f.startswith("updates_")])
    trades_files = sorted([f for f in os.listdir(LOG_DIR) if f.startswith("trades_")])
    
    if not updates_files or not trades_files:
        return None, None
    
    return updates_files[-1], trades_files[-1]

def load_updates():
    """Load PPO update data"""
    updates_file, _ = get_latest_session()
    if not updates_file:
        return pd.DataFrame()
    
    try:
        return pd.read_csv(os.path.join(LOG_DIR, updates_file))
    except:
        return pd.DataFrame()

def load_trades():
    """Load trade data"""
    _, trades_file = get_latest_session()
    if not trades_file:
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(os.path.join(LOG_DIR, trades_file))
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
        return df
    except:
        return pd.DataFrame()

@app.route('/')
def dashboard():
    """Serve V2 Terminal dashboard"""
    return render_template('dashboard.html')

@app.route('/legacy')
def legacy_dashboard():
    """Serve legacy dashboard"""
    return render_template('dashboard.html')

@app.route('/api/updates')
def api_updates():
    """Return PPO update data for charts"""
    df = load_updates()
    if df.empty:
        return jsonify({'data': []})
    
    return jsonify({
        'update_nums': df['update_num'].tolist(),
        'policy_loss': df['policy_loss'].fillna(0).tolist(),
        'value_loss': df['value_loss'].fillna(0).tolist(),
        'entropy': df['entropy'].fillna(0).tolist(),
        'approx_kl': df['approx_kl'].fillna(0).tolist(),
        'clip_fraction': df['clip_fraction'].fillna(0).tolist(),
        'cumulative_pnl': df['cumulative_pnl'].fillna(0).tolist(),
        'cumulative_win_rate': (df['cumulative_win_rate'] * 100).tolist() if 'cumulative_win_rate' in df.columns else [],
    })

@app.route('/api/trades')
def api_trades():
    """Return trade data for charts"""
    df = load_trades()
    if df.empty:
        return jsonify({'data': []})
    
    # Asset counts
    asset_counts = df['asset'].value_counts().to_dict()
    
    # PnL by asset
    pnl_by_asset = df.groupby('asset')['pnl'].sum().to_dict()
    
    # PnL distribution
    pnl_values = df['pnl'].dropna().tolist()
    
    # Duration distribution
    durations = df['duration_sec'].dropna().tolist() if 'duration_sec' in df.columns else []
    
    # Rolling PnL (50-trade window)
    rolling = df['pnl'].rolling(50).mean().dropna().tolist()
    rolling_idx = list(range(50, 50 + len(rolling)))
    
    # Recent trades (last 10)
    recent = df.tail(10)[['timestamp', 'asset', 'action', 'pnl']].to_dict('records')
    
    return jsonify({
        'asset_counts': asset_counts,
        'pnl_by_asset': pnl_by_asset,
        'pnl_values': pnl_values[-500:],
        'durations': durations[-500:],
        'rolling_pnl': rolling[-100:],
        'rolling_idx': rolling_idx[-100:],
        'recent_trades': recent,
        'total_trades': len(df),
    })

@app.route('/api/performance')
def api_performance():
    """Return Scorecard metrics"""
    trades_df = load_trades()
    updates_df = load_updates()
    
    if trades_df.empty:
        return jsonify({})

    pnl_series = trades_df['pnl']
    wins_count = (pnl_series > 0).sum()
    total_count = len(trades_df)
    
    net_pnl = pnl_series.sum()
    win_rate = (wins_count / total_count * 100) if total_count > 0 else 0
    profit_factor = DashboardAnalytics.calculate_profit_factor(pnl_series)
    max_dd = DashboardAnalytics.calculate_max_drawdown(pnl_series.cumsum())
    sharpe = DashboardAnalytics.calculate_sharpe(pnl_series) # Simplified Sharpe
    
    # Entropy from latest update
    current_entropy = updates_df['entropy'].iloc[-1] if not updates_df.empty else 0
    
    # Health Score
    health_score = DashboardAnalytics.calculate_agent_health(win_rate, current_entropy)
    health_status = "Stable Learning" if health_score > 70 else ("Optimizing" if health_score > 40 else "Unstable")

    return jsonify({
        'net_pnl': round(net_pnl, 2),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe, 3),
        'max_drawdown': round(max_dd, 2),
        'health_score': int(health_score),
        'health_status': health_status,
        'total_trades': total_count
    })

@app.route('/api/behavior')
def api_behavior():
    """Return Trade Behavior metrics"""
    df = load_trades()
    if df.empty:
        return jsonify({})
    
    # Avg Holding Time (assuming duration_sec exists, else mock/calc)
    durations = df['duration_sec'].dropna() if 'duration_sec' in df.columns else []
    avg_holding = np.mean(durations) if len(durations) > 0 else 0
    
    # Long vs Short
    if 'side' in df.columns:
        longs = df[df['side'] == 'LONG'].shape[0] if 'side' in df.values else df[df['action'].astype(str).str.contains('BUY')].shape[0] # Fallback logic
        shorts = len(df) - longs
    else:
        # Fallback if side column missing, assume roughly even or extract from action if possible
        # For now just placeholders if missing
        longs = len(df) // 2 
        shorts = len(df) - longs

    long_pct = round(longs / len(df) * 100, 1) if len(df) > 0 else 0
    short_pct = round(100 - long_pct, 1) if len(df) > 0 else 0

    return jsonify({
        'avg_holding_time': round(avg_holding, 1),
        'long_percentage': long_pct,
        'short_percentage': short_pct,
        'long_count': longs,
        'short_count': shorts
    })

@app.route('/api/trades/recent')
def api_recent_trades():
    df = load_trades()
    if df.empty:
        return jsonify([])
    
    # Return last 50 trades for table
    recent = df.tail(50).to_dict('records')
    return jsonify(recent)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5051)
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory to read from (e.g., logs/aegis)')
    args = parser.parse_args()
    
    # Override global LOG_DIR
    LOG_DIR = args.log_dir
    
    print("🚀 Starting RL Training Terminal V2.0...")
    print(f"📊 Open http://localhost:{args.port} in your browser")
    print(f"📂 Reading logs from: {args.log_dir}")
    print("-" * 50)
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
