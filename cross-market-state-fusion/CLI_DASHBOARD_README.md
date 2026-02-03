# CLI Dashboard for Polymarket Trading Bot

A beautiful terminal-based monitoring dashboard built with Textual framework.

## Features

- **Real-time PnL tracking** with color-coded display
- **Active positions table** showing entry/current prices and unrealized PnL
- **Live trade log** with scrollable history
- **Market countdown timers** for all active markets (BTC, ETH, SOL, XRP)
- **RL training metrics** (buffer size, entropy, avg reward, updates)
- **Polymarket balance** display (updates every 30 seconds)
- **Keyboard shortcuts** for quick actions

## Installation

Dependencies are already included in `requirements.txt`:

```bash
pip install textual rich
```

## Usage

### Running the Dashboard

Open a **new terminal window** (separate from your training session) and run:

```bash
cd "c:\Users\Mamaodu Djigo\Desktop\bot15\cross-market-state-fusion"
python cli_dashboard.py
```

The dashboard will automatically connect to your running bot and display live data.

### Keyboard Shortcuts

- **q** - Quit the dashboard
- **r** - Force refresh all data
- **c** - Clear trade history view
- **↑/↓** - Scroll through trade log

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  POLYMARKET TRADING BOT                      [Clock] [Status Bar] │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐  ┌─────────────────────────┐  ┌──────────────┐ │
│  │ PnL Summary │  │   Active Positions      │  │  Trade Log   │ │
│  │   +$12.47   │  │                         │  │              │ │
│  │  Trades: 47 │  │  BTC  UP   $0.52  $0.58 │  │ [14:32] BTC  │ │
│  │   WR: 62%   │  │  ETH  DOWN $0.47  $0.44 │  │ [14:30] ETH  │ │
│  └─────────────┘  └─────────────────────────┘  │ [14:28] SOL  │ │
│                                                 │              │ │
│  ┌─────────────┐                                └──────────────┘ │
│  │   Balance   │                                                 │
│  │  $847.23    │                                                 │
│  └─────────────┘                                                 │
│                                                                   │
│  ┌─────────────┐                                                 │
│  │ RL Metrics  │                                                 │
│  │ Buffer: 256 │                                                 │
│  │ Entropy:2.3 │                                                 │
│  └─────────────┘                                                 │
├──────────────────────────────────────────────────────────────────┤
│  [BTC 4:23]  [ETH 3:51]  [SOL 8:12]  [XRP 12:05]                │
│  Prob: 58.3% Prob: 45.2% Prob: 52.1% Prob: 49.8%                │
└──────────────────────────────────────────────────────────────────┘
```

## Data Sources

The dashboard reads from the same `dashboard_state` object used by the web dashboard (`dashboard_cinematic.py`), ensuring:

- **No interference** with your training
- **Identical data** to the web dashboard  
- **Zero additional API calls** to Polymarket (except for balance fetch)

## Troubleshooting

### Dashboard shows zeros

Make sure:
1. Your training bot is running (`python run.py earnhft --train --size 200`)
2. The bot has dashboard integration enabled (should be automatic in `run.py`)
3. You've made at least one trade (data only appears after first trade)

### Balance not updating

Check that:
1. Your `.env` file has valid credentials (`PRIVATE_KEY`, `WALLET_ADDRESS`)
2. You can run `python helpers/polymarket_execution.py` without errors

### Connection errors

The dashboard will gracefully handle connection errors and continue displaying the last known data. Just press `r` to force a refresh.

## Tips

- Run the dashboard on a second monitor or in a split terminal for best experience
- The dashboard updates every 1 second automatically
- You can run multiple instances to monitor different aspects
- All data is read-only, the dashboard cannot affect your trading

## Advanced Usage

### Customizing Update Intervals

Edit `cli_dashboard.py` and modify:

```python
# Line ~288: Main update interval (default 1 second)
self.set_interval(1.0, self.update_dashboard)

# Line ~291: Balance fetch interval (default 30 seconds)
self.set_interval(30.0, self.update_balance)
```

### Changing Color Themes

Textual uses CSS-like styling. Edit the `CSS` variable in `PolymarketDashboard` class to customize colors.

## Support

If you encounter issues, check:
1. All dependencies are installed: `pip install -r requirements.txt`
2. Python version is 3.10+ 
3. Terminal supports color output (most modern terminals do)
