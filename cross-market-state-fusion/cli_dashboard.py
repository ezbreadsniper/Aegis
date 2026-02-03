#!/usr/bin/env python3
"""
Polymarket Trading Bot - Terminal Dashboard

A beautiful terminal UI for monitoring your trading bot in real-time.
Shows live PnL, positions, trades, markets, and RL metrics.

Usage:
    python cli_dashboard.py

Keyboard Shortcuts:
    q - Quit
    r - Refresh data
    c - Clear trade history
    ↑/↓ - Scroll trade log
"""
import asyncio
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, DataTable, Label
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table as RichTable

# Import dashboard state from existing infrastructure
try:
    from dashboard_cinematic import dashboard_state
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("Warning: dashboard_cinematic not available, using mock data")

# Import balance fetcher
try:
    from helpers.polymarket_execution import get_executor
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False


class PnLPanel(Static):
    """Display PnL summary statistics."""
    
    total_pnl = reactive(0.0)
    trade_count = reactive(0)
    win_count = reactive(0)
    
    def render(self) -> Panel:
        """Render the PnL panel."""
        win_rate = (self.win_count / max(1, self.trade_count)) * 100
        avg_pnl = self.total_pnl / max(1, self.trade_count)
        
        # Color coding
        pnl_color = "green" if self.total_pnl >= 0 else "red"
        wr_color = "green" if win_rate >= 50 else "yellow" if win_rate >= 40 else "red"
        
        content = f"""[bold]PnL SUMMARY[/bold]

[{pnl_color}]Total PnL: ${self.total_pnl:+.2f}[/{pnl_color}]
Trades: {self.trade_count}
Wins: {self.win_count} / Losses: {self.trade_count - self.win_count}
[{wr_color}]Win Rate: {win_rate:.1f}%[/{wr_color}]
Avg PnL/Trade: ${avg_pnl:+.2f}
"""
        
        return Panel(
            content,
            title="📊 Performance",
            border_style="bright_cyan",
        )


class BalancePanel(Static):
    """Display Polymarket balance."""
    
    balance = reactive(0.0)
    last_update = reactive("")
    
    def render(self) -> Panel:
        """Render the balance panel."""
        content = f"""[bold cyan]${self.balance:.2f}[/bold cyan]

[dim]Last Updated:[/dim]
[dim]{self.last_update}[/dim]
"""
        
        return Panel(
            content,
            title="💰 Balance",
            border_style="bright_green",
        )


class RLMetricsPanel(Static):
    """Display RL training metrics."""
    
    buffer_size = reactive(0)
    max_buffer = reactive(2048)
    entropy = reactive(0.0)
    avg_reward = reactive(0.0)
    updates = reactive(0)
    
    def render(self) -> Panel:
        """Render the RL metrics panel."""
        buffer_pct = (self.buffer_size / max(1, self.max_buffer)) * 100
        reward_color = "green" if self.avg_reward >= 0 else "red"
        
        content = f"""[bold]RL METRICS[/bold]

Buffer: {self.buffer_size}/{self.max_buffer} [{buffer_pct:.0f}%]
Updates: {self.updates}
Entropy: {self.entropy:.3f}
[{reward_color}]Avg Reward: {self.avg_reward:+.4f}[/{reward_color}]
"""
        
        return Panel(
            content,
            title="🧠 Learning",
            border_style="bright_magenta",
        )


class PositionsTable(Static):
    """Display active positions in a table."""
    
    positions = reactive({})
    markets = reactive({})
    
    def render(self) -> Panel:
        """Render the positions table."""
        if not self.positions:
            content = "[dim]No active positions[/dim]"
        else:
            table = RichTable(show_header=True, header_style="bold cyan", box=None)
            table.add_column("Asset", style="cyan")
            table.add_column("Side", style="yellow")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("Unrealized PnL", justify="right")
            
            for cid, pos in self.positions.items():
                if pos.get("size", 0) <= 0:
                    continue
                    
                asset = self.markets.get(cid, {}).get("asset", "???")
                side = pos.get("side", "???")
                entry_price = pos.get("entry_price", 0)
                
                # Calculate current price and unrealized PnL
                market = self.markets.get(cid, {})
                current_prob = market.get("prob", entry_price)
                
                # Calculate unrealized PnL
                shares = pos.get("size", 0) / max(entry_price, 0.01)
                if side == "UP":
                    unrealized_pnl = (current_prob - entry_price) * shares
                else:
                    current_down = 1 - current_prob
                    unrealized_pnl = (current_down - entry_price) * shares
                
                pnl_color = "green" if unrealized_pnl >= 0 else "red"
                side_color = "green" if side == "UP" else "blue"
                
                table.add_row(
                    asset,
                    f"[{side_color}]{side}[/{side_color}]",
                    f"${entry_price:.3f}",
                    f"${current_prob:.3f}",
                    f"[{pnl_color}]${unrealized_pnl:+.2f}[/{pnl_color}]"
                )
            
            content = table
        
        return Panel(
            content,
            title="📈 Active Positions",
            border_style="bright_yellow",
        )


class TradeLogPanel(ScrollableContainer):
    """Scrollable log of recent trades."""
    
    trades: List[Dict] = reactive([])
    
    def render(self) -> Panel:
        """Render the trade log."""
        if not self.trades:
            content = "[dim]No trades yet...[/dim]"
        else:
            lines = []
            for trade in self.trades[:50]:  # Show last 50
                time_str = trade.get("time", "")
                asset = trade.get("asset", "???")
                action = trade.get("action", "???")
                size = trade.get("size", 0)
                pnl = trade.get("pnl")
                
                # Color code by action
                if "BUY" in action or "UP" in action:
                    action_color = "green"
                else:
                    action_color = "blue"
                
                # Format PnL if available
                if pnl is not None:
                    pnl_color = "green" if pnl >= 0 else "red"
                    pnl_str = f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]"
                else:
                    pnl_str = f"[dim]${size:.0f}[/dim]"
                
                line = f"[dim]{time_str}[/dim] [{action_color}]{asset}[/{action_color}] {action:10s} {pnl_str}"
                lines.append(line)
            
            content = "\n".join(lines)
        
        return Panel(
            content,
            title="📝 Recent Trades",
            border_style="bright_white",
        )


class MarketCard(Static):
    """Display a single market with countdown timer."""
    
    asset = reactive("")
    prob = reactive(0.5)
    time_left = reactive(0.0)
    has_position = reactive(False)
    position_side = reactive("")
    velocity = reactive(0.0)
    
    def render(self) -> Panel:
        """Render the market card."""
        # Format time remaining
        mins = int(self.time_left)
        secs = int((self.time_left - mins) * 60)
        time_str = f"{mins}:{secs:02d}"
        
        # Color code timer
        if self.time_left < 2:
            time_color = "red bold"
        elif self.time_left < 5:
            time_color = "yellow"
        else:
            time_color = "cyan"
        
        # Velocity indicator
        if abs(self.velocity) > 0.001:
            vel_arrow = "↑" if self.velocity > 0 else "↓"
            vel_color = "green" if self.velocity > 0 else "red"
            vel_str = f"[{vel_color}]{vel_arrow}{abs(self.velocity)*100:.2f}%[/{vel_color}]"
        else:
            vel_str = "[dim]~[/dim]"
        
        # Position indicator
        if self.has_position:
            pos_color = "green" if self.position_side == "UP" else "blue"
            pos_indicator = f"[{pos_color}]● {self.position_side}[/{pos_color}]"
            border_style = "bright_yellow"
        else:
            pos_indicator = "[dim]—[/dim]"
            border_style = "white"
        
        content = f"""[bold cyan]{self.asset}[/bold cyan]
[{time_color}]{time_str}[/{time_color}]

Prob: [bold]{self.prob*100:.1f}%[/bold] {vel_str}
{pos_indicator}
"""
        
        return Panel(
            content,
            border_style=border_style,
        )


class PolymarketDashboard(App):
    """Main Textual application for Polymarket trading bot monitoring."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #header-container {
        height: 3;
        background: $primary;
        border: solid $accent;
    }
    
    #main-container {
        height: 1fr;
        layout: horizontal;
    }
    
    #left-panel {
        width: 1fr;
        height: 100%;
    }
    
    #center-panel {
        width: 2fr;
        height: 100%;
    }
    
    #right-panel {
        width: 1fr;
        height: 100%;
    }
    
    #markets-strip {
        height: 12;
        layout: horizontal;
    }
    
    .market-card {
        width: 1fr;
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("c", "clear_trades", "Clear Trades"),
    ]
    
    def __init__(self):
        super().__init__()
        self.trades_list: List[Dict] = []
        
    def compose(self) -> ComposeResult:
        """Create the dashboard layout."""
        yield Header(show_clock=True)
        
        # Main content area
        with Container(id="main-container"):
            # Left panel - Stats
            with Vertical(id="left-panel"):
                yield PnLPanel(id="pnl_panel")
                yield BalancePanel(id="balance_panel")
                yield RLMetricsPanel(id="rl_panel")
            
            # Center panel - Positions
            with Vertical(id="center-panel"):
                yield PositionsTable(id="positions_table")
            
            # Right panel - Trade log
            with Vertical(id="right-panel"):
                yield TradeLogPanel(id="trade_log")
        
        # Bottom strip - Markets
        with Horizontal(id="markets-strip"):
            for asset in ["BTC", "ETH", "SOL", "XRP"]:
                yield MarketCard(id=f"market_{asset.lower()}", classes="market-card")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Start the dashboard when mounted."""
        self.title = "POLYMARKET TRADING BOT"
        self.sub_title = "Real-time Monitoring Dashboard"
        
        # Start live update task
        self.set_interval(1.0, self.update_dashboard)
        
        # Fetch balance every 30 seconds
        self.set_interval(30.0, self.update_balance)
        self.update_balance()  # Initial fetch
    
    async def update_dashboard(self) -> None:
        """Update all dashboard components with live data."""
        if not DASHBOARD_AVAILABLE:
            return
        
        try:
            # Import the state object
            from dashboard_cinematic import dashboard_state
            
            # Update PnL panel
            pnl_panel = self.query_one("#pnl_panel", PnLPanel)
            pnl_panel.total_pnl = getattr(dashboard_state, 'total_pnl', 0.0)
            pnl_panel.trade_count = getattr(dashboard_state, 'trade_count', 0)
            pnl_panel.win_count = getattr(dashboard_state, 'win_count', 0)
            
            # Update RL metrics
            rl_panel = self.query_one("#rl_panel", RLMetricsPanel)
            rl_panel.buffer_size = getattr(dashboard_state, 'buffer_size', 0)
            rl_panel.max_buffer = getattr(dashboard_state, 'max_buffer', 2048)
            rl_panel.entropy = getattr(dashboard_state, 'entropy', 0.0)
            rl_panel.avg_reward = getattr(dashboard_state, 'avg_reward', 0.0)
            rl_panel.updates = getattr(dashboard_state, 'updates', 0)
            
            # Update positions table
            positions_table = self.query_one("#positions_table", PositionsTable)
            positions_table.positions = getattr(dashboard_state, 'positions', {})
            positions_table.markets = getattr(dashboard_state, 'markets', {})
            
            # Update market cards
            markets = getattr(dashboard_state, 'markets', {})
            for cid, market in markets.items():
                asset = market.get("asset", "").lower()
                if asset:
                    try:
                        card = self.query_one(f"#market_{asset}", MarketCard)
                        card.asset = market.get("asset", "")
                        card.prob = market.get("prob", 0.5)
                        card.time_left = market.get("time_left", 0)
                        card.velocity = market.get("velocity", 0)
                        
                        # Check if we have a position
                        positions = getattr(dashboard_state, 'positions', {})
                        pos = positions.get(cid, {})
                        card.has_position = pos.get("size", 0) > 0
                        card.position_side = pos.get("side", "")
                    except:
                        pass
            
        except Exception as e:
            pass  # Silent fail to avoid crash
    
    async def update_balance(self) -> None:
        """Fetch and update Polymarket balance."""
        if not EXECUTOR_AVAILABLE:
            return
        
        try:
            executor = get_executor()
            balance = executor.get_balance()
            
            balance_panel = self.query_one("#balance_panel", BalancePanel)
            balance_panel.balance = balance
            balance_panel.last_update = datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            pass  # Silent fail
    
    def action_refresh(self) -> None:
        """Force refresh of all data."""
        asyncio.create_task(self.update_dashboard())
        asyncio.create_task(self.update_balance())
    
    def action_clear_trades(self) -> None:
        """Clear the trade history display."""
        self.trades_list.clear()
        trade_log = self.query_one("#trade_log", TradeLogPanel)
        trade_log.trades = []


def main():
    """Run the dashboard."""
    print("Starting Polymarket Trading Bot Dashboard...")
    print("Press 'q' to quit, 'r' to refresh, 'c' to clear trades")
    print()
    
    if not DASHBOARD_AVAILABLE:
        print("⚠️  Warning: Cannot import dashboard_state")
        print("   Make sure your bot is running with dashboard integration")
        print()
    
    app = PolymarketDashboard()
    app.run()


if __name__ == "__main__":
    main()
