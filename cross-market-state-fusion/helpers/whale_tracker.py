#!/usr/bin/env python3
"""
Whale Wallet Tracking for Polymarket.

Tracks historically profitable wallets and provides a whale activity signal
that can be used as a feature for the RL strategy.

Based on 2026 research: following profitable traders provides leading indicators.
"""
import time
import requests
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class WhaleStats:
    """Track performance stats for a wallet."""
    address: str
    total_pnl: float = 0.0
    win_count: int = 0
    trade_count: int = 0
    last_seen: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return self.win_count / max(1, self.trade_count)


class WhaleTracker:
    """
    Tracks whale (profitable wallet) activity on Polymarket.
    
    Uses Polymarket Data API to:
    1. Identify historically profitable wallets
    2. Track their current positions
    3. Generate a whale activity signal (-1 to +1)
    
    Usage:
        tracker = WhaleTracker()
        tracker.update_from_api()  # Fetch latest trades
        signal = tracker.get_signal(token_id)  # Get whale signal for a token
    """
    
    # Polymarket Data API endpoint
    DATA_API = "https://data-api.polymarket.com"
    
    def __init__(
        self,
        min_pnl_threshold: float = 1000.0,  # Min PnL to be considered a whale
        min_trades: int = 10,  # Min trades to qualify
        top_n: int = 50,  # Track top N whales
        lookback_seconds: int = 3600,  # 1 hour lookback for recent activity
    ):
        self.min_pnl_threshold = min_pnl_threshold
        self.min_trades = min_trades
        self.top_n = top_n
        self.lookback_seconds = lookback_seconds
        
        # Whale database
        self.wallet_stats: Dict[str, WhaleStats] = {}
        self.whale_addresses: Set[str] = set()
        
        # Current whale positions per token
        # token_id -> {address: side}
        self.whale_positions: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Cache for API calls
        self._last_fetch = 0.0
        self._fetch_interval = 30.0  # Fetch every 30 seconds
        
    def is_whale(self, address: str) -> bool:
        """Check if address qualifies as a whale."""
        stats = self.wallet_stats.get(address)
        if not stats:
            return False
        return (
            stats.total_pnl >= self.min_pnl_threshold and
            stats.trade_count >= self.min_trades
        )
    
    def update_whale_list(self):
        """Update the set of whale addresses based on current stats."""
        # Sort by PnL and take top N
        sorted_wallets = sorted(
            self.wallet_stats.values(),
            key=lambda x: x.total_pnl,
            reverse=True
        )
        
        qualified = [
            w for w in sorted_wallets
            if w.total_pnl >= self.min_pnl_threshold and w.trade_count >= self.min_trades
        ]
        
        self.whale_addresses = {w.address for w in qualified[:self.top_n]}
        
    def update_from_trades(self, trades: List[dict]):
        """
        Process trades and update whale tracking.
        
        Trade format (from Polymarket API):
        {
            "id": "...",
            "maker": "0x...",  # Wallet address
            "taker": "0x...",
            "token_id": "...",
            "side": "BUY" or "SELL",
            "price": 0.55,
            "size": 100.0,
            "timestamp": 1234567890
        }
        """
        now = time.time()
        
        for trade in trades:
            maker = trade.get("maker", "").lower()
            taker = trade.get("taker", "").lower()
            token_id = trade.get("token_id", "")
            side = trade.get("side", "").upper()
            
            # Track both maker and taker
            for address in [maker, taker]:
                if not address or len(address) < 10:
                    continue
                    
                if address not in self.wallet_stats:
                    self.wallet_stats[address] = WhaleStats(address=address)
                
                stats = self.wallet_stats[address]
                stats.trade_count += 1
                stats.last_seen = now
            
            # Update whale positions
            if maker in self.whale_addresses:
                self.whale_positions[token_id][maker] = side
                
    def update_from_api(self, condition_ids: Optional[List[str]] = None) -> bool:
        """
        Fetch recent trades from Polymarket Data API.
        
        Args:
            condition_ids: Optional list of condition IDs to filter
            
        Returns:
            True if fetch was successful
        """
        now = time.time()
        
        # Rate limiting
        if now - self._last_fetch < self._fetch_interval:
            return False
            
        try:
            # Build URL with optional filtering
            url = f"{self.DATA_API}/trades"
            params = {
                "limit": 500,
            }
            if condition_ids:
                params["condition_id"] = ",".join(condition_ids[:5])  # Max 5
                
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                trades = response.json()
                if isinstance(trades, list):
                    self.update_from_trades(trades)
                    self.update_whale_list()
                    self._last_fetch = now
                    return True
                    
        except Exception as e:
            # Silently fail - don't crash trading engine
            pass
            
        return False
    
    def get_signal(self, token_id: str) -> float:
        """
        Get whale activity signal for a token.
        
        Returns:
            -1.0: Net whale selling
             0.0: Neutral/no whale activity
            +1.0: Net whale buying
        """
        if token_id not in self.whale_positions:
            return 0.0
            
        positions = self.whale_positions[token_id]
        
        if not positions:
            return 0.0
            
        # Count buys vs sells
        buys = sum(1 for side in positions.values() if side == "BUY")
        sells = sum(1 for side in positions.values() if side == "SELL")
        total = buys + sells
        
        if total == 0:
            return 0.0
            
        # Normalized signal: (buys - sells) / total
        return (buys - sells) / total
    
    def get_whale_count(self) -> int:
        """Get number of tracked whales."""
        return len(self.whale_addresses)
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "total_wallets_tracked": len(self.wallet_stats),
            "whale_count": len(self.whale_addresses),
            "tokens_with_activity": len(self.whale_positions),
            "last_fetch": self._last_fetch,
        }


# Singleton instance
_tracker: Optional[WhaleTracker] = None


def get_whale_tracker() -> WhaleTracker:
    """Get or create the global whale tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = WhaleTracker()
    return _tracker
