#!/usr/bin/env python3
"""
Whale Detection Module.

Detects large trades (whales) that often precede price movements:
1. Tracks trade size distribution to find 95th percentile threshold
2. Flags trades significantly larger than normal
3. Provides directional signal (whale buy vs whale sell)

Research basis:
- Large trades >95th percentile often precede moves by 10-30 seconds
- Whale accumulation/distribution patterns are detectable
- Following informed traders (whales) can improve timing
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from collections import deque
import numpy as np
import time


@dataclass
class WhaleDetector:
    """
    Detect unusually large trades that may indicate informed trading.
    
    Maintains rolling statistics of trade sizes to dynamically
    calculate the "whale threshold" (95th percentile).
    """
    
    # Detection parameters
    window_size: int = 500  # Rolling window for percentile calculation
    percentile_threshold: float = 95.0  # What percentile = whale
    decay_ticks: int = 10  # How long whale signal persists
    
    # State per asset
    trade_histories: Dict[str, deque] = field(default_factory=dict)
    whale_thresholds: Dict[str, float] = field(default_factory=dict)
    recent_whales: Dict[str, List[Tuple[float, int, float]]] = field(default_factory=dict)
    # (timestamp, direction, size)
    
    def update(self, asset: str, trade_size: float, is_buy: bool) -> Tuple[bool, int]:
        """
        Update with new trade and check if it's a whale.
        
        Args:
            asset: Asset identifier
            trade_size: Dollar size of trade
            is_buy: Whether trade was a buy (True) or sell (False)
            
        Returns:
            (is_whale, whale_direction) where direction is 1=buy, -1=sell, 0=none
        """
        # Initialize history if needed
        if asset not in self.trade_histories:
            self.trade_histories[asset] = deque(maxlen=self.window_size)
            self.recent_whales[asset] = []
        
        history = self.trade_histories[asset]
        history.append(trade_size)
        
        # Calculate threshold dynamically
        if len(history) >= 20:  # Need minimum data
            sizes = list(history)
            self.whale_thresholds[asset] = np.percentile(sizes, self.percentile_threshold)
        else:
            # Use a default until we have enough data
            self.whale_thresholds[asset] = trade_size * 5  # 5x current = whale
        
        threshold = self.whale_thresholds[asset]
        is_whale = trade_size > threshold
        
        if is_whale:
            direction = 1 if is_buy else -1
            now = time.time()
            self.recent_whales[asset].append((now, direction, trade_size))
            
            # Prune old whale records (older than decay window)
            cutoff = now - (self.decay_ticks * 0.5)  # ~5 seconds
            self.recent_whales[asset] = [
                (t, d, s) for t, d, s in self.recent_whales[asset] if t > cutoff
            ]
            
            return True, direction
        
        return False, 0
    
    def get_whale_signal(self, asset: str) -> float:
        """
        Get aggregated whale signal for an asset.
        
        Returns value in [-1, 1]:
        - Positive: Net whale buying
        - Negative: Net whale selling
        - Zero: No recent whales or balanced
        """
        if asset not in self.recent_whales or not self.recent_whales[asset]:
            return 0.0
        
        # Weight by recency and size
        now = time.time()
        weighted_sum = 0.0
        total_weight = 0.0
        
        for timestamp, direction, size in self.recent_whales[asset]:
            age = now - timestamp
            recency_weight = max(0, 1 - age / 5.0)  # Linear decay over 5 seconds
            weight = recency_weight * (size / self.whale_thresholds.get(asset, 1000))
            weighted_sum += direction * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize to [-1, 1]
        signal = weighted_sum / total_weight
        return max(-1.0, min(1.0, signal))
    
    def get_whale_count(self, asset: str) -> int:
        """Get count of recent whale trades for an asset."""
        if asset not in self.recent_whales:
            return 0
        return len(self.recent_whales[asset])
    
    def get_threshold(self, asset: str) -> float:
        """Get current whale threshold for an asset."""
        return self.whale_thresholds.get(asset, 0.0)
    
    def reset(self, asset: str = None):
        """Reset state for an asset or all assets."""
        if asset:
            self.trade_histories.pop(asset, None)
            self.whale_thresholds.pop(asset, None)
            self.recent_whales.pop(asset, None)
        else:
            self.trade_histories.clear()
            self.whale_thresholds.clear()
            self.recent_whales.clear()


# Global instance
_detector: Optional[WhaleDetector] = None


def get_whale_detector() -> WhaleDetector:
    """Get or create the global WhaleDetector instance."""
    global _detector
    if _detector is None:
        _detector = WhaleDetector()
    return _detector


def update_whale_detection(asset: str, trade_size: float, is_buy: bool) -> Tuple[bool, int]:
    """Update and check if trade is a whale. Convenience function."""
    return get_whale_detector().update(asset, trade_size, is_buy)


def get_whale_signal(asset: str) -> float:
    """Get current whale signal for asset. Returns [-1, 1]."""
    return get_whale_detector().get_whale_signal(asset)


def get_whale_count(asset: str) -> int:
    """Get count of recent whale trades for asset."""
    return get_whale_detector().get_whale_count(asset)
