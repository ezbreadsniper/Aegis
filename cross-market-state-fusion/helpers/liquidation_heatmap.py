#!/usr/bin/env python3
"""
Liquidation Heatmap Module.

Tracks clusters of liquidation events to identify:
1. Price Magnets: Areas where price is likely to be drawn.
2. Reversal Zones: Areas where intense liquidation often marks a local top/bottom.

Logic:
- Accumulate liquidation volume in price buckets.
- Decay buckets over time (liquidity is ephemeral).
- Detect "clusters" of high liquidation volume.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict
import math
import time

@dataclass
class LiquidationCluster:
    price: float
    volume: float
    side: str  # "SELL" (Long Liq) or "BUY" (Short Liq)
    timestamp: float

class LiquidationHeatmap:
    """Tracks liquidation clusters."""
    
    def __init__(self, decay_rate: float = 0.95):
        # asset -> side -> price_bucket -> volume
        self.heatmap: Dict[str, Dict[str, Dict[float, float]]] = defaultdict(
            lambda: {"SELL": defaultdict(float), "BUY": defaultdict(float)}
        )
        self.decay_rate = decay_rate  # Decay per minute
        self.last_decay = time.time()
        
        # Bucket sizes
        self.bucket_sizes = {
            "BTC": 10.0,
            "ETH": 2.0,
            "SOL": 0.5,
            "XRP": 0.005,
        }

    def update(self, asset: str, price: float, volume: float, side: str):
        """Add liquidation event."""
        bucket_size = self.bucket_sizes.get(asset, 1.0)
        bucket = round(price / bucket_size) * bucket_size
        
        self.heatmap[asset][side][bucket] += volume
        
    def _decay(self):
        """Apply decay to all buckets."""
        now = time.time()
        if now - self.last_decay < 60:  # Decay every minute
            return
            
        minutes = (now - self.last_decay) / 60
        factor = self.decay_rate ** minutes
        
        for asset in self.heatmap:
            for side in ["SELL", "BUY"]:
                for bucket in list(self.heatmap[asset][side].keys()):
                    self.heatmap[asset][side][bucket] *= factor
                    if self.heatmap[asset][side][bucket] < 100:  # Cleanup noise
                        del self.heatmap[asset][side][bucket]
                        
        self.last_decay = now

    def get_strongest_level(self, asset: str, side: str) -> Tuple[float, float]:
        """Get price and volume of strongest liquidation level."""
        self._decay()
        buckets = self.heatmap[asset][side]
        if not buckets:
            return 0.0, 0.0
            
        max_price = max(buckets, key=buckets.get)
        return max_price, buckets[max_price]

    def get_proximity_signal(self, asset: str, current_price: float) -> float:
        """
        Calculate proximity to major liquidation clusters.
        Returns:
            -1.0 to 1.0 signal.
            Positive = Price attracted upwards (Short Liqs above).
            Negative = Price attracted downwards (Long Liqs below).
        """
        # Guard against zero price
        if current_price <= 0:
            return 0.0
            
        long_liq_price, long_vol = self.get_strongest_level(asset, "SELL")
        short_liq_price, short_vol = self.get_strongest_level(asset, "BUY")
        
        signal = 0.0
        
        # Attraction to Short Liqs (Shorts buying back -> Price UP)
        if short_vol > 1000 and short_liq_price > current_price:
            dist = (short_liq_price - current_price) / current_price
            if dist < 0.02:  # Within 2%
                signal += (1 - dist/0.02) * (min(short_vol, 1e6) / 1e6)
                
        # Attraction to Long Liqs (Longs selling -> Price DOWN)
        if long_vol > 1000 and long_liq_price < current_price:
            dist = (current_price - long_liq_price) / current_price
            if dist < 0.02:
                signal -= (1 - dist/0.02) * (min(long_vol, 1e6) / 1e6)
                
        return max(-1.0, min(1.0, signal))

# Global instance
_heatmap = LiquidationHeatmap()

def get_liquidation_heatmap() -> LiquidationHeatmap:
    return _heatmap

def update_heatmap(asset: str, price: float, volume: float, side: str):
    get_liquidation_heatmap().update(asset, price, volume, side)

def get_liq_signal(asset: str, price: float) -> float:
    return get_liquidation_heatmap().get_proximity_signal(asset, price)
