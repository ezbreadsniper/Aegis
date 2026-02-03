#!/usr/bin/env python3
"""
Regime Classification Module.

Classifies market regimes for adaptive trading:
1. TRENDING: High ADX, clear directional movement
2. MEAN_REVERTING: Low ADX, range-bound, choppy
3. HIGH_VOLATILITY: Extreme vol, reduce exposure
4. LOW_VOLATILITY: Compression, breakout pending

Each regime suggests different strategy behavior:
- TRENDING: Follow momentum, ride winners longer
- MEAN_REVERTING: Fade extremes, tight stops
- HIGH_VOLATILITY: Reduce size, wait for clarity
- LOW_VOLATILITY: Look for breakouts, standard sizing
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import deque
from enum import Enum
import numpy as np


class Regime(Enum):
    """Market regime classification."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeClassifier:
    """
    Simple regime classifier using ADX + ATR rules.
    
    For 15-minute crypto markets:
    - ADX > 25: Strong trend
    - ADX < 20: Ranging/choppy
    - ATR spike: High volatility event
    - ATR compression: Low volatility, breakout pending
    """
    
    # Thresholds
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0
    vol_high_threshold: float = 2.0   # 2x average = high vol
    vol_low_threshold: float = 0.5    # 0.5x average = low vol
    
    # State per asset
    price_histories: Dict[str, deque] = field(default_factory=dict)
    regimes: Dict[str, Regime] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)
    
    # History length for calculations
    lookback: int = 30  # ~30 minutes of 1-min data
    
    def update(self, asset: str, price: float, 
               returns_1m: float = 0.0, returns_5m: float = 0.0,
               volatility: float = 0.0, avg_volatility: float = 0.0) -> Tuple[Regime, float]:
        """
        Update regime classification for an asset.
        
        Args:
            asset: Asset identifier
            price: Current price
            returns_1m: 1-minute returns
            returns_5m: 5-minute returns
            volatility: Current realized volatility
            avg_volatility: Average volatility (baseline)
            
        Returns:
            (regime, confidence) tuple
        """
        # Initialize history if needed
        if asset not in self.price_histories:
            self.price_histories[asset] = deque(maxlen=self.lookback)
        
        # Add price to history
        self.price_histories[asset].append(price)
        
        # Need minimum history for classification
        if len(self.price_histories[asset]) < 10:
            self.regimes[asset] = Regime.UNKNOWN
            self.confidences[asset] = 0.0
            return Regime.UNKNOWN, 0.0
        
        # Calculate simple directional movement strength (ADX proxy)
        prices = list(self.price_histories[asset])
        dm_strength = self._calculate_dm_strength(prices)
        
        # Calculate volatility ratio
        vol_ratio = volatility / max(0.0001, avg_volatility) if avg_volatility > 0 else 1.0
        
        # Classify regime
        regime, confidence = self._classify(dm_strength, vol_ratio, returns_1m, returns_5m)
        
        self.regimes[asset] = regime
        self.confidences[asset] = confidence
        
        return regime, confidence
    
    def _calculate_dm_strength(self, prices: list) -> float:
        """
        Calculate directional movement strength (simplified ADX).
        
        Higher values = stronger trend, lower values = ranging.
        """
        if len(prices) < 5:
            return 0.0
        
        # Calculate directional moves
        moves = []
        for i in range(1, len(prices)):
            moves.append(prices[i] - prices[i-1])
        
        if not moves:
            return 0.0
        
        # Net directional movement vs total movement
        net_move = abs(sum(moves))
        total_move = sum(abs(m) for m in moves)
        
        if total_move == 0:
            return 0.0
        
        # DM strength: 0 = pure chop, 100 = pure trend
        dm_strength = (net_move / total_move) * 100
        
        return dm_strength
    
    def _classify(self, dm_strength: float, vol_ratio: float,
                  returns_1m: float, returns_5m: float) -> Tuple[Regime, float]:
        """Classify regime based on indicators."""
        confidence = 0.5  # Base confidence
        
        # High volatility takes precedence
        if vol_ratio > self.vol_high_threshold:
            confidence = min(1.0, 0.5 + (vol_ratio - self.vol_high_threshold) * 0.25)
            return Regime.HIGH_VOLATILITY, confidence
        
        # Low volatility (compression)
        if vol_ratio < self.vol_low_threshold:
            confidence = min(1.0, 0.5 + (self.vol_low_threshold - vol_ratio) * 0.5)
            return Regime.LOW_VOLATILITY, confidence
        
        # Trending
        if dm_strength > self.adx_trending_threshold:
            confidence = min(1.0, 0.5 + (dm_strength - self.adx_trending_threshold) / 50)
            return Regime.TRENDING, confidence
        
        # Mean reverting / ranging
        if dm_strength < self.adx_ranging_threshold:
            confidence = min(1.0, 0.5 + (self.adx_ranging_threshold - dm_strength) / 40)
            return Regime.MEAN_REVERTING, confidence
        
        # In between - check recent returns for hints
        if abs(returns_5m) > 0.005:  # 0.5% move in 5m suggests trend
            return Regime.TRENDING, 0.4
        else:
            return Regime.MEAN_REVERTING, 0.4
    
    def get_regime(self, asset: str) -> Tuple[Regime, float]:
        """Get current regime and confidence for an asset."""
        return (
            self.regimes.get(asset, Regime.UNKNOWN),
            self.confidences.get(asset, 0.0)
        )
    
    def get_regime_adjustment(self, asset: str) -> Dict[str, float]:
        """
        Get strategy adjustments based on current regime.
        
        Returns dict with multipliers for:
        - position_size: Size adjustment
        - hold_time: How long to hold
        - edge_threshold: Required edge to trade
        """
        regime, confidence = self.get_regime(asset)
        
        if regime == Regime.TRENDING:
            return {
                "position_size": 1.0,       # Normal size
                "hold_time": 1.5,           # Hold longer
                "edge_threshold": 0.9,      # Lower threshold (momentum is edge)
            }
        elif regime == Regime.MEAN_REVERTING:
            return {
                "position_size": 0.8,       # Slightly reduced
                "hold_time": 0.7,           # Quick exits
                "edge_threshold": 1.0,      # Normal threshold
            }
        elif regime == Regime.HIGH_VOLATILITY:
            return {
                "position_size": 0.5,       # Half size
                "hold_time": 0.5,           # Very quick
                "edge_threshold": 1.3,      # Higher bar to trade
            }
        elif regime == Regime.LOW_VOLATILITY:
            return {
                "position_size": 1.2,       # Slightly larger
                "hold_time": 1.0,           # Normal
                "edge_threshold": 0.8,      # Lower (breakouts coming)
            }
        else:
            return {
                "position_size": 1.0,
                "hold_time": 1.0,
                "edge_threshold": 1.0,
            }
    
    def reset(self, asset: str = None):
        """Reset state for an asset or all assets."""
        if asset:
            self.price_histories.pop(asset, None)
            self.regimes.pop(asset, None)
            self.confidences.pop(asset, None)
        else:
            self.price_histories.clear()
            self.regimes.clear()
            self.confidences.clear()


# Global instance
_classifier: Optional[RegimeClassifier] = None


def get_regime_classifier() -> RegimeClassifier:
    """Get or create the global RegimeClassifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = RegimeClassifier()
    return _classifier


def classify_regime(asset: str, price: float, 
                    returns_1m: float = 0.0, returns_5m: float = 0.0,
                    volatility: float = 0.0, avg_volatility: float = 0.0) -> Tuple[Regime, float]:
    """Convenience function to classify regime."""
    return get_regime_classifier().update(asset, price, returns_1m, returns_5m, volatility, avg_volatility)


def get_regime_adjustments(asset: str) -> Dict[str, float]:
    """Get strategy adjustments for current regime."""
    return get_regime_classifier().get_regime_adjustment(asset)
