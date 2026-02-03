#!/usr/bin/env python3
"""
ARMS: Adaptive Risk Management System (v7.3)

Implements state-of-the-art stop-loss and position sizing:
1. ATR Trailing Stop - Dynamic stop that trails price
2. Hurst Regime Filter - Adjust multiplier for market regime
3. Fractional Kelly - Position sizing based on conviction
4. Stop-Adjusted Rewards - RL training with actual stop PnL

Based on research from:
- Anders et al., 2024 (DRL outperforms fixed stops)
- LuxAlgo 2025 (ATR trailing stops)
- Kaminski & Lo (regime-aware stops)
- Optimal Stop-Loss Analysis (3.6x ATR average optimal)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np


@dataclass
class ARMSPosition:
    """Track position with trailing stop state."""
    asset: str
    entry_price: float
    side: str  # "UP" or "DOWN"
    size: float
    highest_price: float  # For trailing stop (long)
    lowest_price: float   # For trailing stop (short)
    current_stop: float   # Current trailing stop level
    entry_atr: float      # ATR at entry time


@dataclass
class AdaptiveRiskManager:
    """
    ARMS: Adaptive Risk Management System.
    
    Replaces hard $5 stop-loss with:
    - ATR-based trailing stops (3.6x base multiplier)
    - Hurst exponent regime adjustment
    - Fractional Kelly position sizing
    """
    
    # ATR Parameters - TUNED FOR PREDICTION MARKETS
    # Prediction markets have HIGH volatility (0-1 range, 20-40% swings in 15min)
    # Much wider stops needed than crypto spot markets
    base_atr_multiplier: float = 8.0   # 8x ATR (was 3.6x for spot)
    atr_period: int = 14  # Number of bars for ATR calculation
    
    # Hurst Regime Thresholds
    hurst_trending_threshold: float = 0.55
    hurst_mean_reverting_threshold: float = 0.45
    
    # Regime-adjusted multipliers - WIDER for prediction markets
    trending_multiplier: float = 10.0   # Very wide for trends (was 4.1x)
    mean_reverting_multiplier: float = 6.0  # Still reasonably wide (was 3.1x)
    
    # Kelly parameters
    kelly_fraction: float = 0.25  # Use 25% Kelly (conservative)
    
    # State
    price_histories: Dict[str, deque] = field(default_factory=dict)
    atr_values: Dict[str, float] = field(default_factory=dict)
    hurst_values: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, ARMSPosition] = field(default_factory=dict)
    
    def update_price(self, asset: str, high: float, low: float, close: float):
        """Update price history for ATR and Hurst calculation."""
        if asset not in self.price_histories:
            self.price_histories[asset] = deque(maxlen=100)
        
        self.price_histories[asset].append({
            'high': high,
            'low': low,
            'close': close
        })
        
        # Recalculate ATR
        self.atr_values[asset] = self._calculate_atr(asset)
        
        # Recalculate Hurst (less frequently - every 20 bars)
        if len(self.price_histories[asset]) >= 50 and len(self.price_histories[asset]) % 20 == 0:
            self.hurst_values[asset] = self._calculate_hurst(asset)
    
    def _calculate_atr(self, asset: str) -> float:
        """Calculate Average True Range for an asset."""
        history = list(self.price_histories.get(asset, []))
        if len(history) < 2:
            return 0.01  # Default small ATR
        
        true_ranges = []
        for i in range(1, min(len(history), self.atr_period + 1)):
            h = history[-i]['high']
            l = history[-i]['low']
            prev_c = history[-i-1]['close'] if i < len(history) else history[-i]['close']
            
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            true_ranges.append(tr)
        
        # v7.5 Fix: Enforce minimum ATR floor to prevent 0.00 stops
        # Polymarket discrete ticks can lead to 0 volatility periods
        calculated_atr = np.mean(true_ranges) if true_ranges else 0.01
        return max(calculated_atr, 0.005)  # Floor at 0.5% (minimum half-cent risk)
    
    def _calculate_hurst(self, asset: str) -> float:
        """Calculate Hurst exponent for regime detection."""
        history = list(self.price_histories.get(asset, []))
        if len(history) < 50:
            return 0.5  # Default random walk
        
        closes = [bar['close'] for bar in history[-50:]]
        
        # R/S analysis for Hurst exponent
        n = len(closes)
        max_lag = n // 2
        lags = range(2, max_lag)
        
        try:
            tau = []
            for lag in lags:
                diffs = np.subtract(closes[lag:], closes[:-lag])
                tau.append(np.sqrt(np.std(diffs)))
            
            # Filter zeros
            valid = [(lags[i], tau[i]) for i in range(len(tau)) if tau[i] > 0]
            if len(valid) < 2:
                return 0.5
            
            log_lags = np.log([v[0] for v in valid])
            log_tau = np.log([v[1] for v in valid])
            
            # Linear regression
            poly = np.polyfit(log_lags, log_tau, 1)
            return max(0.0, min(1.0, poly[0]))
        except:
            return 0.5
    
    def get_atr_multiplier(self, asset: str) -> float:
        """Get regime-adjusted ATR multiplier."""
        hurst = self.hurst_values.get(asset, 0.5)
        
        if hurst > self.hurst_trending_threshold:
            return self.trending_multiplier
        elif hurst < self.hurst_mean_reverting_threshold:
            return self.mean_reverting_multiplier
        else:
            return self.base_atr_multiplier
    
    def get_atr(self, asset: str) -> float:
        """Get current ATR for an asset."""
        return self.atr_values.get(asset, 0.01)
    
    def get_hurst(self, asset: str) -> float:
        """Get current Hurst exponent for an asset."""
        return self.hurst_values.get(asset, 0.5)
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_since_entry: float,
        lowest_since_entry: float,
        atr: float,
        multiplier: float,
        side: str
    ) -> Tuple[float, bool]:
        """
        Calculate trailing stop level and check if triggered.
        
        v7.5 FIX: Corrected DOWN position logic.
        - UP: We profit when price goes UP. Stop trails below highest price.
              Triggered when price drops below stop.
        - DOWN: We profit when price goes DOWN. Stop trails above highest price
              (since we want to lock in gains as price falls). Actually, for DOWN
              tokens in binary markets, the DOWN token price goes UP when the
              underlying probability goes DOWN. So we trail below the highest
              DOWN token price, triggering when it drops.
        
        Returns:
            (stop_level, is_triggered)
        """
        if side == "UP":
            # Long UP position: stop trails below highest price
            # - highest_since_entry tracks highest UP token price
            # - Stop triggers when price falls too far from high
            stop = highest_since_entry - (atr * multiplier)
            triggered = current_price <= stop
        else:
            # Long DOWN position: stop trails below highest DOWN token price
            # - For DOWN tokens: token price = 1 - prob
            # - When prob drops, DOWN token price rises (good for us)
            # - We trail below the highest DOWN token price we've seen
            # - Stop triggers when DOWN token price falls (prob rises against us)
            # NOTE: highest_since_entry should be tracking highest DOWN token price
            stop = highest_since_entry - (atr * multiplier)
            triggered = current_price <= stop
        
        return stop, triggered
    
    def calculate_kelly_position(
        self,
        true_prob: float,
        market_price: float,
        bankroll: float,
        side: str
    ) -> float:
        """Calculate position size using Fractional Kelly Criterion."""
        # Determine if we're betting on UP or DOWN
        if side == "UP":
            p = true_prob
            b = (1 - market_price) / market_price if market_price > 0 else 0
        else:
            p = 1 - true_prob
            b = market_price / (1 - market_price) if market_price < 1 else 0
        
        if b <= 0:
            return 0
        
        q = 1 - p
        f_star = (b * p - q) / b
        
        # Use fractional Kelly and clamp to reasonable range
        if f_star <= 0:
            return 0
        
        position = bankroll * f_star * self.kelly_fraction
        
        # Clamp to 5-100% of bankroll
        return max(bankroll * 0.05, min(position, bankroll))
    
    def should_stop_out(
        self,
        asset: str,
        entry_price: float,
        current_price: float,
        highest_since_entry: float,
        lowest_since_entry: float,
        side: str
    ) -> Tuple[bool, float, str]:
        """
        Check if position should be stopped out.
        
        Returns:
            (should_stop, stop_price, reason)
        """
        atr = self.get_atr(asset)
        multiplier = self.get_atr_multiplier(asset)
        hurst = self.get_hurst(asset)
        
        stop, triggered = self.calculate_trailing_stop(
            entry_price, current_price,
            highest_since_entry, lowest_since_entry,
            atr, multiplier, side
        )
        
        if triggered:
            regime = "trending" if hurst > 0.55 else "mean_reverting" if hurst < 0.45 else "neutral"
            reason = f"ATR_TRAILING_{multiplier:.1f}x ({regime}, H={hurst:.2f})"
            return True, stop, reason
        
        return False, stop, ""


# Global instance
_arms: Optional[AdaptiveRiskManager] = None


def get_arms() -> AdaptiveRiskManager:
    """Get or create the global ARMS instance."""
    global _arms
    if _arms is None:
        _arms = AdaptiveRiskManager()
    return _arms


def update_arms_price(asset: str, high: float, low: float, close: float):
    """Update ARMS with new price data."""
    get_arms().update_price(asset, high, low, close)


def get_trailing_stop_info(
    asset: str,
    entry_price: float,
    current_price: float,
    highest_since_entry: float,
    lowest_since_entry: float,
    side: str
) -> Tuple[bool, float, str]:
    """Check if position should be stopped out."""
    return get_arms().should_stop_out(
        asset, entry_price, current_price,
        highest_since_entry, lowest_since_entry, side
    )


def get_kelly_position(
    true_prob: float,
    market_price: float,
    bankroll: float,
    side: str
) -> float:
    """Calculate Kelly-optimal position size."""
    return get_arms().calculate_kelly_position(true_prob, market_price, bankroll, side)
