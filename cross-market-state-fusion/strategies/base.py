#!/usr/bin/env python3
"""
Base classes for trading strategies.
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Action(Enum):
    HOLD = 0
    BUY = 1   # Buy UP token
    SELL = 2  # Sell UP token

    @property
    def is_buy(self) -> bool:
        return self == Action.BUY

    @property
    def is_sell(self) -> bool:
        return self == Action.SELL

    @property
    def size_multiplier(self) -> float:
        """Base 50% sizing for trades (adjusted by confidence in TradingEngine)."""
        return 0.5 if self in (Action.BUY, Action.SELL) else 0.0

    def get_confidence_size(self, prob: float) -> float:
        """
        Get position size multiplier based on probability extremeness.

        At extreme probabilities (near 0 or 1), we have higher edge due to
        asymmetric payoffs in binary markets. Scale size accordingly.

        Returns: size multiplier in [0.25, 1.0]
        """
        if self == Action.HOLD:
            return 0.0

        # Distance from 0.5 - higher = more extreme
        extremeness = abs(prob - 0.5) * 2  # [0, 1]

        # Scale from 0.25 (at 0.5) to 1.0 (at extremes)
        # More aggressive at extremes where edge is higher
        base = 0.25
        scale = 0.75  # max additional size

        return base + (scale * extremeness)


@dataclass
class MarketState:
    """Rich market state for 15-min trading decisions."""
    # Core
    asset: str
    prob: float  # Current UP probability
    time_remaining: float  # Fraction of 15 min left (0-1)

    # Orderbook - CRITICAL for 15-min
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    order_book_imbalance_l1: float = 0.0  # Top of book imbalance
    order_book_imbalance_l5: float = 0.0  # Depth imbalance (top 5 levels)

    # Price data
    binance_price: float = 0.0
    binance_change: float = 0.0  # % change since market open

    # History (last N observations)
    prob_history: List[float] = field(default_factory=list)

    # Position
    has_position: bool = False
    position_side: Optional[str] = None  # "UP" or "DOWN"
    position_pnl: float = 0.0  # Unrealized P&L

    # === 15-MIN FOCUSED FEATURES ===
    # Ultra-short momentum (most relevant for 15-min)
    returns_1m: float = 0.0
    returns_5m: float = 0.0
    returns_10m: float = 0.0  # Middle timeframe

    # Order flow - THIS IS THE EDGE
    trade_flow_imbalance: float = 0.0  # [-1, 1] buy vs sell pressure
    cvd: float = 0.0  # Cumulative volume delta
    cvd_acceleration: float = 0.0  # Is CVD speeding up?
    prev_cvd: float = 0.0  # For acceleration calc

    # Microstructure
    trade_intensity: float = 0.0  # Trades per second (rolling)
    large_trade_flag: float = 0.0  # Big order just hit? (0 or 1)
    trade_count: int = 0  # For intensity calc
    last_trade_time: float = 0.0

    # Volatility (short-term)
    realized_vol_5m: float = 0.0
    vol_expansion: float = 0.0  # Current vol vs recent average

    # === NEW HIGH-IMPACT SIGNALS ===
    # Funding rate (extreme values precede reversals 60%+ of time)
    funding_rate: float = 0.0  # Current 8h funding rate
    funding_rate_velocity: float = 0.0  # Change in funding rate
    
    # Open Interest (tracks leveraged positioning)
    open_interest_delta: float = 0.0  # OI change over recent period
    oi_price_divergence: float = 0.0  # OI up + price down = bearish, etc.
    
    # Liquidation pressure (where forced selling/buying happens)
    liquidation_pressure: float = 0.0  # Net liq pressure [-1, 1]
    liquidation_proximity: float = 0.0  # Proximity to liq clusters [-1, 1]
    
    # Edge confidence (for "when NOT to trade" logic)
    edge_confidence: float = 0.0  # Computed edge score [0, 1]
    
    # Spoofing/manipulation detection
    spoof_probability: float = 0.0  # Detected fake liquidity probability [0, 1]
    
    # === v7.2: VPIN TOXIC FLOW DETECTION (OFI-Based) ===
    vpin: float = 0.0  # Volume-Synchronized Probability of Informed Trading [0, 1]
    signed_ofi: float = 0.0  # Signed Order Flow Imbalance for "Toxic Alpha" piggybacking
    vpin_cdf: float = 50.0  # CDF percentile (0-100) for dynamic thresholds
    
    # === NEW DEEP RESEARCH FEATURES ===
    # Velocity acceleration (2nd derivative - precedes velocity by 2-5 ticks)
    velocity_acceleration: float = 0.0  # Rate of change of velocity
    prev_velocity: float = 0.0  # For acceleration calculation
    
    # Event-time phase (15-min markets have distinct phases)
    # 0-5min: "discovery", 5-10min: "consolidation", 10-15min: "resolution"
    event_time_phase: int = 0  # 0=discovery, 1=consolidation, 2=resolution
    
    # Probability Z-score (deviation from rolling mean)
    prob_zscore: float = 0.0  # Current prob vs 10-min mean/std
    
    # Whale detection (large trades > 95th percentile)
    whale_trade_flag: float = 0.0  # 1=whale buy, -1=whale sell, 0=no whale
    whale_trade_count: int = 0  # Recent whale trades

    # Regime context (only slow features worth keeping)
    vol_regime: float = 0.0  # High/low vol environment
    trend_regime: float = 0.0  # Trending or ranging
    
    # Regime multipliers (Meta-Control)
    regime_hold_time_mult: float = 1.0
    regime_size_mult: float = 1.0
    regime_edge_mult: float = 1.0
    
    # === NEW RESEARCH CROSS-ASSET FEATURES ===
    # ETH→BTC Lead-Lag (strongest cross-asset signal, -0.1178 correlation)
    eth_return_lag1: float = 0.0  # ETH return at t-1 for BTC predictions
    
    # Time-of-Day Features (volatility peak 14:00-16:00 UTC)
    utc_hour: int = 0  # Current UTC hour (0-23)
    is_volatility_peak: float = 0.0  # 1.0 if 14:00-16:00 UTC, else 0.0
    
    # Market Synergy (P(SOL up | BTC up) = 74.51%)
    market_synergy_score: float = 0.0  # Multi-asset correlation signal
    
    # === v7.4: FEE AWARENESS ===
    current_fee_rate: float = 0.0  # Current round-trip fee rate at this probability [0, 0.1]

    def to_features(self) -> np.ndarray:
        """Convert to feature vector for ML models. Returns 31 features (v7.5) normalized to [-1, 1].
        
        v7.5 FIX: Removed position_pnl to prevent data leakage. The agent was seeing
        unrealized PnL derived from current prices, which leaked future information.
        """
        velocity = self._velocity(3)  # Shorter window
        vol_5m = self._volatility(30)  # ~5 min of ticks

        # Spread as percentage
        spread_pct = self.spread / max(0.01, self.prob) if self.prob > 0 else 0.0

        # Helper to clamp values to [-1, 1]
        def clamp(x, min_val=-1.0, max_val=1.0):
            return max(min_val, min(max_val, x))

        return np.array([
            # Ultra-short momentum (3) - returns scaled and clamped
            # Typical returns are -0.02 to 0.02, so *50 maps to [-1, 1]
            clamp(self.returns_1m * 50),
            clamp(self.returns_5m * 50),
            clamp(self.returns_10m * 50),

            # Order flow - THE EDGE (4) - already [-1, 1] range mostly
            clamp(self.order_book_imbalance_l1),
            clamp(self.order_book_imbalance_l5),
            clamp(self.trade_flow_imbalance),
            clamp(self.cvd_acceleration * 10),  # CVD accel is small, scale up

            # Microstructure (3)
            clamp(spread_pct * 20),  # Spread ~0-5%, so *20 maps to [0,1]
            clamp(self.trade_intensity / 10),  # Normalize by typical max intensity
            self.large_trade_flag,  # Already 0 or 1

            # Volatility (2)
            clamp(vol_5m * 20),  # Vol ~0-5%, scale up
            clamp(self.vol_expansion),  # Typically [-1, 2], clamp it

            # === FUTURES SIGNALS (4) ===
            clamp(self.funding_rate * 500),  # Extreme funding precedes reversals
            clamp(self.open_interest_delta * 10),  # OI change
            clamp(self.oi_price_divergence),  # OI-price divergence
            clamp(self.liquidation_pressure),  # Liq pressure

            # === DEEP RESEARCH SIGNALS (4) ===
            # Velocity acceleration (2nd derivative - precedes moves)
            clamp(self.velocity_acceleration * 100),  # Small values, scale up
            # Event-time phase: normalized 0-2 to [-1, 1]
            (self.event_time_phase - 1.0),  # 0->-1, 1->0, 2->1
            # Probability Z-score: how extreme is current prob vs rolling mean
            clamp(self.prob_zscore / 2),  # Typical range [-2, 2]
            # Whale signal: net whale buying/selling
            clamp(self.whale_trade_flag),  # Already [-1, 1]

            # Position (3) - v7.5: Removed position_pnl (data leakage fix)
            float(self.has_position),  # 0 or 1
            1.0 if self.position_side == "UP" else (-1.0 if self.position_side == "DOWN" else 0.0),
            self.time_remaining,  # Already [0, 1]

            # Regime (2)
            self.vol_regime,  # 0 or 1
            self.trend_regime,  # 0 or 1
            
            # === NEW RESEARCH CROSS-ASSET FEATURES (2) ===
            # ETH→BTC Lead-Lag: strongest cross-asset signal (-0.1178 correlation)
            clamp(self.eth_return_lag1 * 50),  # Same scaling as momentum returns
            # Time-of-Day: volatility peak window (14:00-16:00 UTC)
            self.is_volatility_peak,  # Already 0 or 1
            
            # === v7.2: VPIN TOXIC FLOW DETECTION (2) ===
            self.vpin,  # Already [0, 1], higher = more toxic
            clamp(self.signed_ofi / 1000),  # Signed OFI for toxic alpha direction
            
            # === v7.3: SPOOF DETECTION RL INTEGRATION (1) ===
            self.spoof_probability,  # Already [0, 1], higher = more likely spoofing
            
            # === v7.4: FEE AWARENESS (1) ===
            clamp(self.current_fee_rate * 30),  # Fee rate ~0-5%, *30 maps to [0, 1.5]
        ], dtype=np.float32)

    def _velocity(self, window: int = 5) -> float:
        """Prob change over last N ticks."""
        if len(self.prob_history) < window:
            return 0.0
        return self.prob - self.prob_history[-window]

    def _volatility(self, window: int = 10) -> float:
        """Rolling std of prob."""
        if len(self.prob_history) < window:
            return 0.0
        recent = self.prob_history[-window:]
        return float(np.std(recent))

    def _momentum(self, window: int = 20) -> float:
        """Longer-term trend."""
        if len(self.prob_history) < window:
            return 0.0
        return self.prob - self.prob_history[-window]

    @property
    def near_expiry(self) -> bool:
        return self.time_remaining < 0.133  # < 2 min

    @property
    def very_near_expiry(self) -> bool:
        return self.time_remaining < 0.033  # < 30 sec


class Strategy(ABC):
    """Base class for all strategies."""

    def __init__(self, name: str):
        self.name = name
        self.training = False

    @abstractmethod
    def act(self, state: MarketState) -> Action:
        """Select action given current state."""
        pass

    def reset(self):
        """Reset any internal state (called between episodes/markets)."""
        pass

    def train(self):
        """Set to training mode."""
        self.training = True

    def eval(self):
        """Set to evaluation mode."""
        self.training = False

    def save(self, path: str):
        """Save strategy parameters."""
        pass

    def load(self, path: str):
        """Load strategy parameters."""
        pass
