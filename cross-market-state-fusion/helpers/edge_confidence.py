#!/usr/bin/env python3
"""
Edge Confidence Gate for "When NOT to Trade" Logic.

This module implements the EdgeConfidenceGate which determines whether
trading conditions warrant taking a position. Based on research showing
that filtering low-edge situations significantly improves win rates.

Key factors considered:
1. Order book imbalance strength
2. Trade flow clarity
3. Spread tightness
4. Funding rate extremes (reversal signals)
5. Liquidation pressure
6. Volatility regime
7. Recent performance (streak detection)
"""
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque


@dataclass
class EdgeConfidenceGate:
    """
    Decide if conditions merit trading at all.
    
    This gate filters out low-edge situations to:
    - Reduce losing trades from noise
    - Preserve capital during uncertain conditions
    - Prevent overtrading in choppy markets
    """
    
    # Thresholds
    min_edge_confidence: float = 0.35  # Base threshold
    high_vol_adjustment: float = 0.10  # Add to threshold in high vol
    drawdown_adjustment: float = 0.10  # Add when in drawdown
    spoof_adjustment: float = 0.15     # Add when spoofing detected
    
    # State tracking
    recent_pnls: deque = field(default_factory=lambda: deque(maxlen=20))
    current_drawdown: float = 0.0
    peak_pnl: float = 0.0
    
    # Hard limits
    min_time_remaining: float = 0.033  # Never trade <30 seconds left
    
    def update_pnl(self, pnl: float, total_pnl: float) -> None:
        """Update performance tracking after a trade."""
        self.recent_pnls.append(pnl)
        
        # Track drawdown
        if total_pnl > self.peak_pnl:
            self.peak_pnl = total_pnl
        self.current_drawdown = (self.peak_pnl - total_pnl) / max(1.0, self.peak_pnl)
    

    
    @property
    def win_streak(self) -> int:
        """Count consecutive wins from the end."""
        streak = 0
        for pnl in reversed(self.recent_pnls):
            if pnl > 0:
                streak += 1
            else:
                break
        return streak
    
    def should_trade(
        self,
        edge_confidence: float,
        spread: float,
        time_remaining: float,
        vol_regime: float,
        spoof_probability: float = 0.0,
        probability: float = 0.5,
        trade_size: float = 25.0,
    ) -> tuple[bool, str]:
        """
        Determine if conditions warrant trading.
        
        v7.5 FIX: Minimal filtering approach based on EarnHFT research.
        
        Philosophy: Keep only HARD safety constraints. Remove soft preference
        adjustments (volatility, drawdown, spoof, fee threshold) so the RL agent
        can LEARN when not to trade via sparse PnL rewards.
        
        Research shows action masking should only block INVALID actions,
        not suboptimal ones. The agent should learn the correlation between
        market conditions and outcomes through experience.
        
        Args:
            edge_confidence: Computed edge score [0, 1] from MarketState
            spread: Current bid-ask spread
            time_remaining: Fraction of time remaining [0, 1]
            vol_regime: Volatility regime indicator (0=low, 1=high) - UNUSED now
            spoof_probability: Probability of spoofing [0, 1] - UNUSED now
            probability: Current market probability [0, 1] - UNUSED now
            trade_size: Trade size in dollars - UNUSED now
            training: If True, disable loss streak circuit breaker for exploration
            
        Returns:
            (should_trade, reason) tuple
        """
        # === HARD SAFETY CONSTRAINTS ONLY ===
        
        # 1. Spread too wide to execute profitably (truly unexecutable)
        if spread > 0.20:  # 20% is extreme, relaxed from 15%
            return False, f"spread_unexecutable ({spread:.2%})"
        
        # 2. Too close to expiry to manage position
        if time_remaining < self.min_time_remaining:  # < 30 sec
            return False, f"no_time_to_close ({time_remaining*15*60:.0f}s left)"
        

        
        # 4. Minimal edge sanity check - very low bar
        MIN_EDGE = 0.10  # Just ensure orderbook isn't completely empty
        if edge_confidence < MIN_EDGE:
            return False, f"no_signal ({edge_confidence:.2f} < {MIN_EDGE})"
        
        # === ALL OTHER CONDITIONS: LET THE AGENT LEARN ===
        # The agent sees these in its state vector:
        # - edge_confidence, vpin, spoof_probability, vol_regime, spread
        # - current_fee_rate (added in v7.4)
        # Through sparse PnL rewards, it will learn when to HOLD.
        
        return True, "conditions_acceptable"
    
    def get_position_size_multiplier(
        self,
        edge_confidence: float,
        policy_confidence: float = 1.0,
    ) -> float:
        """
        Get position size multiplier based on confidence levels.
        
        Higher edge confidence and policy confidence = larger positions.
        
        Args:
            edge_confidence: Edge score from market conditions [0, 1]
            policy_confidence: Model's certainty in action [0, 1]
            
        Returns:
            Size multiplier [0.25, 1.5]
        """
        base = 0.5
        
        # Scale by edge confidence
        edge_factor = 0.25 + (edge_confidence * 0.75)  # [0.25, 1.0]
        
        # Scale by policy confidence
        policy_factor = 0.5 + (policy_confidence * 0.5)  # [0.5, 1.0]
        
        # Boost on winning streak, reduce on losing streak
        streak_factor = 1.0
        if self.win_streak >= 3:
            streak_factor = 1.2  # Increase size after wins
        elif self.loss_streak >= 2:
            streak_factor = 0.7  # Reduce size after losses
        
        multiplier = base * edge_factor * policy_factor * streak_factor
        
        return max(0.25, min(1.5, multiplier))


# Global instance for easy access
_gate: Optional[EdgeConfidenceGate] = None


def get_edge_gate() -> EdgeConfidenceGate:
    """Get or create the global EdgeConfidenceGate instance."""
    global _gate
    if _gate is None:
        _gate = EdgeConfidenceGate()
    return _gate


def reset_edge_gate() -> None:
    """Reset the global EdgeConfidenceGate instance."""
    global _gate
    _gate = None


# Convenience functions
def should_trade(
    edge_confidence: float,
    spread: float,
    time_remaining: float,
    vol_regime: float,
    spoof_probability: float = 0.0,
) -> tuple[bool, str]:
    """Check if conditions warrant trading. See EdgeConfidenceGate.should_trade()."""
    return get_edge_gate().should_trade(
        edge_confidence, spread, time_remaining, vol_regime, spoof_probability
    )


def update_trade_result(pnl: float, total_pnl: float) -> None:
    """Update the gate with trade result."""
    get_edge_gate().update_pnl(pnl, total_pnl)


def get_size_multiplier(edge_confidence: float, policy_confidence: float = 1.0) -> float:
    """Get position size multiplier based on confidence."""
    return get_edge_gate().get_position_size_multiplier(edge_confidence, policy_confidence)
