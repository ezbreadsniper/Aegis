#!/usr/bin/env python3
"""
VPIN (Volume-Synchronized Probability of Informed Trading) Detector.

Implements a real-time VPIN calculator to detect toxic flow and informed trading.
based on Easley et al. (2012).

Key Concepts:
- Volume Buckets: Updates happen based on volume traded, not time.
- Order Imbalance: Measures the disparity between buy and sell pressure.
- Toxicity: High VPIN indicates information asymmetry (insiders/toxic flow).

Usage:
    detector = VPINDetector(bucket_size=1000, window_size=50)
    detector.update(volume=500, direction="BUY")
    is_toxic = detector.is_toxic_flow()
"""
import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Dict


class ToxicityLevel(Enum):
    SAFE = "SAFE"
    ELEVATED = "ELEVATED"
    TOXIC = "TOXIC"
    CRITICAL = "CRITICAL"


class VPINDetector:
    """
    Real-time VPIN calculator using volume-synchronized buckets.
    """
    
    def __init__(
        self,
        bucket_size: float = 1000.0,  # Volume per bucket
        window_size: int = 50,        # Number of buckets for VPIN calculation
        toxicity_threshold: float = 0.75, # Threshold for toxic flow alert
    ):
        """
        Initialize VPIN detector.
        
        Args:
            bucket_size: Volume amount that triggers a new bucket (V)
            window_size: Number of buckets (n) to calculate moving average
            toxicity_threshold: VPIN value above which flow is considered toxic
        """
        self.bucket_size = bucket_size
        self.window_size = window_size
        self.toxicity_threshold = toxicity_threshold
        
        # Current bucket state
        self.current_volume = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        
        # History of buckets: (buy_volume, sell_volume)
        self.buckets = deque(maxlen=window_size)
        
        # Current VPIN value
        self.vpin = 0.0
        self.n_updates = 0
        
    def update(self, volume: float, direction: str) -> float:
        """
        Update detector with new trade or order flow.
        
        Args:
            volume: Size of the trade/order
            direction: "BUY" or "SELL" (or "UP"/"DOWN" mapped to BUY/SELL)
            
        Returns:
            Current VPIN value
        """
        remaining_vol = volume
        
        # Normalize direction
        is_buy = direction.upper() in ["BUY", "UP", "LONG"]
        
        # Process volume (potentially filling multiple buckets)
        while remaining_vol > 0:
            space_in_bucket = self.bucket_size - self.current_volume
            
            fill_amount = min(remaining_vol, space_in_bucket)
            
            self.current_volume += fill_amount
            if is_buy:
                self.current_buy_vol += fill_amount
            else:
                self.current_sell_vol += fill_amount
                
            remaining_vol -= fill_amount
            
            # If bucket full, close it and update VPIN
            if self.current_volume >= self.bucket_size:
                self._close_bucket()
                
        return self.vpin
    
    def _close_bucket(self):
        """Finalize current bucket and recalculate VPIN."""
        # Add bucket to history
        self.buckets.append((self.current_buy_vol, self.current_sell_vol))
        
        # Reset current bucket
        self.current_volume = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        
        # Calculate VPIN if we have enough history
        self._calculate_vpin()
        self.n_updates += 1
        
    def _calculate_vpin(self):
        """
        Calculate VPIN based on bucket history.
        
        VPIN = Sum(|BuyVol - SellVol|) / Sum(TotalVol) over window n
        """
        if len(self.buckets) < 1:
            self.vpin = 0.0
            return

        total_imbalance = 0.0
        total_volume = 0.0
        
        for buy_vol, sell_vol in self.buckets:
            total_imbalance += abs(buy_vol - sell_vol)
            total_volume += (buy_vol + sell_vol)
            
        if total_volume > 0:
            self.vpin = total_imbalance / total_volume
        else:
            self.vpin = 0.0
            
    def is_toxic_flow(self) -> bool:
        """Check if current flow is toxic."""
        return self.vpin > self.toxicity_threshold
        
    def get_toxicity_level(self) -> ToxicityLevel:
        """Classify current toxicity level."""
        if self.vpin > 0.85:
            return ToxicityLevel.CRITICAL
        elif self.vpin > 0.75:
            return ToxicityLevel.TOXIC
        elif self.vpin > 0.50:
            return ToxicityLevel.ELEVATED
        return ToxicityLevel.SAFE
        
    def reset(self):
        """Reset detector state."""
        self.buckets.clear()
        self.current_volume = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        self.vpin = 0.0
        

# Global cache for singleton access per market
_detectors: Dict[str, VPINDetector] = {}

def get_vpin_detector(
    market_id: str, 
    bucket_size: float = 1000.0,
    toxicity_threshold: float = 0.75
) -> VPINDetector:
    """Get or create VPIN detector for a specific market."""
    if market_id not in _detectors:
        _detectors[market_id] = VPINDetector(
            bucket_size=bucket_size, 
            toxicity_threshold=toxicity_threshold
        )
    return _detectors[market_id]

def clear_detector(market_id: str):
    """Remove detector for a market (e.g. when closed)."""
    if market_id in _detectors:
        del _detectors[market_id]


# =============================================================================
# VPIN 2.0: Order Flow Imbalance (OFI) Based Detector
# =============================================================================

class VPINDetectorV2:
    """
    VPIN 2.0: Calculates VPIN based on Order Flow Imbalance (OFI) and EMA smoothing.
    
    This is a superior proxy for toxicity when a direct trade feed is unavailable.
    Instead of using static orderbook depth (which causes false positives in
    directional markets), this detector uses the CHANGE in depth (OFI).
    
    Key Features:
    - OFI-based: Measures intent, not just size.
    - EMA Smoothing: Filters microstructure noise (quote stuffing).
    - Signed OFI: Enables "Toxic Alpha" piggybacking strategies.
    - CDF Normalization: Dynamic thresholds based on historical percentile.
    
    Usage:
        detector = VPINDetectorV2(window_size=50, alpha=0.1)
        vpin, signed_ofi = detector.update(bid_vol=1000, ask_vol=800)
        cdf_percentile = detector.get_vpin_cdf()
    """
    
    def __init__(self, window_size: int = 50, alpha: float = 0.1):
        """
        Initialize VPIN 2.0 detector.
        
        Args:
            window_size: Number of ticks for rolling normalization.
            alpha: EMA smoothing factor (0.1 = slow, 0.3 = fast).
        """
        self.window_size = window_size
        self.alpha = alpha
        
        # Previous tick state for delta calculation
        self.prev_bid_vol = 0.0
        self.prev_ask_vol = 0.0
        
        # OFI history for normalization
        self.ofi_history: deque = deque(maxlen=window_size)
        
        # VPIN history for CDF calculation
        self.vpin_history: deque = deque(maxlen=100)
        
        # Current state
        self.vpin_ema = 0.5  # Initialize at neutral
        self.signed_ofi = 0.0  # For toxic alpha direction
        self.n_updates = 0
        
    def update(self, bid_vol: float, ask_vol: float) -> tuple:
        """
        Update detector with current orderbook state.
        
        Args:
            bid_vol: Current volume at best bid.
            ask_vol: Current volume at best ask.
            
        Returns:
            Tuple of (vpin_ema, signed_ofi)
        """
        # 1. Calculate Order Flow Imbalance (OFI)
        delta_bid = bid_vol - self.prev_bid_vol
        delta_ask = ask_vol - self.prev_ask_vol
        ofi = delta_bid - delta_ask  # Positive = Buy Pressure, Negative = Sell Pressure
        
        # Store previous state
        self.prev_bid_vol = bid_vol
        self.prev_ask_vol = ask_vol
        
        # 2. Store signed OFI for "Toxic Alpha" direction
        self.signed_ofi = ofi
        
        # 3. Normalize OFI magnitude to [0, 1] range for VPIN
        self.ofi_history.append(abs(ofi))
        max_ofi = max(self.ofi_history) if self.ofi_history else 1.0
        current_vpin_raw = abs(ofi) / max_ofi if max_ofi > 0 else 0.5
        
        # 4. Apply EMA Smoothing (Microstructure Noise Filter)
        self.vpin_ema = (self.alpha * current_vpin_raw) + ((1 - self.alpha) * self.vpin_ema)
        
        # 5. Store for CDF calculation
        self.vpin_history.append(self.vpin_ema)
        self.n_updates += 1
        
        return self.vpin_ema, self.signed_ofi
    
    def get_vpin_cdf(self) -> float:
        """
        Get the CDF percentile of the current VPIN value.
        
        Returns:
            Percentile (0-100) of current VPIN relative to history.
            95+ indicates historically extreme toxicity.
        """
        if len(self.vpin_history) < 10:
            return 50.0  # Not enough data, return neutral
        
        sorted_history = sorted(self.vpin_history)
        # Find percentile rank
        rank = sum(1 for v in sorted_history if v <= self.vpin_ema)
        percentile = (rank / len(sorted_history)) * 100
        return percentile
    
    def get_toxicity_level(self) -> ToxicityLevel:
        """Classify current toxicity level using CDF-based dynamic thresholds."""
        cdf = self.get_vpin_cdf()
        if cdf > 99:
            return ToxicityLevel.CRITICAL
        elif cdf > 95:
            return ToxicityLevel.TOXIC
        elif cdf > 75:
            return ToxicityLevel.ELEVATED
        return ToxicityLevel.SAFE
    
    def reset(self):
        """Reset detector state."""
        self.prev_bid_vol = 0.0
        self.prev_ask_vol = 0.0
        self.ofi_history.clear()
        self.vpin_history.clear()
        self.vpin_ema = 0.5
        self.signed_ofi = 0.0


# Global cache for V2 detectors
_detectors_v2: Dict[str, VPINDetectorV2] = {}

def get_vpin_detector_v2(
    market_id: str,
    window_size: int = 50,
    alpha: float = 0.1
) -> VPINDetectorV2:
    """Get or create VPIN 2.0 detector for a specific market."""
    if market_id not in _detectors_v2:
        _detectors_v2[market_id] = VPINDetectorV2(
            window_size=window_size,
            alpha=alpha
        )
    return _detectors_v2[market_id]

def clear_detector_v2(market_id: str):
    """Remove V2 detector for a market (e.g. when closed)."""
    if market_id in _detectors_v2:
        del _detectors_v2[market_id]
