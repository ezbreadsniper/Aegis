#!/usr/bin/env python3
"""
Spoofing Detection Module.

Detects fake liquidity and order book manipulation patterns:
1. Large orders that appear and disappear quickly
2. Orders placed just outside the touch (layering)
3. High cancellation-to-fill ratios
4. Sudden order book changes without price movement

Research basis:
- Spoofing patterns in crypto are similar to traditional markets
- Detection improves trade filtering by avoiding trapped positions
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


@dataclass
class OrderbookSnapshot:
    """Snapshot of orderbook state for comparison."""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    total_bid_size: float = 0.0
    total_ask_size: float = 0.0
    
    def __post_init__(self):
        self.total_bid_size = sum(size for _, size in self.bids[:10])
        self.total_ask_size = sum(size for _, size in self.asks[:10])


@dataclass  
class SpoofingDetector:
    """
    Detect spoofing and fake liquidity patterns.
    
    Maintains rolling history of order book snapshots and analyzes:
    - Order appearance/disappearance velocity
    - Size changes at key price levels
    - Layering patterns (stacked orders just outside touch)
    """
    
    # Detection parameters
    snapshot_interval: float = 0.5  # Seconds between snapshots
    history_length: int = 60  # Keep ~30 seconds of history
    large_order_threshold: float = 0.15  # 15% of total side = large
    disappearance_threshold: float = 0.5  # 50% size drop = suspicious
    layering_depth: int = 5  # Check top 5 price levels for layering
    
    # v7.3: EMA smoothing to reduce false positives
    ema_alpha: float = 0.3  # Higher = more responsive, lower = smoother
    
    # State per market
    histories: Dict[str, deque] = field(default_factory=dict)
    last_snapshots: Dict[str, OrderbookSnapshot] = field(default_factory=dict)
    spoof_scores: Dict[str, float] = field(default_factory=dict)
    
    def update(self, market_id: str, bids: List[Tuple[float, float]], 
               asks: List[Tuple[float, float]]) -> float:
        """
        Update with new orderbook data and return spoof probability.
        
        Args:
            market_id: Unique market identifier
            bids: List of (price, size) tuples, best bid first
            asks: List of (price, size) tuples, best ask first
            
        Returns:
            Spoof probability [0, 1]
        """
        now = time.time()
        
        # Initialize history if needed
        if market_id not in self.histories:
            self.histories[market_id] = deque(maxlen=self.history_length)
        
        # Create snapshot
        snapshot = OrderbookSnapshot(
            timestamp=now,
            bids=bids[:10],
            asks=asks[:10]
        )
        
        # Calculate spoof score
        spoof_score = self._calculate_spoof_score(market_id, snapshot)
        
        # Store snapshot
        self.histories[market_id].append(snapshot)
        self.last_snapshots[market_id] = snapshot
        self.spoof_scores[market_id] = spoof_score
        
        return spoof_score
    
    def _calculate_spoof_score(self, market_id: str, current: OrderbookSnapshot) -> float:
        """Calculate composite spoof probability score (v7.3 Bulletproof).
        
        Based on Fabre & Challet (2025) and spoofing signature research.
        Uses 6 signals with EMA smoothing to reduce false positives.
        """
        history = self.histories.get(market_id)
        if not history or len(history) < 5:
            return 0.0
        
        scores = []
        
        # === ORIGINAL SIGNALS (40% weight) ===
        # 1. Large order disappearance detection (20%)
        disappearance_score = self._detect_disappearing_orders(history, current)
        scores.append(disappearance_score * 0.20)
        
        # 2. Layering pattern detection (10%)
        layering_score = self._detect_layering(current)
        scores.append(layering_score * 0.10)
        
        # 3. Size volatility at top of book (10%)
        volatility_score = self._detect_size_volatility(history)
        scores.append(volatility_score * 0.10)
        
        # === v7.3 NEW SIGNALS (60% weight) ===
        # 4. Posting distance anomaly - Fabre & Challet 2025 (25%)
        posting_score = self._detect_posting_distance_anomaly(current)
        scores.append(posting_score * 0.25)
        
        # 5. Flickering detection - rapid L1 changes (20%)
        flickering_score = self._detect_flickering(history)
        scores.append(flickering_score * 0.20)
        
        # 6. Imbalance divergence - depth vs price movement (15%)
        divergence_score = self._detect_imbalance_divergence(history, current)
        scores.append(divergence_score * 0.15)
        
        raw_score = min(1.0, sum(scores))
        
        # v7.3: EMA smoothing to reduce false positives
        old_score = self.spoof_scores.get(market_id, 0.0)
        smoothed_score = self.ema_alpha * raw_score + (1 - self.ema_alpha) * old_score
        
        return smoothed_score
    
    def _detect_disappearing_orders(self, history: deque, 
                                     current: OrderbookSnapshot) -> float:
        """Detect large orders that appear and disappear quickly."""
        if len(history) < 3:
            return 0.0
        
        # Compare to snapshot from 2-3 seconds ago
        compare_idx = max(0, len(history) - 6)  # ~3 seconds at 0.5s intervals
        old = history[compare_idx]
        
        # Track large orders that disappeared
        disappearance_count = 0
        
        # Check bids
        old_large_bids = {price: size for price, size in old.bids 
                         if size > old.total_bid_size * self.large_order_threshold}
        for price, old_size in old_large_bids.items():
            current_size = next((s for p, s in current.bids if abs(p - price) < 0.001), 0)
            if current_size < old_size * (1 - self.disappearance_threshold):
                disappearance_count += 1
        
        # Check asks
        old_large_asks = {price: size for price, size in old.asks
                         if size > old.total_ask_size * self.large_order_threshold}
        for price, old_size in old_large_asks.items():
            current_size = next((s for p, s in current.asks if abs(p - price) < 0.001), 0)
            if current_size < old_size * (1 - self.disappearance_threshold):
                disappearance_count += 1
        
        # More disappearances = higher spoof probability
        return min(1.0, disappearance_count * 0.25)
    
    def _detect_layering(self, snapshot: OrderbookSnapshot) -> float:
        """
        Detect layering pattern: stacked orders increasing in size away from touch.
        
        Spoofing often shows pyramid patterns where each level has more size,
        designed to create false impression of support/resistance.
        """
        if len(snapshot.bids) < self.layering_depth or len(snapshot.asks) < self.layering_depth:
            return 0.0
        
        layering_score = 0.0
        
        # Check for increasing bid sizes (deeper = larger)
        bid_sizes = [size for _, size in snapshot.bids[:self.layering_depth]]
        if all(bid_sizes[i] <= bid_sizes[i+1] for i in range(len(bid_sizes)-1)):
            if bid_sizes[-1] > bid_sizes[0] * 3:  # Significant pyramid
                layering_score += 0.5
        
        # Check for increasing ask sizes
        ask_sizes = [size for _, size in snapshot.asks[:self.layering_depth]]
        if all(ask_sizes[i] <= ask_sizes[i+1] for i in range(len(ask_sizes)-1)):
            if ask_sizes[-1] > ask_sizes[0] * 3:
                layering_score += 0.5
        
        return layering_score
    
    def _detect_size_volatility(self, history: deque) -> float:
        """Detect unusually high size volatility at top of book."""
        if len(history) < 10:
            return 0.0
        
        # Get L1 bid sizes over time
        bid_sizes = [snap.bids[0][1] if snap.bids else 0 for snap in history]
        ask_sizes = [snap.asks[0][1] if snap.asks else 0 for snap in history]
        
        # Calculate coefficient of variation
        def cv(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            if mean == 0:
                return 0
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return (variance ** 0.5) / mean
        
        bid_cv = cv(bid_sizes[-10:])
        ask_cv = cv(ask_sizes[-10:])
        
        # High CV = high volatility = potential manipulation
        avg_cv = (bid_cv + ask_cv) / 2
        
        # CV > 1.0 is abnormally volatile
        return min(1.0, avg_cv / 1.5)
    
    # === v7.3 NEW DETECTION METHODS ===
    
    def _detect_posting_distance_anomaly(self, snapshot: OrderbookSnapshot) -> float:
        """Detect large orders placed far from touch (Fabre & Challet 2025).
        
        Key finding: "31% of large orders could spoof the market"
        Posting distance is CRITICAL for spoofing detection.
        """
        if not snapshot.bids or not snapshot.asks or len(snapshot.bids) < 3:
            return 0.0
        
        best_bid = snapshot.bids[0][0]
        best_ask = snapshot.asks[0][0]
        mid = (best_bid + best_ask) / 2
        
        if mid == 0:
            return 0.0
        
        spoof_score = 0.0
        
        # Check for large orders far from touch on bid side
        for i, (price, size) in enumerate(snapshot.bids[1:5]):  # Levels 2-5
            distance = abs(best_bid - price) / mid  # Normalized distance
            # Large order (>25% of total) placed far from touch (>2% away)
            if size > snapshot.total_bid_size * 0.25 and distance > 0.02:
                spoof_score += 0.25
        
        # Check for large orders far from touch on ask side
        for i, (price, size) in enumerate(snapshot.asks[1:5]):
            distance = abs(price - best_ask) / mid
            if size > snapshot.total_ask_size * 0.25 and distance > 0.02:
                spoof_score += 0.25
        
        return min(1.0, spoof_score)
    
    def _detect_flickering(self, history: deque) -> float:
        """Detect rapid order placement/cancellation patterns (flickering).
        
        Signature: OrderLifetime < 100ms AND Size > 10x average indicates spoofing.
        We approximate by counting L1 price level changes.
        """
        if len(history) < 10:
            return 0.0
        
        # Count L1 price changes in recent history
        l1_changes = 0
        recent = list(history)[-10:]
        
        for i in range(1, len(recent)):
            # Bid L1 changed
            if recent[i].bids and recent[i-1].bids:
                if abs(recent[i].bids[0][0] - recent[i-1].bids[0][0]) > 0.0001:
                    l1_changes += 1
            # Ask L1 changed
            if recent[i].asks and recent[i-1].asks:
                if abs(recent[i].asks[0][0] - recent[i-1].asks[0][0]) > 0.0001:
                    l1_changes += 1
        
        # High L1 change rate = flickering
        max_possible_changes = (len(recent) - 1) * 2  # Both bid and ask
        if max_possible_changes == 0:
            return 0.0
        
        change_rate = l1_changes / max_possible_changes
        
        if change_rate > 0.6:  # >60% of ticks have L1 changes
            return 1.0
        elif change_rate > 0.4:
            return 0.5
        elif change_rate > 0.25:
            return 0.25
        return 0.0
    
    def _detect_imbalance_divergence(self, history: deque, current: OrderbookSnapshot) -> float:
        """Detect when depth imbalance doesn't match price movement.
        
        Signature: DepthImbalance > 0.8 AND PriceImpact < 0.1 * Expected = SPOOFING
        Strong bid imbalance but price dropping = fake bids
        Strong ask imbalance but price rising = fake asks
        """
        if len(history) < 10 or not current.bids or not current.asks:
            return 0.0
        
        # Current depth imbalance
        total = current.total_bid_size + current.total_ask_size
        if total == 0:
            return 0.0
        imbalance = (current.total_bid_size - current.total_ask_size) / total
        
        # Get old snapshot for price comparison
        old = list(history)[0]
        if not old.bids or not old.asks:
            return 0.0
        
        # Calculate mid-price change
        old_mid = (old.bids[0][0] + old.asks[0][0]) / 2
        new_mid = (current.bids[0][0] + current.asks[0][0]) / 2
        
        if old_mid == 0:
            return 0.0
        
        price_change = (new_mid - old_mid) / old_mid
        
        # Strong bid imbalance but price going down = spoofing
        # Strong ask imbalance but price going up = spoofing
        if abs(imbalance) > 0.6:
            if (imbalance > 0.4 and price_change < -0.001) or \
               (imbalance < -0.4 and price_change > 0.001):
                return 0.8
            elif (imbalance > 0.3 and price_change < -0.0005) or \
                 (imbalance < -0.3 and price_change > 0.0005):
                return 0.4
        
        return 0.0
    
    def get_spoof_probability(self, market_id: str) -> float:
        """Get current spoof probability for a market."""
        return self.spoof_scores.get(market_id, 0.0)
    
    def reset(self, market_id: str = None):
        """Reset state for a market or all markets."""
        if market_id:
            self.histories.pop(market_id, None)
            self.last_snapshots.pop(market_id, None)
            self.spoof_scores.pop(market_id, None)
        else:
            self.histories.clear()
            self.last_snapshots.clear()
            self.spoof_scores.clear()


# Global instance
_detector: Optional[SpoofingDetector] = None


def get_spoof_detector() -> SpoofingDetector:
    """Get or create the global SpoofingDetector instance."""
    global _detector
    if _detector is None:
        _detector = SpoofingDetector()
    return _detector


def update_spoof_detection(market_id: str, bids: List, asks: List) -> float:
    """Update and return spoof probability. Convenience function."""
    return get_spoof_detector().update(market_id, bids, asks)


def get_spoof_probability(market_id: str) -> float:
    """Get current spoof probability for market."""
    return get_spoof_detector().get_spoof_probability(market_id)
