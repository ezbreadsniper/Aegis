"""
Polymarket 15-min trading helpers.
"""
from .polymarket_api import (
    get_15m_markets,
    get_next_market,
    Market,
)
from .binance_wss import (
    BinanceStreamer,
    get_current_prices,
)
from .orderbook_wss import (
    OrderbookStreamer,
)
from .binance_futures import (
    FuturesStreamer,
    FuturesState,
    get_futures_snapshot,
)
from .training_logger import (
    TrainingLogger,
    get_logger,
    reset_logger,
)
from .polymarket_execution import (
    PolymarketExecutor,
    get_executor,
    is_live_trading_enabled,
    OrderResult,
    SafetyConfig,
)
from .edge_confidence import (
    EdgeConfidenceGate,
    get_edge_gate,
    reset_edge_gate,
    should_trade,
    update_trade_result,
    get_size_multiplier,
)
from .spoofing_detection import (
    SpoofingDetector,
    get_spoof_detector,
    update_spoof_detection,
    get_spoof_probability,
)
from .regime_classifier import (
    RegimeClassifier,
    Regime,
    get_regime_classifier,
    classify_regime,
    get_regime_adjustments,
)
from .whale_detection import (
    WhaleDetector,
    get_whale_detector,
    update_whale_detection,
    get_whale_signal,
    get_whale_count,
)
from .whale_tracker import (
    WhaleTracker,
    get_whale_tracker,
)
from .liquidation_heatmap import (
    LiquidationHeatmap,
    get_liquidation_heatmap,
    update_heatmap,
    get_liq_signal,
)
from .polymarket_fees import (
    calculate_fee,
    round_trip_fee,
    fee_threshold_for_trade,
    get_fee_info,
    should_skip_due_to_fees,
)

# Backwards compat
get_active_markets = get_15m_markets

__all__ = [
    "get_15m_markets",
    "get_active_markets",
    "get_next_market",
    "Market",
    "BinanceStreamer",
    "get_current_prices",
    "OrderbookStreamer",
    "FuturesStreamer",
    "FuturesState",
    "get_futures_snapshot",
    "TrainingLogger",
    "get_logger",
    "reset_logger",
    "PolymarketExecutor",
    "get_executor",
    "is_live_trading_enabled",
    "OrderResult",
    "SafetyConfig",
    "EdgeConfidenceGate",
    "get_edge_gate",
    "reset_edge_gate",
    "should_trade",
    "update_trade_result",
    "get_size_multiplier",
    "SpoofingDetector",
    "get_spoof_detector",
    "update_spoof_detection",
    "get_spoof_probability",
    "RegimeClassifier",
    "Regime",
    "get_regime_classifier",
    "classify_regime",
    "get_regime_adjustments",
    "WhaleDetector",
    "get_whale_detector",
    "update_whale_detection",
    "get_whale_signal",
    "get_whale_count",
    "WhaleTracker",
    "get_whale_tracker",
    # v7.4: Fee calculation
    "calculate_fee",
    "round_trip_fee",
    "fee_threshold_for_trade",
    "get_fee_info",
    "should_skip_due_to_fees",
]



