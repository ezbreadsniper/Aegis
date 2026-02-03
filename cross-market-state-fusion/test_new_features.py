#!/usr/bin/env python3
"""Integration test for new features."""
import sys
sys.path.insert(0, ".")

print("=== Testing Imports ===")
try:
    from helpers import get_liq_signal, get_regime_classifier, classify_regime, get_regime_adjustments
    print("[OK] All helper imports work")
except ImportError as e:
    print(f"[FAIL] Helper import failed: {e}")
    sys.exit(1)

print("\n=== Testing MarketState Fields ===")
try:
    from strategies import MarketState
    s = MarketState(asset="BTC", prob=0.5, time_remaining=0.5)  # Provide required args
    
    assert hasattr(s, "regime_size_mult"), "Missing regime_size_mult"
    assert hasattr(s, "regime_hold_time_mult"), "Missing regime_hold_time_mult"
    assert hasattr(s, "regime_edge_mult"), "Missing regime_edge_mult"
    assert hasattr(s, "liquidation_proximity"), "Missing liquidation_proximity"
    print("[OK] MarketState has all new fields")
except Exception as e:
    print(f"[FAIL] MarketState check failed: {e}")
    sys.exit(1)

print("\n=== Testing Regime Classifier ===")
try:
    regime, conf = classify_regime("BTC", 90000.0, returns_1m=0.001, returns_5m=0.003, volatility=0.02, avg_volatility=0.015)
    print(f"[OK] Regime classified: {regime.value} (conf={conf:.2f})")
except Exception as e:
    print(f"[FAIL] classify_regime failed: {e}")
    sys.exit(1)

print("\n=== Testing Liquidation Heatmap ===")
try:
    from helpers import update_heatmap, get_liquidation_heatmap
    update_heatmap("BTC", 89000.0, 50000, "SELL")
    update_heatmap("BTC", 91000.0, 75000, "BUY")
    signal = get_liq_signal("BTC", 89500.0)
    print(f"[OK] Liq signal computed: {signal:.4f}")
except Exception as e:
    print(f"[FAIL] Liquidation heatmap failed: {e}")
    sys.exit(1)

print("\n=== All Tests Passed ===")
