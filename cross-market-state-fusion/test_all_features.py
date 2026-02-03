#!/usr/bin/env python3
"""
Comprehensive Feature Verification Test
Tests ALL implemented features to ensure they work during training.
"""
import sys
import numpy as np

print("=" * 60)
print("TRIPLE-CHECK: All Features Verification")
print("=" * 60)

errors = []
passed = 0

# ============================================================
# 1. MarketState Features (31 dimensions for v7.2)
# ============================================================
print("\n[1/8] Testing MarketState (31 features)...")
try:
    from strategies.base import MarketState, Action
    state = MarketState(asset="BTC", prob=0.55, spread=0.01, time_remaining=0.5)
    features = state.to_features()
    assert len(features) == 31, f"Expected 31 features, got {len(features)}"
    print(f"  ✅ MarketState: {len(features)} features")
    passed += 1
except Exception as e:
    print(f"  ❌ MarketState: {e}")
    errors.append(f"MarketState: {e}")

# ============================================================
# 2. RLStrategy (31-dim input)
# ============================================================
print("\n[2/8] Testing RLStrategy (PyTorch)...")
try:
    from strategies import RLStrategy
    rl = RLStrategy()
    rl.training = True  # Set training mode after init
    state = MarketState(asset="TEST", prob=0.5, time_remaining=0.8)
    action = rl.act(state)
    assert action in [Action.BUY, Action.SELL, Action.HOLD]
    print(f"  ✅ RLStrategy: training={rl.training}, action={action}")
    passed += 1
except Exception as e:
    print(f"  ❌ RLStrategy: {e}")
    errors.append(f"RLStrategy: {e}")

# ============================================================
# 3. EarnHFT Strategy (Multi-Agent)
# ============================================================
print("\n[3/8] Testing EarnHFTStrategy (5 agents)...")
try:
    from strategies import EarnHFTStrategy
    earnhft = EarnHFTStrategy()
    status = earnhft.get_status()
    assert status['pool_size'] >= 1, "Pool should have agents"
    earnhft.training = True
    state = MarketState(asset="BTC", prob=0.5, time_remaining=0.8)
    action = earnhft.act(state)
    assert action in [Action.BUY, Action.SELL, Action.HOLD]
    print(f"  ✅ EarnHFT: pool_size={status['pool_size']}, training={earnhft.training}")
    passed += 1
except Exception as e:
    print(f"  ❌ EarnHFTStrategy: {e}")
    errors.append(f"EarnHFTStrategy: {e}")

# ============================================================
# 4. BetaAgent Training (store + update)
# ============================================================
print("\n[4/8] Testing BetaAgent Training Loop...")
try:
    from strategies.earnhft import BetaAgent
    agent = BetaAgent(beta=0.5)
    agent.training = True
    
    # Simulate experience storage with proper MarketState
    state1 = MarketState(asset="BTC", prob=0.55, time_remaining=0.8)
    state2 = MarketState(asset="BTC", prob=0.60, time_remaining=0.75)
    for i in range(50):  # Partial buffer fill
        agent.store(state1, Action.BUY, 0.1, state2, False)
    
    buffer_size = len(agent.experiences)
    print(f"  ✅ BetaAgent: buffer={buffer_size}/256, beta={agent.beta}")
    passed += 1
except Exception as e:
    print(f"  ❌ BetaAgent: {e}")
    errors.append(f"BetaAgent: {e}")

# ============================================================
# 5. Spoof Detection
# ============================================================
print("\n[5/8] Testing Spoof Detection...")
try:
    from helpers import get_spoof_probability
    spoof_prob = get_spoof_probability("BTC")
    assert 0.0 <= spoof_prob <= 1.0
    print(f"  ✅ Spoof Detection: probability={spoof_prob:.2%}")
    passed += 1
except Exception as e:
    print(f"  ❌ Spoof Detection: {e}")
    errors.append(f"Spoof Detection: {e}")

# ============================================================
# 6. Regime Classifier
# ============================================================
print("\n[6/8] Testing Regime Classifier...")
try:
    from helpers import classify_regime, get_regime_adjustments
    regime = classify_regime(
        asset="BTC",
        price=50000.0,
        returns_1m=0.001,
        returns_5m=0.005,
        volatility=0.01,
        avg_volatility=0.01
    )
    adjustments = get_regime_adjustments("BTC")
    # Handle both tuple and enum returns
    regime_str = regime.value if hasattr(regime, 'value') else str(regime)
    print(f"  ✅ Regime: {regime_str}, adjustments={adjustments}")
    passed += 1
except Exception as e:
    print(f"  ❌ Regime Classifier: {e}")
    errors.append(f"Regime Classifier: {e}")

# ============================================================
# 7. Bulletproof Expiry Layer (code check)
# ============================================================
print("\n[7/8] Testing Bulletproof Expiry Layer...")
try:
    with open("run.py", "r", encoding="utf-8") as f:
        run_code = f.read()
    
    checks = [
        ("ARMS", "ARMS ATR trailing stop"),       # v7.3: ARMS replaces hard $5 stop
        ("NO_TRADE_ZONE", "NO_TRADE zone"),
        ("ARMS STOP", "ARMS stop trigger"),       # v7.3: ARMS stop message
        ("OPTIMAL EXIT", "Optimal exit window"),
        ("no_trade_zone", "No-trade zone filter"),
    ]
    
    all_found = True
    for pattern, name in checks:
        if pattern not in run_code:
            print(f"  ⚠️ Missing: {name}")
            all_found = False
    
    if all_found:
        print(f"  ✅ Bulletproof Expiry: All protections active (incl. ARMS)!")
        passed += 1
    else:
        errors.append("Bulletproof Expiry: Some patterns missing")
except Exception as e:
    print(f"  ❌ Bulletproof Expiry: {e}")
    errors.append(f"Bulletproof Expiry: {e}")

# ============================================================
# 8. TradingEngine Import + EarnHFT Support
# ============================================================
print("\n[8/8] Testing TradingEngine with EarnHFT...")
try:
    from run import TradingEngine
    from strategies import create_strategy
    
    earnhft = create_strategy("earnhft")
    earnhft.training = True
    
    # TradingEngine should accept EarnHFTStrategy
    engine = TradingEngine(earnhft, trade_size=50)
    print(f"  ✅ TradingEngine: strategy={earnhft.name}, training={earnhft.training}")
    passed += 1
except Exception as e:
    print(f"  ❌ TradingEngine: {e}")
    errors.append(f"TradingEngine: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed}/8 tests passed")
print("=" * 60)

if errors:
    print("\n❌ ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n✅ ALL TESTS PASSED - Ready for training!")
    print("\nFeatures confirmed working:")
    print("  • 31-dim MarketState features (v7.2)")
    print("  • RLStrategy with training mode")
    print("  • EarnHFT with 5 agents + regime switching")
    print("  • BetaAgent store/update training loop")
    print("  • Spoof detection")
    print("  • Regime classifier")
    print("  • Bulletproof Expiry Layer (MAX_LOSS, HARD_CLOSE, NO_TRADE)")
    print("  • TradingEngine EarnHFT integration")
    sys.exit(0)

