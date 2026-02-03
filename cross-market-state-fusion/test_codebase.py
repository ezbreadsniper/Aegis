#!/usr/bin/env python3
"""
Comprehensive Codebase Verification Script.

Tests all components of the trading bot including:
1. Base features (MarketState, Action)
2. RL strategies (RLStrategy, BetaAgent)
3. Helpers (whale_tracker, spoofing_detection, etc.)
4. EarnHFT system (AgentPool, Router, EarnHFTStrategy)
5. Dimension consistency across all components
"""
import sys
import traceback

def test_section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

def test_pass(msg: str):
    print(f"  ✓ {msg}")

def test_fail(msg: str, err: str = None):
    print(f"  ✗ FAIL: {msg}")
    if err:
        print(f"    Error: {err}")
    return False

all_passed = True

# ============================================================
# TEST 1: Base Imports
# ============================================================
test_section("1. Base Imports")
try:
    from strategies.base import Strategy, MarketState, Action
    test_pass("Strategy, MarketState, Action imported")
except Exception as e:
    all_passed = test_fail("Base imports", str(e))

# ============================================================
# TEST 2: MarketState Feature Dimensions
# ============================================================
test_section("2. MarketState Feature Dimensions")
try:
    state = MarketState(
        asset="BTC",
        prob=0.55,
        time_remaining=600,
    )
    features = state.to_features()
    expected_dim = 31  # v7.5: removed position_pnl (data leakage fix)
    if len(features) == expected_dim:
        test_pass(f"Feature vector length: {len(features)} (expected {expected_dim})")
    else:
        all_passed = test_fail(f"Feature vector wrong: {len(features)} != {expected_dim}")
except Exception as e:
    all_passed = test_fail("MarketState features", str(e))

# ============================================================
# TEST 3: MarketState Fields
# ============================================================
test_section("3. MarketState Fields (NEW RESEARCH)")
try:
    required_fields = [
        'eth_return_lag1',
        'utc_hour', 
        'is_volatility_peak',
        'market_synergy_score',
        'spoof_probability',
        'liquidation_pressure',
        'whale_trade_flag',
    ]
    missing = []
    for field in required_fields:
        if not hasattr(MarketState, field):
            missing.append(field)
    
    if not missing:
        test_pass(f"All {len(required_fields)} required fields present")
    else:
        all_passed = test_fail(f"Missing fields: {missing}")
except Exception as e:
    all_passed = test_fail("MarketState fields", str(e))

# ============================================================
# TEST 4: RLStrategy (PyTorch)
# ============================================================
test_section("4. RLStrategy (PyTorch backend)")
try:
    from strategies import RLStrategy
    rl = RLStrategy()
    if rl.input_dim == 31:
        test_pass(f"RLStrategy input_dim: {rl.input_dim}")
    else:
        all_passed = test_fail(f"RLStrategy input_dim wrong: {rl.input_dim} (expected 31)")
except Exception as e:
    all_passed = test_fail("RLStrategy", str(e))

# ============================================================
# TEST 5: RLStrategy Neural Networks
# ============================================================
test_section("5. RLStrategy Neural Networks")
try:
    import torch
    from strategies.rl_pytorch import Actor, Critic
    
    actor = Actor(input_dim=31)
    critic = Critic(input_dim=31)
    
    current_state = torch.zeros((1, 31))
    temporal_state = torch.zeros((1, 5 * 31))
    
    probs = actor(current_state, temporal_state)
    value = critic(current_state, temporal_state)
    
    if probs.shape == (1, 3):
        test_pass(f"Actor output shape: {tuple(probs.shape)}")
    else:
        all_passed = test_fail(f"Actor shape wrong: {probs.shape}")
        
    if value.shape == (1, 1):
        test_pass(f"Critic output shape: {tuple(value.shape)}")
    else:
        all_passed = test_fail(f"Critic shape wrong: {value.shape}")
except Exception as e:
    all_passed = test_fail("Neural networks", str(e))

# ============================================================
# TEST 6: EarnHFT Imports
# ============================================================
test_section("6. EarnHFT Module Imports")
try:
    from strategies.earnhft import BetaAgent, AgentPool, Router, EarnHFTStrategy
    test_pass("All EarnHFT components imported")
except Exception as e:
    all_passed = test_fail("EarnHFT imports", str(e))

# ============================================================
# TEST 7: BetaAgent
# ============================================================
test_section("7. BetaAgent Configuration")
try:
    from strategies.earnhft import BetaAgent
    for beta in [0.1, 0.5, 0.9]:
        agent = BetaAgent(beta=beta)
        if agent.beta != beta:
            all_passed = test_fail(f"BetaAgent beta mismatch: {agent.beta} != {beta}")
        if agent.input_dim != 31:
            all_passed = test_fail(f"BetaAgent input_dim wrong: {agent.input_dim} (expected 31)")
    test_pass(f"BetaAgent: beta param ✓, input_dim=31 ✓")
except Exception as e:
    all_passed = test_fail("BetaAgent", str(e))

# ============================================================
# TEST 8: BetaAgent Beta-Adjusted Reward
# ============================================================
test_section("8. BetaAgent Reward Function")
try:
    from strategies.earnhft import BetaAgent
    
    # Conservative agent (low beta) should penalize losses more
    conservative = BetaAgent(beta=0.1)
    aggressive = BetaAgent(beta=0.9)
    
    # Simulate a loss
    loss_reward_conservative = conservative.compute_beta_reward(-100)
    aggressive.peak_pnl = 0
    aggressive.current_pnl = 0
    loss_reward_aggressive = aggressive.compute_beta_reward(-100)
    
    # Conservative should give more negative reward (more punishment for loss)
    if loss_reward_conservative < loss_reward_aggressive:
        test_pass(f"Conservative punishes loss more: {loss_reward_conservative:.2f} < {loss_reward_aggressive:.2f}")
    else:
        all_passed = test_fail(f"Beta reward logic wrong: {loss_reward_conservative} vs {loss_reward_aggressive}")
except Exception as e:
    all_passed = test_fail("Beta reward", str(e))

# ============================================================
# TEST 9: AgentPool
# ============================================================
test_section("9. AgentPool Regime Classification")
try:
    from strategies.earnhft import AgentPool
    from strategies.earnhft.agent_pool import MarketRegime
    
    pool = AgentPool()
    
    # Test regime classification
    regime_up = pool.classify_regime(0.002, 0.005, 0.01, 0.01)
    regime_down = pool.classify_regime(-0.002, -0.005, 0.01, 0.01)
    regime_volatile = pool.classify_regime(0.001, 0.001, 0.02, 0.01)
    
    if regime_up == MarketRegime.TRENDING_UP:
        test_pass(f"Trending up detection: {regime_up.value}")
    else:
        all_passed = test_fail(f"Trending up wrong: {regime_up}")
        
    if regime_down == MarketRegime.TRENDING_DOWN:
        test_pass(f"Trending down detection: {regime_down.value}")
    else:
        all_passed = test_fail(f"Trending down wrong: {regime_down}")
        
    if regime_volatile == MarketRegime.HIGH_VOLATILITY:
        test_pass(f"High volatility detection: {regime_volatile.value}")
    else:
        all_passed = test_fail(f"High volatility wrong: {regime_volatile}")
except Exception as e:
    all_passed = test_fail("AgentPool", str(e))

# ============================================================
# TEST 10: Router
# ============================================================
test_section("10. Router Network")
try:
    from strategies.earnhft import Router
    import numpy as np
    
    router = Router(num_agents=5, state_dim=31)
    
    # Test agent selection
    state = np.zeros(31, dtype=np.float32)
    agent_idx = router.select_agent(state, training=False)
    
    if 0 <= agent_idx < 5:
        test_pass(f"Router selected agent: {agent_idx} (valid range 0-4)")
    else:
        all_passed = test_fail(f"Router returned invalid index: {agent_idx}")
except Exception as e:
    all_passed = test_fail("Router", str(e))

# ============================================================
# TEST 11: Helpers - Whale Tracker
# ============================================================
test_section("11. Whale Tracker (NEW)")
try:
    from helpers import get_whale_tracker
    
    tracker = get_whale_tracker()
    stats = tracker.get_stats()
    
    if "whale_count" in stats:
        test_pass(f"WhaleTracker functional: {stats}")
    else:
        all_passed = test_fail("WhaleTracker missing stats")
except Exception as e:
    all_passed = test_fail("WhaleTracker", str(e))

# ============================================================
# TEST 12: Helpers - Spoof Detection
# ============================================================
test_section("12. Spoof Detection")
try:
    from helpers import get_spoof_detector, get_spoof_probability
    
    detector = get_spoof_detector()
    prob = get_spoof_probability("BTC")  # Requires market_id
    
    if 0.0 <= prob <= 1.0:
        test_pass(f"Spoof probability: {prob:.2f} (valid range)")
    else:
        all_passed = test_fail(f"Spoof probability out of range: {prob}")
except Exception as e:
    all_passed = test_fail("SpoofDetector", str(e))

# ============================================================
# TEST 13: Helpers - Regime Classifier
# ============================================================
test_section("13. Regime Classifier")
try:
    from helpers import classify_regime, get_regime_adjustments
    
    regime, conf = classify_regime(asset="BTC", price=50000.0, returns_5m=0.02, volatility=0.01)
    adjustments = get_regime_adjustments("BTC")  # Requires asset
    
    test_pass(f"Regime: {regime} (confidence: {conf:.2f})")
    test_pass(f"Adjustments: size_mult={adjustments.get('size_mult', 1.0):.2f}")
except Exception as e:
    all_passed = test_fail("RegimeClassifier", str(e))

# ============================================================
# TEST 14: Pool Config
# ============================================================
test_section("14. EarnHFT Pool Configuration")
try:
    import os
    import json
    
    config_path = "earnhft_agents/pool_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        test_pass(f"Pool config found with {len(config)} regimes")
        for regime, info in config.items():
            test_pass(f"  {regime} -> β={info['beta']}")
    else:
        test_pass("Pool config not yet created (expected if not trained)")
except Exception as e:
    all_passed = test_fail("Pool config", str(e))

# ============================================================
# SUMMARY
# ============================================================
test_section("SUMMARY")
if all_passed:
    print("\n  ✅ ALL TESTS PASSED\n")
    sys.exit(0)
else:
    print("\n  ❌ SOME TESTS FAILED\n")
    sys.exit(1)
