#!/usr/bin/env python3
"""Test script for NEW RESEARCH feature upgrades."""
import sys
sys.path.insert(0, ".")

print("=== Testing NEW RESEARCH Feature Upgrades ===\n")

# Test 1: Feature vector dimension
print("1. Testing Feature Vector Dimension...")
try:
    from strategies import MarketState
    s = MarketState(asset="BTC", prob=0.5, time_remaining=0.5)
    features = s.to_features()
    expected = 28
    actual = len(features)
    if actual == expected:
        print(f"   [PASS] Feature vector length: {actual} (expected {expected})")
    else:
        print(f"   [FAIL] Feature vector length: {actual} (expected {expected})")
        sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Error: {e}")
    sys.exit(1)

# Test 2: New cross-asset fields exist
print("\n2. Testing Cross-Asset Feature Fields...")
try:
    s = MarketState(asset="BTC", prob=0.5, time_remaining=0.5)
    assert hasattr(s, "eth_return_lag1"), "Missing eth_return_lag1"
    assert hasattr(s, "utc_hour"), "Missing utc_hour"
    assert hasattr(s, "is_volatility_peak"), "Missing is_volatility_peak"
    assert hasattr(s, "market_synergy_score"), "Missing market_synergy_score"
    print("   [PASS] All cross-asset fields present")
except AssertionError as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# Test 3: RLStrategy input_dim
print("\n3. Testing RLStrategy Input Dimension...")
try:
    from strategies import RLStrategy
    strategy = RLStrategy()
    expected = 28
    actual = strategy.input_dim
    if actual == expected:
        print(f"   [PASS] RLStrategy input_dim: {actual} (expected {expected})")
    else:
        print(f"   [FAIL] RLStrategy input_dim: {actual} (expected {expected})")
        sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Error: {e}")
    sys.exit(1)

# Test 4: Actor/Critic networks accept 28-dim input
print("\n4. Testing Neural Network Dimensions...")
try:
    import sys
    if sys.platform == "darwin":
        import mlx.core as mx
        from strategies.rl_mlx import Actor, Critic
        
        actor = Actor(input_dim=28)
        critic = Critic(input_dim=28)
        
        current_state = mx.zeros((1, 28))
        temporal_state = mx.zeros((1, 5 * 28))
        
        probs = actor(current_state, temporal_state)
        value = critic(current_state, temporal_state)
        
        mx.eval(probs, value)
        
        probs_shape = tuple(probs.shape)
        value_shape = tuple(value.shape)
    else:
        import torch
        from strategies.rl_pytorch import Actor, Critic
        
        actor = Actor(input_dim=28)
        critic = Critic(input_dim=28)
        
        current_state = torch.zeros((1, 28))
        temporal_state = torch.zeros((1, 5 * 28))
        
        probs = actor(current_state, temporal_state)
        value = critic(current_state, temporal_state)
        
        probs_shape = tuple(probs.shape)
        value_shape = tuple(value.shape)
    
    if probs_shape == (1, 3) and value_shape == (1, 1):
        print(f"   [PASS] Actor output: {probs_shape}, Critic output: {value_shape}")
    else:
        print(f"   [FAIL] Unexpected shapes - Actor: {probs_shape}, Critic: {value_shape}")
        sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All NEW RESEARCH Tests Passed ===")
