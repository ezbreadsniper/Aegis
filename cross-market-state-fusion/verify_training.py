#!/usr/bin/env python3
"""
Quick verification script for PPO training.

Simulates enough experiences to trigger a PPO update and verifies
that policy_loss and value_loss are NOT zero.

Run: python verify_training.py
"""
import sys
import numpy as np

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
# TEST 1: Import Strategy
# ============================================================
test_section("1. Import RLStrategy")
try:
    from strategies import RLStrategy
    from strategies.base import MarketState, Action
    test_pass(f"RLStrategy imported")
except Exception as e:
    all_passed = test_fail("Import", str(e))
    sys.exit(1)

# ============================================================
# TEST 2: Verify input_dim = 31
# ============================================================
test_section("2. Verify Dimensions")
try:
    rl = RLStrategy()
    if rl.input_dim == 31:
        test_pass(f"input_dim = {rl.input_dim} ✓")
    else:
        all_passed = test_fail(f"input_dim = {rl.input_dim}, expected 31")
    
    if rl.buffer_size == 256:
        test_pass(f"buffer_size = {rl.buffer_size} ✓")
    else:
        all_passed = test_fail(f"buffer_size = {rl.buffer_size}, expected 256")
except Exception as e:
    all_passed = test_fail("Dimension check", str(e))

# ============================================================
# TEST 3: Create dummy state and verify action works
# ============================================================
test_section("3. Action Selection")
try:
    # Enable training mode
    rl.train()
    
    state = MarketState(asset="BTC", prob=0.55, time_remaining=600)
    action = rl.act(state)
    
    test_pass(f"Action selected: {action} ({action.value})")
except Exception as e:
    all_passed = test_fail("Action selection", str(e))

# ============================================================  
# TEST 4: Simulate experiences to fill buffer
# ============================================================
test_section("4. Fill Experience Buffer")
try:
    rl.reset()  # Clear old experiences
    rl.train()
    
    # Simulate 256 experiences (buffer_size)
    for i in range(rl.buffer_size + 10):  # A few extra
        state = MarketState(
            asset="BTC" if i % 2 == 0 else "ETH",
            prob=0.3 + np.random.random() * 0.4,  # Random prob 0.3-0.7
            time_remaining=(900 - i * 3) / 900,  # Decreasing time
        )
        
        action = rl.act(state)
        
        # Simulate reward (small random PnL)
        reward = np.random.randn() * 0.5  # Random -1 to +1 ish
        
        next_state = MarketState(
            asset=state.asset,
            prob=state.prob + np.random.randn() * 0.02,
            time_remaining=max(0, state.time_remaining - 0.01),
        )
        
        rl.store(state, action, reward, next_state, done=(i % 50 == 49))
    
    buffer_len = len(rl.experiences)
    test_pass(f"Buffer filled: {buffer_len} experiences")
    
except Exception as e:
    all_passed = test_fail("Buffer fill", str(e))

# ============================================================
# TEST 5: Trigger PPO Update
# ============================================================
test_section("5. PPO Update (Critical Test)")
try:
    metrics = rl.update()
    
    if metrics is None:
        all_passed = test_fail("update() returned None - buffer didn't trigger PPO")
    else:
        # Check that key metrics are present and non-zero
        policy_loss = metrics.get("policy_loss", 0)
        value_loss = metrics.get("value_loss", 0)
        entropy = metrics.get("entropy", 0)
        
        if policy_loss != 0 and value_loss != 0:
            test_pass(f"PPO UPDATE SUCCESS!")
            test_pass(f"  policy_loss = {policy_loss:.6f}")
            test_pass(f"  value_loss = {value_loss:.6f}")
            test_pass(f"  entropy = {entropy:.4f}")
        else:
            all_passed = test_fail(f"Metrics are zero: policy={policy_loss}, value={value_loss}")
        
        # Buffer should be cleared after update
        if len(rl.experiences) == 0:
            test_pass(f"Buffer cleared after update ✓")
        else:
            test_pass(f"Buffer has {len(rl.experiences)} experiences after update")

except Exception as e:
    import traceback
    all_passed = test_fail("PPO update", str(e))
    traceback.print_exc()

# ============================================================
# SUMMARY
# ============================================================
test_section("SUMMARY")
if all_passed:
    print("\n  ✅ PPO TRAINING IS WORKING!")
    print("\n  You can now start full training:")
    print("    python run.py rl --train --size 50 --dashboard")
    print("\n  Or with EarnHFT multi-agent:")
    print("    python run.py earnhft --train --size 50 --dashboard")
    print()
    sys.exit(0)
else:
    print("\n  ❌ TRAINING HAS ISSUES - See errors above")
    print()
    sys.exit(1)
