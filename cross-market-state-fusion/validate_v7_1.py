
import sys
import os
import numpy as np
import torch

try:
    print("1. Checking Imports...")
    from helpers.vpin_detector import VPINDetector
    from helpers.state_normalizer import StateNormalizer
    from helpers.pca_fusion import PCAStateFusion
    from helpers.spoofing_detection import SpoofingDetector  # Check this specific import path
    from helpers.edge_confidence import EdgeConfidenceGate # Check this one too
    print("   - All helper modules imported successfully.")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

try:
    print("\n2. Checking MarketState Feature Vector...")
    from strategies.base import MarketState
    # Create a dummy state
    state = MarketState(
        asset="BTC", prob=0.5, time_remaining=0.5,
        vpin=0.5,  # Check if vpin field exists
        spoof_probability=0.1
    )
    features = state.to_features()
    print(f"   - Feature vector shape: {features.shape}")
    if features.shape[0] != 29:
        print(f"❌ FEATURE DIMENSION ERROR: Expected 29, got {features.shape[0]}")
        sys.exit(1)
    print("   - Feature vector dimension correct (29).")
except Exception as e:
    print(f"❌ MARKET STATE ERROR: {e}")
    sys.exit(1)

try:
    print("\n3. Checking RL Strategy Integration...")
    from strategies.rl_pytorch import RLStrategy, MarketState, Action
    strategy = RLStrategy(input_dim=29) # Initialize with new dim
    
    # Mock act() call
    action = strategy.act(state)
    print("   - strategy.act() executed successfully.")
    
    # Check if normalizer exists in strategy
    if not hasattr(strategy, 'normalizer'):
         print("❌ RL STRATEGY ERROR: 'normalizer' attribute missing.")
         sys.exit(1)
    print("   - strategy.normalizer verified.")

except Exception as e:
    print(f"❌ RL STRATEGY ERROR: {e}")
    sys.exit(1)

print("\n✅ V7.1 SYSTEM CHECK PASSED! READY FOR TRAINING.")
