#!/usr/bin/env python3
"""
State Normalizer - Z-Score Normalization with Running Statistics

Normalizes all continuous state features to zero mean and unit variance
for improved RL training stability and convergence. Uses exponential
moving average (EMA) for online learning without storing full history.

Why This Matters:
- Raw features have vastly different scales (BTC price ~$90k vs prob ~0.5)
- Neural networks learn poorly with unscaled inputs
- Small-magnitude features get ignored without normalization
- Z-score brings everything to comparable [-3, 3] range

Usage:
    normalizer = StateNormalizer(feature_dim=29)
    
    # During training
    normalized_state = normalizer.normalize(raw_state_vector)
    normalizer.update_stats(raw_state_vector)  # Update running stats
"""
import numpy as np
from pathlib import Path
import json


class StateNormalizer:
    """
    Online Z-score normalization with exponential moving average.
    
    Maintains running mean and std for each feature dimension.
    Uses EMA for memory-efficient online learning.
    """
    
    def __init__(
        self,
        feature_dim: int = 31,  # v7.5: 31 features (removed position_pnl)
        alpha: float = 0.001,    # EMA decay rate (smaller = slower adaptation)
        epsilon: float = 1e-8,   # Prevent division by zero
        clip_range: float = 10.0,  # Clip normalized values to [-clip, +clip]
    ):
        """
        Initialize state normalizer.
        
        Args:
            feature_dim: Number of features in state vector
            alpha: EMA decay rate for updating statistics
            epsilon: Small constant to prevent division by zero
            clip_range: Clip normalized values to this range
        """
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # Running statistics (EMA)
        self.mean = np.zeros(feature_dim, dtype=np.float32)
        self.var = np.ones(feature_dim, dtype=np.float32)  # Variance, not std
        self.std = np.ones(feature_dim, dtype=np.float32)
        
        # Track number of updates
        self.n_updates = 0
        
    def normalize(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Normalize state vector to zero mean, unit variance.
        
        Args:
            state_vector: Raw state features (unnormalized)
            
        Returns:
            Normalized state vector, clipped to [-clip_range, +clip_range]
        """
        # Z-score normalization: (x - mean) / std
        normalized = (state_vector - self.mean) / (self.std + self.epsilon)
        
        # Clip to prevent extreme values
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        return normalized.astype(np.float32)
    
    def update_stats(self, state_vector: np.ndarray):
        """
        Update running statistics with new observation using EMA.
        
        Args:
            state_vector: New state vector to incorporate into statistics
        """
        self.n_updates += 1
        
        # Use larger alpha for first few updates (faster initial adaptation)
        effective_alpha = min(0.1, self.alpha * (100 / max(1, self.n_updates)))
        
        # Update mean: EMA(mean)
        self.mean = (1 - effective_alpha) * self.mean + effective_alpha * state_vector
        
        # Update variance: EMA(var)
        squared_diff = (state_vector - self.mean) ** 2
        self.var = (1 - effective_alpha) * self.var + effective_alpha * squared_diff
        
        # Update std (square root of variance)
        self.std = np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, normalized_vector: np.ndarray) -> np.ndarray:
        """
        Convert normalized state back to original scale.
        
        Useful for interpreting or visualizing normalized states.
        
        Args:
            normalized_vector: Normalized state
            
        Returns:
            Denormalized state in original scale
        """
        return normalized_vector * self.std + self.mean
    
    def get_stats(self) -> dict:
        """Get current normalization statistics."""
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'var': self.var.tolist(),
            'n_updates': self.n_updates,
            'alpha': self.alpha,
        }
    
    def save(self, filepath: str):
        """Save normalization statistics to file."""
        stats = self.get_stats()
        stats['feature_dim'] = self.feature_dim
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[StateNormalizer] Saved to {filepath}")
    
    def load(self, filepath: str):
        """Load normalization statistics from file."""
        with open(filepath, 'r') as f:
            stats = json.load(f)
        
        self.feature_dim = stats['feature_dim']
        self.mean = np.array(stats['mean'], dtype=np.float32)
        self.std = np.array(stats['std'], dtype=np.float32)
        self.var = np.array(stats['var'], dtype=np.float32)
        self.n_updates = stats['n_updates']
        self.alpha = stats.get('alpha', 0.001)
        
        print(f"[StateNormalizer] Loaded from {filepath} ({self.n_updates} updates)")
    
    def reset(self):
        """Reset statistics to initial state."""
        self.mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.var = np.ones(self.feature_dim, dtype=np.float32)
        self.std = np.ones(self.feature_dim, dtype=np.float32)
        self.n_updates = 0


# Singleton instance
_state_normalizer: StateNormalizer = None


def get_state_normalizer(feature_dim: int = 31, **kwargs) -> StateNormalizer:
    """
    Get or create the global state normalizer.
    
    Args:
        feature_dim: Number of features (default 31 for v7.3)
        **kwargs: Additional StateNormalizer parameters
        
    Returns:
        StateNormalizer instance
    """
    global _state_normalizer
    if _state_normalizer is None:
        _state_normalizer = StateNormalizer(feature_dim=feature_dim, **kwargs)
    return _state_normalizer


if __name__ == "__main__":
    print("Testing StateNormalizer...")
    
    # Test with 29-dimensional state
    normalizer = StateNormalizer(feature_dim=29, alpha=0.01)
    
    print("\n1. Initial state (mean=0, std=1):")
    print(f"   Mean: {normalizer.mean[:5]}...")
    print(f"   Std:  {normalizer.std[:5]}...")
    
    # Simulate some states with different scales
    print("\n2. Simulating states with different scales...")
    for i in range(100):
        # Feature 0: Large scale (price-like, 0-100k)
        # Feature 1: Small scale (probability, 0-1)
        # Feature 2: Medium scale (returns, -0.05 to 0.05)
        state = np.random.randn(29).astype(np.float32)
        state[0] *= 50000  # BTC price scale
        state[1] = np.random.rand() * 2 - 1  # Probability range
        state[2] *= 0.02  # Small returns
        
        # Update statistics
        normalizer.update_stats(state)
        
        # Normalize
        normalized = normalizer.normalize(state)
        
        if i % 20 == 0:
            print(f"   Update {i}: Feature 0 mean={normalizer.mean[0]:.1f}, std={normalizer.std[0]:.1f}")
    
    print("\n3. Final statistics:")
    print(f"   Feature 0 (large scale): mean={normalizer.mean[0]:.1f}, std={normalizer.std[0]:.1f}")
    print(f"   Feature 1 (small scale): mean={normalizer.mean[1]:.3f}, std={normalizer.std[1]:.3f}")
    print(f"   Feature 2 (medium scale): mean={normalizer.mean[2]:.4f}, std={normalizer.std[2]:.4f}")
    
    # Test normalization
    test_state = np.random.randn(29).astype(np.float32)
    test_state[0] = 90000.0  # BTC price
    test_state[1] = 0.75  # Probability
    
    normalized = normalizer.normalize(test_state)
    print(f"\n4. Test normalization:")
    print(f"   Raw state[0]: {test_state[0]:.0f} → Normalized: {normalized[0]:.2f}")
    print(f"   Raw state[1]: {test_state[1]:.2f} → Normalized: {normalized[1]:.2f}")
    
    # Test save/load
    normalizer.save("test_normalizer.json")
    
    new_normalizer = StateNormalizer()
    new_normalizer.load("test_normalizer.json")
    print(f"\n5. Loaded normalizer has {new_normalizer.n_updates} updates")
    
    # Clean up
    Path("test_normalizer.json").unlink()
    
    print("\n✓ StateNormalizer test complete")
