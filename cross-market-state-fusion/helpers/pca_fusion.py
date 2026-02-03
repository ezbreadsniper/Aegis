#!/usr/bin/env python3
"""
PCA State Fusion - Principal Component Analysis for Feature Reduction

Reduces redundant price features (BTC, ETH, SOL, XRP) to 2 principal
components that capture >95% of the variance. This simplifies the state
space for RL while preserving essential information.

Why This Matters:
- BTC/ETH/SOL/XRP prices are highly correlated (>0.8)
- Redundant features confuse neural networks
- PC1 = "Market Sentiment" (overall crypto direction)
- PC2 = "Altcoin Beta" (alt-specific movements)
- Reduces 4 correlated features to 2 independent ones

Usage:
    pca = PCAStateFusion()
    pca.fit(price_history)  # Fit on historical data
    
    # Transform current prices
    pc1, pc2 = pca.transform([btc_price, eth_price, sol_price, xrp_price])
"""
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple


class PCAStateFusion:
    """
    PCA-based feature reduction for correlated price features.
    
    Transforms [BTC, ETH, SOL, XRP] prices into 2 principal components:
    - PC1: Market Sentiment (overall crypto trend)
    - PC2: Altcoin Beta (alt-specific variance)
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize PCA state fusion.
        
        Args:
            n_components: Number of principal components (default 2)
        """
        self.n_components = n_components
        
        # PCA components (learned from data)
        self.components = None  # Shape: (n_components, n_features)
        self.mean = None  # Feature means for centering
        self.std = None  # Feature stds for scaling
        
        # Variance explained by each component
        self.explained_variance_ratio = None
        
        self.is_fitted = False
    
    def fit(self, price_history: np.ndarray):
        """
        Fit PCA on historical price data.
        
        Args:
            price_history: Array of shape (n_samples, n_features)
                          e.g., (1000, 4) for 1000 samples of [BTC, ETH, SOL, XRP]
        """
        if price_history.shape[0] < 10:
            raise ValueError(f"Need at least 10 samples to fit PCA, got {price_history.shape[0]}")
        
        # Center and scale data
        self.mean = np.mean(price_history, axis=0)
        self.std = np.std(price_history, axis=0) + 1e-8  # Prevent division by zero
        
        scaled_data = (price_history - self.mean) / self.std
        
        # Compute covariance matrix
        cov_matrix = np.cov(scaled_data.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components].T
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        
        self.is_fitted = True
        
        print(f"[PCA] Fitted on {price_history.shape[0]} samples")
        print(f"[PCA] Explained variance: {self.explained_variance_ratio * 100}")
        print(f"[PCA] Total variance captured: {np.sum(self.explained_variance_ratio) * 100:.1f}%")
    
    def transform(self, prices: List[float]) -> np.ndarray:
        """
        Transform prices to principal components.
        
        Args:
            prices: List of [BTC, ETH, SOL, XRP] prices
            
        Returns:
            Principal components [PC1, PC2]
        """
        if not self.is_fitted:
            # Return zeros if not fitted yet (cold start)
            return np.zeros(self.n_components, dtype=np.float32)
        
        prices_array = np.array(prices, dtype=np.float32)
        
        # Center and scale
        scaled_prices = (prices_array - self.mean) / self.std
        
        # Project onto principal components
        pcs = np.dot(self.components, scaled_prices)
        
        return pcs.astype(np.float32)
    
    def fit_transform(self, price_history: np.ndarray) -> np.ndarray:
        """Fit PCA and transform data in one step."""
        self.fit(price_history)
        
        # Transform all samples
        scaled_data = (price_history - self.mean) / self.std
        transformed = np.dot(scaled_data, self.components.T)
        
        return transformed
    
    def inverse_transform(self, pcs: np.ndarray) -> np.ndarray:
        """
        Transform principal components back to original prices.
        
        Useful for interpretation and validation.
        
        Args:
            pcs: Principal components [PC1, PC2]
            
        Returns:
            Reconstructed prices [BTC, ETH, SOL, XRP]
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet")
        
        # Inverse projection
        scaled_prices = np.dot(pcs, self.components)
        
        # Unscale
        prices = scaled_prices * self.std + self.mean
        
        return prices
    
    def get_stats(self) -> dict:
        """Get PCA statistics."""
        if not self.is_fitted:
            return {'is_fitted': False}
        
        return {
            'is_fitted': True,
            'n_components': self.n_components,
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'components': self.components.tolist(),
            'explained_variance_ratio': self.explained_variance_ratio.tolist(),
        }
    
    def save(self, filepath: str):
        """Save PCA parameters to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted PCA")
        
        stats = self.get_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[PCA] Saved to {filepath}")
    
    def load(self, filepath: str):
        """Load PCA parameters from file."""
        with open(filepath, 'r') as f:
            stats = json.load(f)
        
        self.n_components = stats['n_components']
        self.mean = np.array(stats['mean'], dtype=np.float32)
        self.std = np.array(stats['std'], dtype=np.float32)
        self.components = np.array(stats['components'], dtype=np.float32)
        self.explained_variance_ratio = np.array(stats['explained_variance_ratio'], dtype=np.float32)
        self.is_fitted = True
        
        print(f"[PCA] Loaded from {filepath}")
        print(f"[PCA] Variance captured: {np.sum(self.explained_variance_ratio) * 100:.1f}%")


# Singleton instance
_pca_fusion: PCAStateFusion = None


def get_pca_fusion(n_components: int = 2) -> PCAStateFusion:
    """
    Get or create the global PCA fusion instance.
    
    Args:
        n_components: Number of principal components
        
    Returns:
        PCAStateFusion instance
    """
    global _pca_fusion
    if _pca_fusion is None:
        _pca_fusion = PCAStateFusion(n_components=n_components)
    return _pca_fusion


if __name__ == "__main__":
    print("Testing PCAStateFusion...")
    
    # Simulate correlated price data (BTC, ETH, SOL, XRP)
    np.random.seed(42)
    n_samples = 500
    
    # BTC as base (high correlation)
    btc = 90000 + np.cumsum(np.random.randn(n_samples) * 1000)
    
    # ETH correlated with BTC (0.8 correlation)
    eth = 3200 + 0.035 * (btc - 90000) + np.random.randn(n_samples) * 50
    
    # SOL also correlated (0.75 correlation)
    sol = 135 + 0.0015 * (btc - 90000) + np.random.randn(n_samples) * 5
    
    # XRP less correlated (0.6 correlation)
    xrp = 2.1 + 0.00002 * (btc - 90000) + np.random.randn(n_samples) * 0.1
    
    price_history = np.column_stack([btc, eth, sol, xrp])
    
    print(f"\n1. Generated {n_samples} synthetic price samples")
    print(f"   BTC: ${btc[0]:.0f} to ${btc[-1]:.0f}")
    print(f"   ETH: ${eth[0]:.0f} to ${eth[-1]:.0f}")
    print(f"   SOL: ${sol[0]:.1f} to ${sol[-1]:.1f}")
    print(f"   XRP: ${xrp[0]:.2f} to ${xrp[-1]:.2f}")
    
    # Fit PCA
    pca = PCAStateFusion(n_components=2)
    transformed = pca.fit_transform(price_history)
    
    print(f"\n2. PCA Results:")
    print(f"   PC1 (Market Sentiment): explains {pca.explained_variance_ratio[0]*100:.1f}% variance")
    print(f"   PC2 (Altcoin Beta): explains {pca.explained_variance_ratio[1]*100:.1f}% variance")
    print(f"   Total: {np.sum(pca.explained_variance_ratio)*100:.1f}% variance captured")
    
    # Test transform
    test_prices = [95000.0, 3300.0, 140.0, 2.2]  # Current prices
    pcs = pca.transform(test_prices)
    
    print(f"\n3. Transform test prices:")
    print(f"   Input: BTC=${test_prices[0]}, ETH=${test_prices[1]}, SOL=${test_prices[2]}, XRP=${test_prices[3]}")
    print(f"   Output: PC1={pcs[0]:.2f}, PC2={pcs[1]:.2f}")
    
    # Test inverse transform
    reconstructed = pca.inverse_transform(pcs)
    print(f"\n4. Inverse transform (reconstruction):")
    print(f"   Original:      BTC=${test_prices[0]:.0f}, ETH=${test_prices[1]:.0f}, SOL=${test_prices[2]:.1f}, XRP=${test_prices[3]:.2f}")
    print(f"   Reconstructed: BTC=${reconstructed[0]:.0f}, ETH=${reconstructed[1]:.0f}, SOL=${reconstructed[2]:.1f}, XRP=${reconstructed[3]:.2f}")
    
    # Test save/load
    pca.save("test_pca.json")
    
    new_pca = PCAStateFusion()
    new_pca.load("test_pca.json")
    
    # Clean up
    Path("test_pca.json").unlink()
    
    print("\n✓ PCAStateFusion test complete")
