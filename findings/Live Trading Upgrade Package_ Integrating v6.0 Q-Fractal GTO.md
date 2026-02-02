# Live Trading Upgrade Package: Integrating v6.0 Q-Fractal GTO

This package provides the necessary code snippets and configuration updates to transform the `cross-market-state-fusion` bot into a live-trading, institutional-grade system based on the v6.0 "Q-Fractal GTO" strategy.

## 1. Dependency Update (`requirements.txt`)

The existing dependencies are insufficient for the advanced mathematical and data fusion requirements. Replace your current `requirements.txt` with the following:

```text
# Core MLX dependencies (as per original)
mlx>=0.5.0
numpy>=1.24.0

# Core Trading/Data dependencies
websockets>=12.0
requests>=2.31.0
flask>=3.0.0
flask-socketio>=5.3.0

# v6.0 Advanced Math & Data Fusion Dependencies
# For Hurst Exponent (FMA)
nolds>=0.5.2 
# For Sentiment Analysis (Alternative Data)
vaderSentiment>=3.3.2 
# For On-Chain/Whale Tracking (Conceptual API Client)
# Note: Actual on-chain data requires a dedicated node/API, but this is a placeholder.
# We assume a new 'onchain_api.py' is created in 'helpers/'.
```

## 2. Feature Expansion and FMA Integration (`strategies/base.py`)

The `MarketState` class must be expanded from 18 to 24 dimensions to include the FMA and Alternative Data features.

**File**: `strategies/base.py`

**Snippet 2.1: New MarketState Features**

Add the following fields to your `MarketState` dataclass:

```python
@dataclass
class MarketState:
    # ... (Existing 18 features) ...
    
    # --- FMA Features (2) ---
    hurst_exponent: float = 0.5  # Hurst Exponent (H)
    fma_regime: str = "RANDOM"   # Regime: TRENDING, MEAN_REVERTING, RANDOM

    # --- Alternative Data Features (4) ---
    whale_flow_imbalance: float = 0.0 # Net flow of top 100 traders
    social_sentiment_score: float = 0.0 # VADER score from social media
    news_alpha_signal: float = 0.0 # LLM-extracted signal from local news
    
    # ... (Existing methods) ...
```

**Snippet 2.2: Dynamic Sizing with FMA**

Update the `get_confidence_size` method to incorporate the FMA regime for dynamic position sizing.

```python
# In strategies/base.py, inside the Action class:

def get_confidence_size(self, prob: float, hurst_exponent: float) -> float:
    """
    Get position size multiplier based on probability extremeness AND FMA regime.
    
    If trending (H > 0.55), we increase max size.
    If mean-reverting (H < 0.45), we decrease max size and prefer market-making.
    """
    if self == Action.HOLD:
        return 0.0
    
    # Base confidence sizing (as per original)
    extremeness = abs(prob - 0.5) * 2 # [0, 1]
    base = 0.25
    scale = 0.75
    confidence_size = base + (scale * extremeness)

    # FMA Regime Adjustment (Hurst Exponent)
    if hurst_exponent > 0.55:
        # Trending Regime: Increase max size by 20%
        fma_multiplier = 1.2 
    elif hurst_exponent < 0.45:
        # Mean-Reverting Regime: Decrease max size by 30%
        fma_multiplier = 0.7 
    else:
        # Random/Neutral Regime
        fma_multiplier = 1.0
        
    return min(1.0, confidence_size * fma_multiplier)
```

## 3. Execution Layer Upgrade (`run.py`)

The `TradingEngine` must be updated to use the QIO logic and the Polymarket CLOB API for real execution.

**File**: `run.py` (Conceptual Update)

**Snippet 3.1: QIO-Inspired Execution Logic**

Replace the paper trading logic within your `TradingEngine`'s `execute_trade` method with the following conceptual framework.

```python
# In run.py, inside the TradingEngine class:

def execute_trade(self, action: Action, market_state: MarketState, size_usd: float) -> bool:
    """
    Executes a trade using the QIO-inspired, Post-Only logic.
    
    This replaces the paper trading assumption with real CLOB API calls.
    """
    if action == Action.HOLD:
        return True

    # 1. Determine Target Price (GTO Policy Output)
    # The RL agent's output probability (prob_up) is converted to a target price.
    target_price = market_state.prob 
    
    # 2. QIO Order Routing (Conceptual)
    # In a real implementation, this would call a dedicated QUBO solver service.
    # For now, we simulate the optimal price and pool selection.
    
    # The optimal price is slightly better than the current best bid/ask to ensure a fill
    # while still acting as a maker (Post-Only).
    if action == Action.BUY:
        # Target price is slightly below the current best ask (to act as a maker)
        order_price = target_price - 0.0001 
        side = "BUY_UP"
    else: # action == Action.SELL
        # Target price is slightly above the current best bid (to act as a maker)
        order_price = target_price + 0.0001
        side = "SELL_UP"

    # 3. Polymarket CLOB API Call (Post-Only)
    try:
        # We assume a 'polymarket_clob_api' is available in helpers/
        # The 'post_only=True' flag is critical to avoid taker fees.
        order_result = self.polymarket_clob_api.place_order(
            market_id=market_state.market_id,
            side=side,
            price=order_price,
            size_usd=size_usd,
            post_only=True # CRITICAL: Ensures maker status
        )
        
        if order_result.status == "FILLED" or order_result.status == "PENDING":
            print(f"Executed {side} order at {order_price}. Status: {order_result.status}")
            return True
        else:
            print(f"Order failed: {order_result.reason}")
            return False
            
    except Exception as e:
        print(f"Execution error: {e}")
        return False
```

## 4. GTO Policy and Adversarial Training (`strategies/rl_mlx.py`)

The PPO agent needs to be hardened against adversarial attacks.

**File**: `strategies/rl_mlx.py` (Conceptual Update)

**Snippet 4.1: Adversarial Training Loop**

In the PPO training loop, specifically in the `update_policy` method, you must introduce an adversarial step.

```python
# In strategies/rl_mlx.py, inside the PPOAgent class:

def update_policy(self, batch: List[Experience]):
    # ... (Existing PPO setup) ...

    for epoch in range(self.n_epochs):
        # --- NEW: Adversarial Training Step ---
        # 1. Generate Adversarial Examples (Perturbations)
        # This is a conceptual step. In a real system, this would use a
        # PGD (Projected Gradient Descent) attack to find the perturbation delta.
        # For now, we simulate a small random noise injection to increase robustness.
        
        # This forces the model to learn to ignore small, artificial price movements.
        adversarial_noise = mx.random.uniform(
            low=-self.adversarial_epsilon, 
            high=self.adversarial_epsilon, 
            shape=batch_states.shape
        )
        
        # Perturb the state before calculating loss
        perturbed_states = batch_states + adversarial_noise
        
        # 2. Calculate Loss on Perturbed States (Min-Max Problem)
        # The rest of the PPO loss calculation proceeds as normal, but uses
        # the perturbed_states for the Actor and Critic forward passes.
        
        # ... (Rest of the PPO loss calculation using perturbed_states) ...
        
        # 3. Apply Gradients
        # ... (Existing gradient application) ...
        
    # ... (End of update_policy) ...
```

**Configuration Update**: Add `adversarial_epsilon` to your PPO hyperparameters in `strategies/rl_mlx.py` (e.g., `adversarial_epsilon = 0.005`).
