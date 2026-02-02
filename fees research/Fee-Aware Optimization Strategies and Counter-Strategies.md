# Fee-Aware Optimization Strategies and Counter-Strategies

## Executive Summary

This document presents state-of-the-art fee optimization techniques adapted for Polymarket's new 15-minute market fee structure. The strategies are derived from high-frequency trading research, market microstructure theory, and prediction market liquidity provisioning frameworks.

---

## Part 1: Structural Changes to Trade Execution

### 1.1 Maker-Only Order Execution (CRITICAL)

**Rationale**: Maker orders pay ZERO fees and earn rebates. This is the single most impactful change.

**Implementation**:
```python
# Current (Taker-Heavy) Approach
def execute_trade(signal, price, size):
    return market_order(side=signal, size=size)  # Always taker

# Fee-Optimized Approach
def execute_trade_fee_aware(signal, price, size, urgency):
    if urgency < 0.7:  # Low urgency - use maker
        return limit_order(
            side=signal,
            price=price - 0.01 if signal == 'BUY' else price + 0.01,
            size=size,
            time_in_force='GTC'
        )
    elif urgency < 0.9:  # Medium urgency - aggressive limit
        return limit_order(
            side=signal,
            price=price,  # At market
            size=size,
            time_in_force='IOC'
        )
    else:  # High urgency only - use taker
        return market_order(side=signal, size=size)
```

**Expected Impact**:
- 80% maker ratio → 80% fee reduction
- Monthly fee savings: ~$800,000 (based on current volume)
- Trade-off: Slower execution, potential adverse selection

### 1.2 Probability-Aware Trade Sizing

**Rationale**: Fees are highest at 50% probability, lowest at extremes. Adjust position sizing accordingly.

**Fee-Adjusted Position Sizing Formula**:
```
adjusted_size = base_size × fee_multiplier(probability)

where fee_multiplier(p) = 1 / (1 + k × 4 × p × (1-p))
k = fee_sensitivity_parameter (recommended: 0.5-2.0)
```

**Implementation**:
```python
def calculate_fee_adjusted_size(base_size, probability, k=1.0):
    """
    Reduce position size when fees are high (near 50%)
    Increase position size when fees are low (near extremes)
    """
    fee_factor = 4 * probability * (1 - probability)  # Peaks at 1.0 at 50%
    fee_multiplier = 1 / (1 + k * fee_factor)
    return base_size * fee_multiplier

# Examples:
# At 50%: fee_multiplier = 0.50 (halve position)
# At 20%: fee_multiplier = 0.61 (reduce by 39%)
# At 10%: fee_multiplier = 0.74 (reduce by 26%)
# At 5%:  fee_multiplier = 0.84 (reduce by 16%)
```

### 1.3 Holding Period Extension

**Rationale**: Round-trip fees are fixed per trade. Longer holds amortize fees over larger expected moves.

**Current Strategy**: 15-minute markets with frequent entry/exit
**Optimized Strategy**: Selective entry, hold through multiple 15-minute intervals

**Implementation**:
```python
class HoldingPeriodOptimizer:
    def __init__(self, min_expected_move=0.05, fee_threshold=0.02):
        self.min_expected_move = min_expected_move
        self.fee_threshold = fee_threshold
    
    def should_exit(self, position, current_price, expected_move, fees_paid):
        """
        Only exit if expected remaining move justifies exit fees
        """
        exit_fee = calculate_fee(current_price, position.shares, is_buy=False)
        remaining_edge = expected_move - fees_paid - exit_fee
        
        return remaining_edge > self.min_expected_move
```

---

## Part 2: Fee-Minimization Techniques from HFT Research

### 2.1 Order Splitting and TWAP Execution

**Technique**: Split large orders into smaller chunks executed over time as maker orders.

**Benefits**:
- Increases maker fill probability
- Reduces market impact
- Averages entry/exit prices

**Implementation**:
```python
def twap_execution(total_size, duration_minutes, price_limit):
    """
    Time-Weighted Average Price execution
    """
    num_slices = duration_minutes * 4  # 4 slices per minute
    slice_size = total_size / num_slices
    
    for i in range(num_slices):
        place_limit_order(
            size=slice_size,
            price=price_limit,
            time_in_force='IOC'
        )
        sleep(15)  # 15 seconds between slices
```

### 2.2 Passive Liquidity Provision

**Technique**: Instead of taking liquidity, provide it and earn rebates.

**Strategy Transformation**:
```
Current: Signal → Market Order → Pay Fee
Optimized: Signal → Limit Order → Earn Rebate (if filled)
```

**Rebate Calculation**:
- Daily rebate pool = Total taker fees collected
- Your share = Your maker volume / Total maker volume
- Expected rebate: 0.5-2% of maker volume (varies by market)

### 2.3 Fee-Aware Signal Filtering

**Technique**: Only act on signals where expected edge exceeds fee threshold.

**Implementation**:
```python
def fee_aware_signal_filter(signal_strength, probability, trade_size):
    """
    Filter signals that don't clear the fee hurdle
    """
    round_trip_fee = calculate_round_trip_fee(probability, trade_size)
    fee_as_pct = round_trip_fee / trade_size
    
    # Require signal strength to exceed fee + buffer
    min_signal_strength = fee_as_pct + 0.02  # 2% buffer
    
    return signal_strength > min_signal_strength
```

---

## Part 3: Advanced Tactics

### 3.1 Volatility Gating

**Concept**: Trade more aggressively when volatility is high (larger moves justify fees), reduce activity when volatility is low.

**Implementation**:
```python
class VolatilityGate:
    def __init__(self, vol_threshold_low=0.02, vol_threshold_high=0.08):
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
    
    def get_trade_multiplier(self, current_volatility):
        if current_volatility < self.vol_threshold_low:
            return 0.0  # No trading in low vol
        elif current_volatility > self.vol_threshold_high:
            return 2.0  # Aggressive in high vol
        else:
            # Linear interpolation
            return (current_volatility - self.vol_threshold_low) / \
                   (self.vol_threshold_high - self.vol_threshold_low)
```

### 3.2 Regime Detection

**Concept**: Identify market regimes where fee-adjusted edge is positive.

**Regime Types**:
1. **Trending**: High directional moves → Fees easily covered
2. **Mean-Reverting**: Predictable oscillations → Selective trading
3. **Choppy**: Random noise → Avoid trading (fees dominate)

**Implementation**:
```python
class RegimeDetector:
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def detect_regime(self, price_history):
        returns = np.diff(price_history) / price_history[:-1]
        
        # Hurst exponent estimation
        hurst = self.estimate_hurst(returns)
        
        if hurst > 0.6:
            return 'TRENDING'
        elif hurst < 0.4:
            return 'MEAN_REVERTING'
        else:
            return 'CHOPPY'
    
    def get_fee_adjustment(self, regime):
        adjustments = {
            'TRENDING': 1.5,      # Trade more
            'MEAN_REVERTING': 1.0, # Normal
            'CHOPPY': 0.2         # Trade less
        }
        return adjustments.get(regime, 1.0)
```

### 3.3 Selective Participation

**Concept**: Only participate in markets/times where fee-adjusted edge is highest.

**Criteria for Participation**:
1. Probability near extremes (<25% or >75%) → Lower fees
2. High volatility expected → Larger moves
3. Strong signal conviction → Higher win rate
4. Maker order likely to fill → Zero fees

**Implementation**:
```python
def should_participate(market_state, signal):
    score = 0
    
    # Probability score (prefer extremes)
    prob = market_state.probability
    prob_score = 1 - 4 * prob * (1 - prob)  # 0 at 50%, 1 at extremes
    score += prob_score * 30
    
    # Volatility score
    if market_state.volatility > 0.05:
        score += 25
    
    # Signal strength score
    score += signal.strength * 30
    
    # Maker fill probability score
    if market_state.spread > 0.02:
        score += 15  # Wide spread = easier maker fills
    
    return score > 60  # Threshold for participation
```

### 3.4 Delayed Execution

**Concept**: Wait for optimal execution conditions rather than immediate execution.

**Optimal Conditions**:
- Price moves toward your limit order
- Spread narrows
- Probability moves toward extremes
- Volatility increases

**Implementation**:
```python
class DelayedExecutor:
    def __init__(self, max_delay_seconds=60):
        self.max_delay = max_delay_seconds
        self.pending_orders = []
    
    def queue_order(self, order, conditions):
        self.pending_orders.append({
            'order': order,
            'conditions': conditions,
            'queued_at': time.time()
        })
    
    def check_conditions(self, market_state):
        for pending in self.pending_orders:
            if self.conditions_met(pending['conditions'], market_state):
                self.execute(pending['order'])
                self.pending_orders.remove(pending)
            elif time.time() - pending['queued_at'] > self.max_delay:
                # Timeout - execute anyway or cancel
                self.cancel_or_execute(pending)
```

---

## Part 4: Fee Exploitation Strategies

### 4.1 Maker Rebate Farming

**Concept**: Provide liquidity specifically to earn rebates, not for directional exposure.

**Strategy**:
1. Place symmetric limit orders on both sides
2. Earn rebates when either side fills
3. Hedge directional exposure immediately

**Risk**: Adverse selection (informed traders pick off your orders)

### 4.2 Fee Arbitrage Between Order Types

**Concept**: Exploit the asymmetry between maker (free) and taker (costly) orders.

**Strategy**:
1. Identify situations where maker orders have high fill probability
2. Use maker orders exclusively in these situations
3. Accept taker fees only when urgency is critical

### 4.3 Cross-Market Fee Optimization

**Concept**: If trading multiple Polymarket markets, concentrate volume in lowest-fee markets.

**Implementation**:
- Rank markets by effective fee rate
- Allocate more capital to lower-fee markets
- Use higher-fee markets only for diversification

---

## Part 5: RL Model Modifications

### 5.1 Fee-Aware Reward Function

**Current Reward**:
```python
reward = pnl_change
```

**Fee-Aware Reward**:
```python
def calculate_reward(action, state, next_state):
    pnl_change = next_state.portfolio_value - state.portfolio_value
    
    # Calculate fees for this action
    if action.is_trade:
        fee = calculate_fee(state.price, action.size, action.is_buy)
        if action.is_taker:
            fee_cost = fee
        else:
            fee_cost = -fee * rebate_rate  # Negative cost = rebate
    else:
        fee_cost = 0
    
    # Penalize high-frequency trading
    frequency_penalty = 0.01 * state.trades_last_hour
    
    return pnl_change - fee_cost - frequency_penalty
```

### 5.2 Action Space Modification

**Current Action Space**:
```python
actions = ['BUY', 'SELL', 'HOLD']
```

**Fee-Aware Action Space**:
```python
actions = [
    'BUY_MAKER',      # Limit order to buy
    'BUY_TAKER',      # Market order to buy
    'SELL_MAKER',     # Limit order to sell
    'SELL_TAKER',     # Market order to sell
    'HOLD',           # No action
    'CANCEL_ORDERS'   # Cancel pending orders
]
```

### 5.3 State Space Enhancement

**Additional State Features**:
```python
state_features = [
    # Existing features...
    
    # Fee-related features
    'current_fee_rate',           # Fee at current probability
    'fee_rate_gradient',          # How fee changes with price
    'maker_fill_probability',     # Likelihood of maker fill
    'time_to_market_close',       # Urgency factor
    'pending_maker_orders',       # Outstanding limit orders
    'cumulative_fees_today',      # Fee budget tracking
    'rebate_earned_today',        # Rebate tracking
]
```

---

## Part 6: Implementation Priority Matrix

| Strategy | Impact | Complexity | Risk | Priority |
|----------|--------|------------|------|----------|
| Maker-Only Orders | HIGH | LOW | LOW | 1 (CRITICAL) |
| Fee-Aware Reward Function | HIGH | MEDIUM | LOW | 2 |
| Probability-Aware Sizing | MEDIUM | LOW | LOW | 3 |
| Volatility Gating | MEDIUM | MEDIUM | LOW | 4 |
| Regime Detection | MEDIUM | HIGH | MEDIUM | 5 |
| Holding Period Extension | MEDIUM | LOW | MEDIUM | 6 |
| Selective Participation | MEDIUM | MEDIUM | LOW | 7 |
| Order Splitting (TWAP) | LOW | MEDIUM | LOW | 8 |
| Maker Rebate Farming | LOW | HIGH | HIGH | 9 |

---

## Part 7: Code Changes Required in Current Strategy

### 7.1 trade_executor.py Modifications

```python
# Add fee calculation to TradeExecutor class
class FeeAwareTradeExecutor(TradeExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fee_calculator = PolymarketFeeCalculator()
        self.maker_preference = 0.8  # 80% maker target
    
    def execute(self, signal, price, size):
        # Calculate expected fee
        expected_fee = self.fee_calculator.round_trip(price, size)
        
        # Check if signal clears fee hurdle
        if signal.expected_pnl < expected_fee * 1.5:
            return None  # Skip trade
        
        # Determine order type
        if random.random() < self.maker_preference:
            return self.execute_maker(signal, price, size)
        else:
            return self.execute_taker(signal, price, size)
```

### 7.2 reward_calculator.py Modifications

```python
# Add fee-aware reward calculation
class FeeAwareRewardCalculator:
    def __init__(self, fee_weight=1.0, frequency_penalty=0.01):
        self.fee_weight = fee_weight
        self.frequency_penalty = frequency_penalty
        self.fee_calculator = PolymarketFeeCalculator()
    
    def calculate(self, state, action, next_state):
        # Base PnL
        pnl = next_state.portfolio_value - state.portfolio_value
        
        # Fee cost
        if action.is_trade:
            fee = self.fee_calculator.calculate(
                state.price, action.size, action.is_buy, action.is_maker
            )
            fee_cost = fee * self.fee_weight
        else:
            fee_cost = 0
        
        # Frequency penalty
        freq_penalty = self.frequency_penalty * state.recent_trade_count
        
        return pnl - fee_cost - freq_penalty
```

### 7.3 config.py Additions

```python
# Fee configuration
FEE_CONFIG = {
    'base_rate': 1.5625,
    'maker_fee': 0.0,
    'taker_fee_formula': 'quadratic',
    'rebate_rate': 0.5,  # Estimated rebate as % of fees paid
    
    # Optimization parameters
    'maker_preference': 0.8,
    'min_edge_over_fees': 1.5,
    'max_trades_per_hour': 100,
    'fee_budget_daily': 1000,
}
```

