# Monte Carlo Fee Drag Simulation Results

## 30-Day Projection Summary

| Scenario | Trades/Day | Maker % | Win Rate | Monthly PnL (Before) | Monthly PnL (After) | Monthly Fees | Status |
|----------|------------|---------|----------|---------------------|---------------------|--------------|--------|
| Current Strategy (No Changes) | 30,000 | 0% | 23% | -$176,876 | -$1,194,690 | $1,017,815 | ❌ Unprofitable |
| Reduced Frequency (10x) | 3,000 | 0% | 23% | -$17,200 | -$118,838 | $101,638 | ❌ Unprofitable |
| Maker-Only Orders | 15,000 | 100% | 23% | -$88,807 | -$88,807 | $0 | ❌ Unprofitable |
| 50% Maker Orders | 20,000 | 50% | 23% | -$117,980 | -$456,487 | $338,507 | ❌ Unprofitable |
| Trade at Extremes Only | 10,000 | 0% | 23% | -$59,904 | -$368,904 | $309,001 | ❌ Unprofitable |
| Improved Win Rate (35%) | 15,000 | 50% | 35% | $100,382 | -$153,596 | $253,978 | ❌ Unprofitable |
| **Improved Win Rate (50%)** | 10,000 | 50% | 50% | $225,102 | **$55,723** | $169,378 | ✅ **Profitable** |
| **Optimal Configuration** | 5,000 | 80% | 40% | $59,016 | **$28,321** | $30,696 | ✅ **Profitable** |

## Key Findings

### 1. Current Strategy is Catastrophically Unprofitable
- **Monthly Loss (After Fees): $1,194,690**
- **Monthly Fee Drag: $1,017,815**
- The strategy loses money even before fees (-$176,876/month)
- Fees amplify losses by 6.75x

### 2. Fee Mitigation Alone is Insufficient
- Even with 100% maker orders (zero fees), the strategy loses $88,807/month
- The fundamental issue is the 23% win rate, not just fees
- Reducing trade frequency helps but doesn't solve the core problem

### 3. Minimum Requirements for Profitability
To achieve profitability, the strategy needs:

| Configuration | Win Rate | Maker % | Trades/Day | Monthly Profit |
|---------------|----------|---------|------------|----------------|
| Minimum Viable | 50% | 50% | 10,000 | $55,723 |
| Optimal | 40% | 80% | 5,000 | $28,321 |

### 4. Break-Even Analysis

**At 23% Win Rate (Current):**
- Cannot break even at any trade frequency or maker ratio
- Strategy is fundamentally negative EV

**At 35% Win Rate:**
- Still unprofitable even with 50% maker orders
- Fees consume all potential profits

**At 40% Win Rate:**
- Profitable with 80% maker orders and 5,000 trades/day
- Tight margins, high execution risk

**At 50% Win Rate:**
- Profitable with 50% maker orders and 10,000 trades/day
- More robust profit margins

## Critical Conclusions

1. **The 23% win rate is the primary problem** - not the fees
2. **Fees expose the fundamental weakness** - they don't create it
3. **Win rate must increase to at least 40%** for any chance of profitability
4. **Maker orders are essential** - at least 50-80% of volume
5. **Trade frequency must decrease** - from 30,000 to 5,000-10,000/day
6. **Complete strategy overhaul required** - incremental changes won't work

## Recommended Path Forward

### Phase 1: Immediate (Stop the Bleeding)
1. Halt live trading immediately
2. Switch to paper trading for validation
3. Audit all PnL calculations

### Phase 2: Strategy Redesign
1. Retrain RL model with fee-aware reward function
2. Target minimum 40% win rate
3. Implement maker-only order execution
4. Reduce trade frequency by 80-90%

### Phase 3: Validation
1. Paper trade for minimum 30 days
2. Verify win rate exceeds 40%
3. Confirm maker order fill rate exceeds 80%
4. Calculate actual fee drag

### Phase 4: Gradual Redeployment
1. Start with 10% of target capital
2. Scale up only if metrics hold
3. Continuous monitoring of fee impact

