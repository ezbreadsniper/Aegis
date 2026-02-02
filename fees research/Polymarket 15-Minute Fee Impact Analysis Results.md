# Polymarket 15-Minute Fee Impact Analysis Results

## Executive Summary

The analysis reveals a **critical finding**: The current EarnHFT strategy is already operating at negative expected value (-$0.19 per trade) even before fees are applied. With the new Polymarket fee structure, the strategy becomes catastrophically unprofitable, with daily losses projected in the range of $27,000-$48,000 depending on trade frequency.

## 1. Fee Structure Analysis

### Fee Amounts at Different Price Points (100 shares)

| Price | Buy Fee ($) | Sell Fee ($) | Round-Trip ($) | Buy Rate (%) | Sell Rate (%) |
|-------|-------------|--------------|----------------|--------------|---------------|
| $0.10 | $0.06       | $0.56        | $0.62          | 0.56%        | 5.62%         |
| $0.20 | $0.20       | $1.00        | $1.20          | 1.00%        | 5.00%         |
| $0.30 | $0.39       | $1.31        | $1.71          | 1.31%        | 4.38%         |
| $0.40 | $0.60       | $1.50        | $2.10          | 1.50%        | 3.75%         |
| $0.50 | $0.78       | $1.56        | $2.34          | 1.56%        | 3.12%         |
| $0.60 | $0.90       | $1.50        | $2.40          | 1.50%        | 2.50%         |
| $0.70 | $0.92       | $1.31        | $2.23          | 1.31%        | 1.88%         |
| $0.80 | $0.80       | $1.00        | $1.80          | 1.00%        | 1.25%         |
| $0.90 | $0.51       | $0.56        | $1.07          | 0.56%        | 0.62%         |

### Key Fee Characteristics

1. **Asymmetric Fee Structure**: Selling is significantly more expensive than buying at most price points
2. **Peak Fee Zone**: Maximum fees occur at 50-60% probability range
3. **Extreme Price Advantage**: Fees drop dramatically at price extremes (<20% or >80%)
4. **Maker Exemption**: Limit orders that add liquidity pay ZERO fees

## 2. Strategy-Specific Impact

### Current Strategy Metrics
- Average Trade Size: $25.00
- Estimated Shares (at 50%): 50 shares
- Entry Fee (at 50%): $0.39
- Exit Fee (at 50%): $0.78
- **Round-Trip Fee (at 50%): $1.17**

### Expected Value Analysis

| Metric | Value |
|--------|-------|
| Current Win Rate | 23.0% |
| Average Win | $2.50 |
| Average Loss | $1.00 |
| **EV Before Fees** | **-$0.19 per trade** |

**CRITICAL FINDING**: The strategy is already negative EV before fees are applied!

### EV After Fees at Different Price Points

| Price | Round-Trip Fee | EV Before | EV After | EV Change |
|-------|----------------|-----------|----------|-----------|
| $0.30 | $1.42          | -$0.19    | -$1.62   | +729%     |
| $0.40 | $1.31          | -$0.19    | -$1.51   | +673%     |
| $0.50 | $1.17          | -$0.19    | -$1.37   | +601%     |
| $0.60 | $1.00          | -$0.19    | -$1.20   | +513%     |
| $0.70 | $0.80          | -$0.19    | -$0.99   | +409%     |

## 3. Break-Even Analysis

### Required Win Rates to Break Even (With Fees)

| Probability | Round-Trip Fee | Required Win Rate | Current Win Rate | Gap |
|-------------|----------------|-------------------|------------------|-----|
| 30%         | $1.42          | 69.2%             | 23.0%            | 46.2% |
| 50%         | $1.17          | 62.1%             | 23.0%            | 39.1% |
| 70%         | $0.80          | 51.3%             | 23.0%            | 28.3% |

**The current 23% win rate is catastrophically below the required break-even threshold at all probability levels.**

## 4. Daily Session Impact

### Weighted Average Analysis
- Weighted Average Round-Trip Fee: $1.15
- Trades per 10hr Session: 35,000
- Estimated Actual Trades: 30,000

### Daily Financial Impact

| Metric | Value |
|--------|-------|
| Daily Fee Drag | $34,406.25 |
| Daily EV Before Fees | -$5,850.00 |
| Daily EV After Fees | -$40,256.25 |
| Profit Reduction | 100% (already negative) |

## 5. Compounding Effect Over Time

| Time Period | Fee Drag | Cumulative Loss (After Fees) |
|-------------|----------|------------------------------|
| Daily       | $34,406  | $40,256                      |
| Weekly      | $240,844 | $281,794                     |
| Monthly     | $1,032,188 | $1,207,688                 |

## 6. Scenario Analysis

### Worst Case: All trades at 50% probability
- Trades/Day: 35,000
- Round-Trip Fee: $1.17
- Daily Fee Drag: $41,016
- Daily EV After Fees: -$47,841
- **STATUS: ❌ UNPROFITABLE**

### Average Case: Trades distributed across 30-70% range
- Trades/Day: 20,000
- Round-Trip Fee: $1.17
- Daily Fee Drag: $23,438
- Daily EV After Fees: -$27,338
- **STATUS: ❌ UNPROFITABLE**

### Best Case: Trades at extremes (20%, 80%)
- Trades/Day: 10,000
- Round-Trip Fee: $1.42
- Daily Fee Drag: $14,219
- Daily EV After Fees: -$16,169
- **STATUS: ❌ UNPROFITABLE**

### Optimized: 50% maker orders
- Trades/Day: 15,000
- Effective Fee: $0.59
- Daily Fee Drag: $8,789
- Daily EV After Fees: -$11,714
- **STATUS: ❌ UNPROFITABLE**

## 7. Critical Conclusions

1. **Strategy is fundamentally broken**: Even before fees, the strategy has negative EV
2. **Fees amplify losses**: The new fee structure increases daily losses by 6-8x
3. **Win rate is insufficient**: Current 23% win rate requires 51-69% to break even with fees
4. **Trade frequency is counterproductive**: More trades = more losses
5. **Maker orders are essential**: Only way to reduce fee impact is through limit orders
6. **Complete strategy overhaul required**: Incremental changes will not fix this

## 8. Immediate Actions Required

### MANDATORY (Survival)
1. **HALT LIVE TRADING IMMEDIATELY** - Strategy is losing money
2. **Audit paper trading metrics** - Verify if 23% win rate is accurate
3. **Re-examine PnL calculations** - Current metrics suggest fundamental issues

### HIGH PRIORITY (Fee Mitigation)
1. **Implement maker-only order execution** - Eliminates 100% of fees
2. **Add fee modeling to RL reward function** - Retrain with fee-aware rewards
3. **Reduce trade frequency by 90%+** - Only trade high-conviction setups

### MEDIUM PRIORITY (Strategy Optimization)
1. **Increase position sizing** - Larger trades amortize fixed costs better
2. **Focus on price extremes** - Lower fees at <20% or >80% probability
3. **Extend holding periods** - Reduce round-trips per session

