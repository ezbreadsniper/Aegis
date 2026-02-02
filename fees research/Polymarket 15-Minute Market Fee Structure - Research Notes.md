# Polymarket 15-Minute Market Fee Structure - Research Notes

## Source: Official Polymarket Documentation (docs.polymarket.com)
Date Accessed: 2026-01-06

## Key Findings

### Fee Scope
- Fees apply ONLY to 15-minute crypto markets
- All other Polymarket markets remain fee-free
- Fees are TAKER ONLY - makers do not pay fees

### Fee Mechanism
- Fees are deducted from the PROCEEDS of your trade (what you receive)
- BUY: Pay USDC → Receive Tokens → Fee taken in Tokens
- SELL: Pay Tokens → Receive USDC → Fee taken in USDC

### Fee Amount vs Effective Rate
- Fee amount (in units): Highest at 50% probability, lowest at extremes
- Effective fee rate (as % of trade): Depends on whether buying or selling
- Graph shows feeRate=0.25, exponent=2 formula curve

### Fee Distribution
- All collected taker fees are redistributed daily to market makers as USDC rebates
- Polymarket does NOT retain these fees - they fund the Maker Rebates Program
- Rebates are proportional to share of executed maker liquidity

### Fee Precision
- Fees rounded to 4 decimal places
- Smallest fee charged: 0.0001 USDC
- Very small trades near price extremes may incur no fee at all

## Fee Tables (Need to capture)
- Buying Fees (100 shares) - table showing fees at different price points
- Selling Fees (100 shares) - table showing fees at different price points

## Critical for Strategy
- Fee peaks at 50% probability (up to ~3% reported in news)
- Fee approaches zero at price extremes (near 0% or 100%)
- Maker orders are fee-free and earn rebates
- Only taker orders incur fees


## Buying Fees (100 shares) - Fee deducted in tokens
When you BUY, the fee is deducted in tokens. The effective rate peaks at 50% probability.

| Price | Trade Cost | Fee (tokens) | Fee Value |
|-------|------------|--------------|-----------|
| $0.10 | $10        | 0.20         | $0.02     |
| $0.20 | $20        | 0.64         | $0.13     |
| $0.30 | $30        | 1.10         | $0.33     |
| $0.40 | $40        | 1.44         | $0.58     |
| $0.50 | $50        | 1.56         | $0.78     |
| $0.60 | $60        | 1.44         | $0.86     |
| $0.70 | $70        | 1.10         | $0.77     |
| $0.80 | $80        | 0.64         | $0.51     |
| $0.90 | $90        | 0.20         | $0.18     |

## Selling Fees (100 shares) - Fee deducted in USDC
When you SELL, the fee is deducted in USDC. The effective rate peaks around 30% probability.

| Price | Proceeds | Fee (USDC) | Effective Rate |
|-------|----------|------------|----------------|
| $0.10 | $10      | $0.20      | 2.0%           |
| $0.20 | $20      | $0.64      | 3.2%           |
| $0.30 | $30      | $1.10      | 3.7%           |

(Need to capture more rows...)


## Complete Selling Fees Table (100 shares)

| Price | Proceeds | Fee (USDC) | Effective Rate |
|-------|----------|------------|----------------|
| $0.10 | $10      | $0.20      | 2.0%           |
| $0.20 | $20      | $0.64      | 3.2%           |
| $0.30 | $30      | $1.10      | 3.7%           |
| $0.40 | $40      | $1.44      | 3.6%           |
| $0.50 | $50      | $1.56      | 3.1%           |
| $0.60 | $60      | $1.44      | 2.4%           |
| $0.70 | $70      | $1.10      | 1.6%           |
| $0.80 | $80      | $0.64      | 0.8%           |
| $0.90 | $90      | $0.20      | 0.2%           |

## Key Insight: Why Selling is More Expensive
When buying, the fee is valued at the token price (e.g., 1.56 tokens × $0.50 = $0.78).
When selling, the fee is taken directly in USDC ($1.56). Same fee units, different dollar impact.

## Fee Formula Observed
From the graph: feeRate=0.25, exponent=2
This appears to be a quadratic formula that peaks at 50% probability.

Fee calculation appears to follow: fee = feeRate * 4 * p * (1-p) where p is the probability
At p=0.50: fee = 0.25 * 4 * 0.5 * 0.5 = 0.25 (25% of the maximum)
Maximum fee in tokens for 100 shares at 50%: 1.56 tokens

## Critical Strategy Implications
1. Fees are HIGHEST at 50% probability (~3.1% effective for selling, ~1.56% for buying)
2. Fees approach ZERO at price extremes (0.2% at 90%, 2.0% at 10%)
3. MAKER ORDERS ARE FREE - using limit orders avoids all fees
4. Round-trip fee at 50%: ~$0.78 (buy) + ~$1.56 (sell) = ~$2.34 per 100 shares ($50 trade)
5. This represents 4.68% round-trip fee drag at the worst case


## Technical API Details (Developer Documentation)

### Fee Rate API Endpoint
```
GET https://clob.polymarket.com/fee-rate?token_id={token_id}
```

Response for fee-enabled markets:
```json
{
  "fee_rate_bps": 1000
}
```

- Fee-enabled markets return value like `1000` (10% in basis points = 1%)
- Fee-free markets return `0`
- NOTE: 1000 bps = 10%, but this is the BASE rate, actual fee varies by price

### Fee Rate in Signed Orders
The `feeRateBps` field must be included in the signed order payload:
```json
{
  "salt": "12345",
  "maker": "0x...",
  "signer": "0x...",
  "taker": "0x...",
  "tokenId": "...",
  "makerAmount": "50000000",
  "takerAmount": "100000000",
  "expiration": "0",
  "nonce": "0",
  "feeRateBps": "1000",
  "side": "0",
  "signatureType": 2,
  "signature": "0x..."
}
```

### Important Implementation Notes
1. Always fetch fee_rate_bps dynamically - DO NOT hardcode
2. Fee rate may vary by market or change over time
3. feeRateBps is part of the signed payload - CLOB validates signature against it
4. Official CLOB clients (TypeScript/Python) handle fees automatically

### Fee Behavior Summary
- Fees deducted from PROCEEDS of trade (what you receive)
- BUY: Pay USDC → Receive Tokens → Fee in Tokens
- SELL: Pay Tokens → Receive USDC → Fee in USDC
- Effective rate differs between buying and selling due to denomination

### Maker Rebates
- 100% of collected fees redistributed as maker rebates
- Paid daily in USDC directly to wallet
- Proportional to share of executed maker volume
- Trades at price extremes contribute less to rebate pool


## Complete Effective Rates Tables (Developer Documentation)

### Effective Rates: Buying (100 shares)
When buying, the fee is in tokens. Effective rate peaks at 50%.

| Price | Fee (tokens) | Fee ($) | Effective Rate |
|-------|--------------|---------|----------------|
| $0.10 | 0.20         | $0.02   | 0.2%           |
| $0.30 | 1.10         | $0.33   | 1.1%           |
| $0.50 | 1.56         | $0.78   | 1.6%           |
| $0.70 | 1.10         | $0.77   | 1.1%           |
| $0.90 | 0.20         | $0.18   | 0.2%           |

### Effective Rates: Selling (100 shares)
When selling, the fee is in USDC. Effective rate peaks around 30%.

| Price | Proceeds | Fee ($) | Effective Rate |
|-------|----------|---------|----------------|
| $0.10 | $10      | $0.20   | 2.0%           |
| $0.30 | $30      | $1.10   | 3.7%           |
| $0.50 | $50      | $1.56   | 3.1%           |
| $0.70 | $70      | $1.10   | 1.6%           |
| $0.90 | $90      | $0.20   | 0.2%           |

### Key Observations
1. BUYING effective rate peaks at 1.6% at 50% probability
2. SELLING effective rate peaks at 3.7% at 30% probability
3. Both approach 0.2% at price extremes (10% and 90%)
4. Selling is more expensive than buying at most price points
5. Round-trip at 50%: 1.6% + 3.1% = 4.7% total fee drag

### Maker Rebates Summary
- Eligibility: Orders must add liquidity (maker orders) and get filled
- Calculation: Proportional to share of executed maker volume
- Payment: Daily in USDC, paid directly to wallet
- Rebate Pool: 100% of collected fees redistributed as maker rebates


## Detailed Fee Table from Cointelegraph Article (Jan 6, 2026)

| Price (1 Share) | Fee (1 Share) | Price (100 Shares) | Fee (100 Shares) |
|-----------------|---------------|--------------------|--------------------|
| $0.01           | $0            | $1                 | $0.0025            |
| $0.05           | $0.0006       | $5                 | $0.0564            |
| $0.30           | $0.011        | $30                | $1.1025            |
| $0.40           | $0.0144       | $40                | $1.44              |
| $0.50           | $0.0156       | $50                | $1.5625            |
| $0.60           | $0.0144       | $60                | $1.44              |
| $0.70           | $0.011        | $70                | $1.1025            |
| $0.80           | $0.0064       | $80                | $0.64              |
| $0.90           | $0.002        | $90                | $0.2025            |
| $0.99           | $0            | $99                | $0.0025            |

## Community Analysis and Reactions

### Key Points from Community Discussion:
1. **Wash Trading Protection**: User 0x_opus said the change would "increase protection from wash trading"
2. **Not a Classic Fee**: Polymarket is not "starting to charge users in the classic sense" - fees are redirected to market makers
3. **Anti-HFT Measure**: Trader kiruwaaaaaa described the move as "directed against high-frequency bots"
4. **Liquidity Incentive**: Fee-funded rebates would incentivize tighter spreads and more consistent liquidity
5. **Sustainable Cash Flow**: User Tawer955 said the system creates sustainable cash flow for liquidity providers and reduces incentives for bots that previously exploited free liquidity

### Impact Assessment:
- For most Polymarket users, impact will be LIMITED
- New fees do NOT apply to longer-term event markets, political markets, or non-crypto predictions
- Fees fall sharply near probability extremes
- Very small trades are rounded down (may incur no fee)

## Sources:
- Polymarket Official Documentation: https://docs.polymarket.com/polymarket-learn/trading/fees
- Polymarket Developer Docs: https://docs.polymarket.com/developers/market-makers/maker-rebates-program
- Cointelegraph via TradingView: https://www.tradingview.com/news/cointelegraph:e59c32089094b:0-polymarket-quietly-introduces-taker-fees-on-15-minute-crypto-markets/

