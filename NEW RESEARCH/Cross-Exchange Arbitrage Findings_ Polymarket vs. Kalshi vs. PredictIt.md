# Cross-Exchange Arbitrage Findings: Polymarket vs. Kalshi vs. PredictIt

## 1. Market Dynamics & Lead-Lag Relationships
- **Polymarket Leads Kalshi**: Research indicates that Polymarket returns significantly predict Kalshi returns, especially during high-volatility events. This is likely due to Polymarket's higher liquidity and crypto-native user base, which reacts faster to global news [1].
- **PredictIt's "Retail Lag"**: PredictIt often lags both Polymarket and Kalshi due to its $850 position limit and slower, retail-heavy demographic. This creates persistent arbitrage opportunities for bots that can navigate its unique fee structure (10% on profits).

## 2. Fee Structures & Arbitrage Thresholds
| Platform | Trading Fee | Withdrawal Fee | Arbitrage Threshold |
| :--- | :--- | :--- | :--- |
| **Polymarket** | 0% (Maker/Taker) | 0% | ~2.5% (Slippage/Gas) |
| **Kalshi** | ~0.7% | 0% | ~3.5% |
| **PredictIt** | 0% (Buy/Sell) | 10% on Profits | ~12% (High hurdle) |

## 3. Execution Mechanics
- **Automated Market Mapping**: Profitable bots use fuzzy logic and LLMs to map identical markets across platforms (e.g., "BTC > 100k" on Polymarket vs. "Will Bitcoin hit 100k?" on Kalshi).
- **Simultaneous Execution**: To minimize "leg risk" (one side fills, the other doesn't), bots use atomic-like execution or high-speed limit orders on both platforms.
- **Delta-Neutral Setup**: Buying YES on Platform A and NO on Platform B when the combined price is < $1.00 (accounting for fees).

## 4. Key Arbitrage Opportunities
- **Binary Complement Across Platforms**: $\text{Price}_A(\text{YES}) + \text{Price}_B(\text{NO}) < 1$.
- **Information Arbitrage**: Front-running Kalshi's AMM based on Polymarket's CLOB movements.
- **Regulatory Arbitrage**: Exploiting price differences driven by Kalshi's CFTC-regulated status vs. Polymarket's offshore/unregulated status.
