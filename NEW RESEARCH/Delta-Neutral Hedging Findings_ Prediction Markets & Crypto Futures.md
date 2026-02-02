# Delta-Neutral Hedging Findings: Prediction Markets & Crypto Futures

## 1. The "Basis" in Prediction Markets
- **Definition**: The difference between the probability implied by a prediction market (e.g., Polymarket) and the price of a correlated derivative (e.g., Binance Futures or Options).
- **Basis Trade**: If Polymarket assigns a 60% chance to "BTC > 100k" ($0.60) but the 100k Call Option on Binance is trading at an implied probability of 55%, a trader can buy the option and sell the Polymarket YES shares to capture the 5% spread.

## 2. Delta-Neutral Hedging Strategies
- **Hedging YES Shares with Futures**: If a bot buys YES shares for "BTC > 100k", it is effectively "Long Delta". To hedge this, it can short an equivalent amount of BTC Perpetual Futures on Binance. This cancels out the price risk of BTC, leaving only the "event risk" (the probability of hitting 100k).
- **Dynamic Rebalancing**: As the price of BTC moves, the delta of the YES shares changes. The bot must continuously rebalance its futures position to maintain delta neutrality.
- **Funding Rate Arbitrage**: By holding a delta-neutral position (Long YES on Poly, Short Futures on Binance), the bot can also collect the funding rate on the short futures position if it is positive.

## 3. Hedging with Options Chains
- **Synthetic YES/NO**: An option chain can be used to create a synthetic prediction market position. For example, a **Bull Spread** (Long Call at K1, Short Call at K2) mimics the payoff of a "Price > K" prediction market.
- **Volatility Arbitrage**: If Polymarket is pricing in high volatility (wide spread between YES and NO), but the options market is pricing in low volatility (low IV), a bot can sell the Polymarket "straddle" (YES + NO) and buy the options straddle.

## 4. Key Advantages for the Bot
- **Risk Reduction**: Delta-neutral hedging allows the bot to take larger positions on Polymarket without being exposed to the underlying asset's price volatility.
- **Capital Efficiency**: Using futures for hedging requires less capital (due to leverage) than holding the spot asset.
- **New Alpha**: Exploiting the mispricing between the "Event Probability" (Polymarket) and the "Price Probability" (Options/Futures).
