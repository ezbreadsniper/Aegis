# Advanced Market Making Findings: Prediction Market CLOBs

## 1. The Market Maker's Edge
- **Scalability**: Market making is the most scalable strategy on Polymarket. Instead of directional betting, a bot simultaneously posts limit orders to buy YES and sell YES (or buy NO), capturing the bid-ask spread [1].
- **Fee Advantage**: By acting as a maker, the bot avoids the 2% taker fee and can potentially earn liquidity rewards from Polymarket's "Builder" program.

## 2. Inventory Management Strategies
- **Skewed Quotes**: To manage inventory, the bot should adjust its quotes based on its current position. If it is "Long YES", it should lower its bid for YES and lower its ask for YES to encourage selling and discourage further buying.
- **Inventory-Based Price Prediction**: Research shows that the aggregate inventory of liquidity providers can forecast future price movements. A bot can use its own inventory levels as a feature in its RL model to predict short-term price reversals.
- **Delta-Neutral Rebalancing**: Market makers often hedge their net inventory using correlated assets (e.g., Binance Futures) to remain delta-neutral and focus purely on spread capture.

## 3. Advanced Liquidity Provision Techniques
- **Bayesian Market Making**: Using a Bayesian approach to update the "fair price" based on incoming trade flow. If more trades are hitting the ask than the bid, the fair price is likely higher than the current mid-price.
- **Adverse Selection Protection**: Bots must detect "informed" traders (whales or news-driven traders) and widen their spreads or pull quotes during periods of high information flow to avoid being "picked off".
- **Cross-Market Liquidity Provision**: Providing liquidity on both Polymarket and Kalshi simultaneously, using the price on the more liquid platform (usually Polymarket) to set quotes on the less liquid one.

## 4. Key Metrics for Market Makers
| Metric | Definition | Goal |
| :--- | :--- | :--- |
| **Inventory Risk** | The risk of holding a large, unhedged position. | Minimize via skewed quotes and hedging. |
| **Adverse Selection** | The risk of trading with someone who has better information. | Minimize via spread widening and flow analysis. |
| **Fill Rate** | The percentage of limit orders that are executed. | Maximize while maintaining profitability. |
| **Rebate Capture** | The amount earned from liquidity rewards. | Maximize via consistent top-of-book presence. |
