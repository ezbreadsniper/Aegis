# Research: Sub-Tick Microstructure & MEV-Aware Execution

## 1. Sub-Tick Microstructure Arbitrage
- **Ghost Liquidity**: In fragmented markets, "Ghost Liquidity" refers to orders that appear on multiple venues but are canceled as soon as one is filled. On Polymarket, this manifests as bots mirroring Binance order books.
- **Microstructure Signals**: Top traders (e.g., the "Anonymous Trader" who flipped $1K to $2M) use sub-tick signals like **Quote Stuffing** and **Order Layering** to detect when a large participant is about to move the price.
- **Tick-Level Lead-Lag**: While 15-min data shows general trends, sub-second (tick) data reveals that Binance price updates precede Polymarket by 50-150ms. This is the "Golden Window" for latency arbitrage.

## 2. MEV-Aware Execution on Polygon
- **Front-Running (Sandwiching)**: MEV bots on Polygon can detect pending Polymarket transactions in the mempool and "sandwich" them by placing orders before and after to profit from the slippage.
- **MEV Protection**: Using private RPC endpoints (e.g., Flashbots-style bundles or dRPC) can shield transactions from the public mempool, preventing front-running.
- **Priority Bribes**: In highly competitive markets, paying a small "bribe" (priority fee) to block builders ensures your order is processed at the top of the block, which is critical for winning latency races.

## 3. Cross-Venue Liquidity Fragmentation
- **Polymarket vs. Kalshi**: Kalshi often has deeper liquidity for macro events, while Polymarket dominates in crypto-native events. A "Unicorn" strategy involves **Cross-Venue Mean Reversion**—betting that price gaps between the two will close.
- **Binance as the "Source of Truth"**: For crypto binary markets, Binance Futures is the primary liquidity source. Polymarket is essentially a "derivative" of Binance.

## 4. Hidden Markov Model (HMM) Regime Switching
- **Regime States**: HMMs can classify market states into "High Volatility/Trending," "Low Volatility/Ranging," and "Toxic/Manipulated."
- **Adaptive Strategy**: The bot should switch from **Trend Following** (High Vol) to **Market Making** (Low Vol) to **Risk-Off** (Toxic) based on the HMM state.
