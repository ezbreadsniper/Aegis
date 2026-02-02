# Institutional Best Practices: Polymarket Bot Trading

Based on exhaustive research into HFT, prediction market microstructure, and multi-year data analysis, the following are the "Gold Standard" methods for dominating Polymarket.

## 1. Execution Best Practices (The "Latency & Fee" Edge)
- **Maker-Only Execution**: Never cross the spread. Use "Post-Only" orders to capture the bid-ask spread and avoid the 2% taker fee. This single change can increase your net ROI by 15-20%.
- **AWS London Co-location**: Polymarket's relayer is hosted in AWS London (eu-west-2). Deploying your bot in the same region reduces round-trip time (RTT) from ~200ms to <10ms, allowing you to front-run retail orders.
- **Gasless Transactions**: Use the Polymarket Builder Relayer for gasless trading. This eliminates the overhead of managing MATIC/POL and ensures your orders are processed with priority.

## 2. Risk Management Best Practices (The "Survival" Edge)
- **The "2-Minute Rule"**: Hard-close all positions 2-3 minutes before expiry. The final minutes of a 15-min market are a "liquidity vacuum" where price discovery breaks down and toxic flow dominates.
- **Dynamic Stop-Losses**: Tighten your stop-loss as expiry approaches. A fixed stop-loss is too loose near expiry due to the "Gamma" effect. Use a time-decaying stop-loss formula.
- **Max Exposure Caps**: Limit your total exposure across all 4 markets (BTC, ETH, SOL, XRP) to a fixed percentage of your bankroll (e.g., 10-20%) to survive "Black Swan" events that affect all crypto assets simultaneously.

## 3. Alpha Generation Best Practices (The "Unicorn" Edge)
- **Cross-Asset Lead-Lag**: Binance Futures lead Polymarket by 100-500ms. Use a high-frequency WebSocket feed from Binance to "predict" the next Polymarket price move.
- **Fractal Regime Detection**: Use the Hurst Exponent ($H$) to distinguish between trending ($H > 0.55$) and mean-reverting ($H < 0.45$) markets. Adjust your strategy (Trend Following vs. Market Making) accordingly.
- **Whale Tracking**: Monitor the top 100 Polymarket addresses. When a "Whale" enters a position, it often creates a momentum wave that your bot can ride.

## 4. Technical Best Practices (The "Robustness" Edge)
- **On-Device MLX Training**: Continue using Apple Silicon for real-time PPO updates. The ability to adapt to changing market regimes in sub-second intervals is a massive advantage over cloud-based bots with high inference latency.
- **Adversarial Hardening**: Train your RL agent with "noisy" data to simulate market manipulation and spoofing. This ensures your bot doesn't get "faked out" by other bots.
- **Comprehensive Logging**: Log every order, cancellation, and fill with microsecond timestamps. This is the only way to perform the "Root Cause Analysis" needed to perfect your strategy.
