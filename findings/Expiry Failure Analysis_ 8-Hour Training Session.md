# Expiry Failure Analysis: 8-Hour Training Session

## 1. The "Expiry Trap" Phenomenon
The session data reveals a critical structural flaw: **Forced Expiry Closes**. While the bot achieved a respectable **42% win rate**, the gains were wiped out by 6 catastrophic trades that hit the maximum loss at expiry.

### 1.1. Failure Mechanics
- **Late-Stage Inversion**: The bot holds a position (e.g., DOWN) while the market moves against it in the final minutes.
- **Liquidity Vacuum**: As expiry approaches, the Polymarket CLOB liquidity for that specific 15-min market evaporates.
- **Forced Settlement**: Because the bot lacks an "Early Exit" logic, it is forced to settle at the binary outcome (0 or 1).
- **Asymmetric Loss**: A single forced settlement at 0.99 (for a DOWN position) results in a ~$20 loss, requiring ~10-15 winning trades to recover.

## 2. Quantitative Impact
| Metric | Value | Impact |
| :--- | :--- | :--- |
| **Total PnL** | -$82.23 | Net loss despite high win rate. |
| **Expiry Losses** | ~$104.73 | The sum of the 6 identified expiry failures. |
| **Adjusted PnL** | +$22.50 | PnL if expiry losses were capped at -$5.00. |

## 3. Root Cause: The "Time-Remaining" Blind Spot
The `TemporalEncoder` and `PPOAgent` are not sufficiently weighting the `time_remaining` feature as a **Risk Multiplier**. They treat it as just another feature, rather than a hard constraint that should trigger a "Risk-Off" regime.

## 4. Proposed Fix: The "Bulletproof Expiry & Risk Layer"
1.  **Hard Expiry Cutoff**: Mandatory closure of all positions at `time_remaining < 0.2` (3 minutes).
2.  **Dynamic Stop-Loss**: A volatility-adjusted stop-loss that tightens as expiry approaches.
3.  **No-Trade Zone**: Disable new entries when `time_remaining < 0.33` (5 minutes).
4.  **Gamma-Aware Sizing**: Reduce position size linearly as the "Gamma" (sensitivity to price moves near expiry) increases.
