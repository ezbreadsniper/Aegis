#!/usr/bin/env python3
"""
Unified runner for Polymarket trading strategies.

Usage:
    python run.py                     # List available strategies
    python run.py random              # Run random baseline
    python run.py mean_revert         # Run mean reversion
    python run.py rl                  # Run RL strategy
    python run.py gating              # Run gating (MoE)
    python run.py rl --train          # Train RL strategy
    python run.py rl --train --dashboard  # Train with web dashboard
    python run.py rl --live           # LIVE trading with real money
"""
import asyncio
import argparse
import copy
import random
import sys
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch

# v7.5 FIX: Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sys.path.insert(0, ".")
from helpers import get_15m_markets, BinanceStreamer, OrderbookStreamer, Market, FuturesStreamer, get_logger
from helpers import get_executor, is_live_trading_enabled, PolymarketExecutor
from helpers.spoofing_detection import SpoofingDetector
from helpers.edge_confidence import EdgeConfidenceGate, update_trade_result
# v7.1: VPIN Toxic Flow Detection (Upgraded to v7.2: OFI-based)
from helpers.vpin_detector import get_vpin_detector, clear_detector as clear_vpin_detector
from helpers.vpin_detector import get_vpin_detector_v2, clear_detector_v2 as clear_vpin_detector_v2
from helpers import get_regime_classifier, classify_regime, get_regime_adjustments, get_liq_signal
# v7.3: ARMS - Adaptive Risk Management System (ATR trailing stops, Hurst regime filter)
from helpers.arms import get_arms, update_arms_price, get_trailing_stop_info, get_kelly_position
from strategies import (
    Strategy, MarketState, Action,
    create_strategy, AVAILABLE_STRATEGIES,
    RLStrategy, EarnHFTStrategy, LacunaStrategy,
)


# Dashboard integration (optional)
try:
    from dashboard_cinematic import update_dashboard_state, update_rl_metrics, emit_rl_buffer, run_dashboard, emit_trade
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    def update_dashboard_state(**kwargs): pass
    def update_rl_metrics(metrics): pass
    def emit_rl_buffer(buffer_size, max_buffer=256, avg_reward=None): pass
    def emit_trade(action, asset, size=0, pnl=None): pass


@dataclass
class Position:
    """Track position for a market."""
    asset: str
    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_prob: float = 0.0
    time_remaining_at_entry: float = 0.0
    # v7.3: ARMS trailing stop tracking
    highest_price: float = 0.0  # Highest price since entry (for long stops)
    lowest_price: float = 1.0   # Lowest price since entry (for short stops)


class TradingEngine:
    """
    Trading engine with strategy harness.
    Supports both paper trading and live trading modes.
    """

    def __init__(self, strategy: Strategy, trade_size: float = 50.0, live_mode: bool = False):
        self.strategy = strategy
        self.trade_size = trade_size
        self.live_mode = live_mode

        # Initialize executor for live trading
        self.executor: Optional[PolymarketExecutor] = None
        if self.live_mode:
            try:
                self.executor = get_executor()
                print("\n" + "=" * 60)
                print("⚠️  LIVE TRADING MODE ENABLED ⚠️")
                print("=" * 60)
                print(f"Wallet: {self.executor.wallet_address}")
                print(f"Max Position Size: ${self.executor.safety.max_position_size:.2f}")
                print(f"Max Exposure: ${self.executor.safety.max_exposure:.2f}")
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"\n❌ Failed to initialize live trading: {e}")
                print("Falling back to paper trading mode.\n")
                self.live_mode = False

        # Streamers
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])

        # Detectors
        self.spoof_detector = SpoofingDetector()
        # v7.1: VPIN detectors are per-market (indexed by condition_id)
        self.vpin_detectors: Dict[str, any] = {} # Stores VPINDetector instances
        self.edge_gate = EdgeConfidenceGate()
        self.regime_classifier = get_regime_classifier()

        # State
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.states: Dict[str, MarketState] = {}
        self.prev_states: Dict[str, MarketState] = {}  # For RL transitions
        self.open_prices: Dict[str, float] = {}  # Binance price at market open
        self.token_ids: Dict[str, Dict[str, str]] = {}  # cid -> {"UP": token_id, "DOWN": token_id}
        self.running = False

        # Stats
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.filtered_count = 0  # Trades blocked by EdgeConfidenceGate

        # Pending rewards for RL (set on position close)
        self.pending_rewards: Dict[str, float] = {}

        self.session_start_pnl = 0.0

        # Logger (for RL/EarnHFT/Lacuna training)
        if isinstance(strategy, (RLStrategy, EarnHFTStrategy, LacunaStrategy)):
            self.logger = get_logger(strategy_prefix=strategy.name)
        else:
            self.logger = None
        
        # Edge confidence gate, spoof detector, and regime classifier
        # self.edge_gate = get_edge_gate() # Replaced by EdgeConfidenceGate()
        # self.spoof_detector = get_spoof_detector() # Replaced by SpoofingDetector()
        # self.regime_classifier = get_regime_classifier() # Kept

    def refresh_markets(self):
        """Find active 15-min markets."""
        print("\n" + "=" * 60)
        print(f"STRATEGY: {self.strategy.name.upper()}")
        print("=" * 60)

        markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        now = datetime.now(timezone.utc)

        # Clear stale data
        self.markets.clear()
        self.states.clear()

        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            if mins_left < 0.5:
                continue

            print(f"\n{m.asset} 15m | {mins_left:.1f}m left")
            print(f"  UP: {m.price_up:.3f} | DOWN: {m.price_down:.3f}")

            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)

            # Init state
            self.states[m.condition_id] = MarketState(
                asset=m.asset,
                prob=m.price_up,
                time_remaining=mins_left / 15.0,
            )

            # Init position
            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

            # Record open price
            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0:
                self.open_prices[m.condition_id] = current_price

            # Store token IDs for live trading
            self.token_ids[m.condition_id] = {
                "UP": m.token_up,
                "DOWN": m.token_down,
            }
            
            # Initialize VPIN detector for this market
            if m.condition_id not in self.vpin_detectors:
                self.vpin_detectors[m.condition_id] = get_vpin_detector(m.condition_id)


        if not self.markets:
            print("\nNo active markets!")
        else:
            # Clear stale orderbook subscriptions
            active_cids = set(self.markets.keys())
            self.orderbook_streamer.clear_stale(active_cids)
            
            # v7.5 FIX: Initialize circuit breaker session start to current PnL
            # This ensures daily_loss = session_start_pnl - total_pnl is accurate
            self.session_start_pnl = self.total_pnl

    def execute_action(self, cid: str, action: Action, state: MarketState):
        """Execute trade (paper or live) with flexible sizing and liquidity-aware scaling."""
        if action == Action.HOLD:
            return

        pos = self.positions.get(cid)
        if not pos:
            return



        price = state.prob
        base_trade_amount = self.trade_size * action.size_multiplier
        
        # === NEW RESEARCH: Liquidity-Aware Sizing ===
        # Scale down position based on spoof probability to avoid fake liquidity
        spoof_prob = getattr(state, 'spoof_probability', 0.0)
        liquidity_multiplier = max(0.25, 1.0 - spoof_prob)  # Never go below 25%
        trade_amount = base_trade_amount * liquidity_multiplier
        
        # === EDGE-BASED SIZING (2026 Loss Prevention) ===
        # Scale position size by edge strength - weaker edge = smaller position
        edge_strength = abs(state.prob - 0.50)
        if edge_strength < 0.05:      # 45-55% prob
            edge_mult = 0.5           # Half size - very uncertain
        elif edge_strength < 0.10:    # 40-60% prob
            edge_mult = 0.75          # Three-quarter size
        elif edge_strength < 0.15:    # 35-65% prob
            edge_mult = 0.85          # Slight reduction
        else:
            edge_mult = 1.0           # Full size - strong edge
        trade_amount = trade_amount * edge_mult
        
        if spoof_prob > 0.3:  # Log significant reductions
            print(f"    ⚠️ Spoof detected ({spoof_prob:.1%}): size ${base_trade_amount:.0f}→${trade_amount:.0f}")

        # Get token IDs for live trading
        tokens = self.token_ids.get(cid, {})

        # v8.1: Removed 15s mandatory hold - now using wait_for_fill() for confirmations
        # Minimal safety buffer to prevent race conditions (0.5s instead of 15s)
        MIN_HOLD_SECONDS = 0.5
        if pos.size > 0 and pos.entry_time and self.live_mode:
            time_held = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
            if time_held < MIN_HOLD_SECONDS:
                return

        # Close existing position if switching sides
        if pos.size > 0:
            if action.is_sell and pos.side == "UP":
                # Close UP position - calculate shares we actually own
                shares_owned = pos.size / pos.entry_price
                if self.live_mode and self.executor and tokens.get("UP"):
                    result = self.executor.place_order(
                        token_id=tokens["UP"],
                        side="SELL",
                        size=pos.size,
                        price=price,
                        shares_override=shares_owned,  # Pass actual shares
                    )
                    if not result.success:
                        print(f"    ⚠️ LIVE ORDER FAILED: {result.error}")
                        return

                shares = pos.size / pos.entry_price
                pnl = (price - pos.entry_price) * shares
                self._record_trade(pos, price, pnl, "CLOSE UP", cid=cid)
                self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                pos.size = 0
                pos.side = None
                return

            elif action.is_buy and pos.side == "DOWN":
                # Close DOWN position - calculate shares we actually own
                exit_down_price = 1 - price  # Current DOWN token price
                shares_owned = pos.size / pos.entry_price
                if self.live_mode and self.executor and tokens.get("DOWN"):
                    result = self.executor.place_order(
                        token_id=tokens["DOWN"],
                        side="SELL",
                        size=pos.size,
                        price=exit_down_price,
                        shares_override=shares_owned,  # Pass actual shares
                    )
                    if not result.success:
                        print(f"    ⚠️ LIVE ORDER FAILED: {result.error}")
                        return

                shares = pos.size / pos.entry_price
                pnl = (exit_down_price - pos.entry_price) * shares  # DOWN token went up = profit
                self._record_trade(pos, price, pnl, "CLOSE DOWN", cid=cid)
                self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                pos.size = 0
                pos.side = None
                return

        # Open new position
        if pos.size == 0:
            # v8.1: NO BUY ZONE - Don't open new positions in last 2 minutes
            NO_BUY_ZONE_MINS = 2.0
            time_left_mins = state.time_remaining * 15  # Convert to minutes
            if time_left_mins < NO_BUY_ZONE_MINS:
                return  # Too close to expiry, skip new positions
            
            size_label = {0.25: "SM", 0.5: "MD", 1.0: "LG"}.get(action.size_multiplier, "")

            if action.is_buy:
                # Buy UP token
                if self.live_mode and self.executor and tokens.get("UP"):
                    result = self.executor.place_order(
                        token_id=tokens["UP"],
                        side="BUY",
                        size=trade_amount,
                        price=price,
                    )
                    if not result.success:
                        print(f"    ⚠️ LIVE ORDER FAILED: {result.error}")
                        return
                    # v8.4: VERIFIED EXECUTION - wait for fill confirmation
                    if result.order_id:
                        fill_result = self.executor.wait_for_fill(result.order_id, timeout=15.0)
                        if fill_result.status == "MATCHED" or fill_result.status == "PAPER_FILLED":
                            print(f"    ✅ ORDER CONFIRMED: {fill_result.status}")
                        else:
                            print(f"    ⚠️ ORDER NOT CONFIRMED: {fill_result.status} - still updating position")
                            # Continue anyway - order may fill later

                pos.side = "UP"
                pos.size = trade_amount
                pos.entry_price = price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price
                pos.time_remaining_at_entry = state.time_remaining
                pos.highest_price = price
                pos.lowest_price = price
                mode_label = "🔴 LIVE" if self.live_mode else "PAPER"
                print(f"    [{mode_label}] OPEN {pos.asset} UP ({size_label}) ${trade_amount:.0f} @ {price:.3f}")
                emit_trade(f"BUY_{size_label}", pos.asset, pos.size)

            elif action.is_sell:
                # Buy DOWN token
                down_price = 1 - price
                if self.live_mode and self.executor and tokens.get("DOWN"):
                    result = self.executor.place_order(
                        token_id=tokens["DOWN"],
                        side="BUY",
                        size=trade_amount,
                        price=down_price,
                    )
                    if not result.success:
                        print(f"    ⚠️ LIVE ORDER FAILED: {result.error}")
                        return
                    # v8.4: VERIFIED EXECUTION - wait for fill confirmation
                    if result.order_id:
                        fill_result = self.executor.wait_for_fill(result.order_id, timeout=15.0)
                        if fill_result.status == "MATCHED" or fill_result.status == "PAPER_FILLED":
                            print(f"    ✅ ORDER CONFIRMED: {fill_result.status}")
                        else:
                            print(f"    ⚠️ ORDER NOT CONFIRMED: {fill_result.status} - still updating position")

                pos.side = "DOWN"
                pos.size = trade_amount
                pos.entry_price = down_price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price
                pos.time_remaining_at_entry = state.time_remaining
                pos.highest_price = down_price
                pos.lowest_price = down_price
                mode_label = "🔴 LIVE" if self.live_mode else "PAPER"
                print(f"    [{mode_label}] OPEN {pos.asset} DOWN ({size_label}) ${trade_amount:.0f} @ {down_price:.3f}")
                emit_trade(f"SELL_{size_label}", pos.asset, pos.size)

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        """Record completed trade."""
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        print(f"    {action} {pos.asset} @ {price:.3f} | PnL: ${pnl:+.2f}")
        # Emit to dashboard
        emit_trade(action, pos.asset, pos.size, pnl)
        
        # Update edge gate with trade result (for streak tracking)
        update_trade_result(pnl, self.total_pnl)
        
        # Log to CSV

        # Log to CSV
        if self.logger and pos.entry_time:
            duration = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
            binance_change = 0.0
            if cid and cid in self.open_prices:
                current = self.price_streamer.get_price(pos.asset)
                if current > 0 and self.open_prices[cid] > 0:
                    binance_change = (current - self.open_prices[cid]) / self.open_prices[cid]

            self.logger.log_trade(
                asset=pos.asset,
                action="BUY" if "UP" in action else "SELL",
                side=pos.side or "UNKNOWN",
                entry_price=pos.entry_price,
                exit_price=price,
                size=pos.size,
                pnl=pnl,
                duration_sec=duration,
                time_remaining=pos.time_remaining_at_entry,
                prob_at_entry=pos.entry_prob,
                prob_at_exit=price,
                binance_change=binance_change,
                condition_id=cid
            )

    def _compute_step_reward(self, cid: str, state: MarketState, action: Action, pos: Position) -> float:
        """Compute reward signal for RL training - PURE PnL only (v7.5).
        
        v7.5 FIX: Removed reward shaping penalties (fee, VPIN, spoof).
        TRAINING_JOURNAL.md documents how reward shaping caused Phase 1 failure.
        Pure sparse PnL rewards are more honest and prevent gaming the reward function.
        """
        # Pure reward: realized PnL from closed trades only
        return self.pending_rewards.pop(cid, 0.0)

    def close_all_positions(self):
        """Close all positions at current prices."""
        for cid, pos in self.positions.items():
            if pos.size > 0:
                state = self.states.get(cid)
                if state:
                    price = state.prob
                    shares = pos.size / pos.entry_price
                    if pos.side == "UP":
                        pnl = (price - pos.entry_price) * shares
                    else:
                        exit_down_price = 1 - price
                        pnl = (exit_down_price - pos.entry_price) * shares

                    self._record_trade(pos, price, pnl, f"FORCE CLOSE {pos.side}", cid=cid)
                    self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                    pos.size = 0
                    pos.side = None

    async def decision_loop(self):
        """Main trading loop."""
        tick = 0
        tick_interval = 0.5  # 500ms ticks for faster decisions
        while self.running:
            await asyncio.sleep(tick_interval)
            tick += 1
            now = datetime.now(timezone.utc)

            # Check expired markets
            expired = [cid for cid, m in self.markets.items() if m.end_time <= now]
            for cid in expired:
                print(f"\n  EXPIRED: {self.markets[cid].asset}")
                
                # Clean up VPIN detector for this market
                clear_vpin_detector(cid)

                # RL: Store terminal experience with final PnL
                if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                    state = self.states.get(cid)
                    prev_state = self.prev_states.get(cid)
                    pos = self.positions.get(cid)
                    if state and prev_state:
                        # Terminal reward is the realized PnL
                        terminal_reward = state.position_pnl if pos and pos.size > 0 else 0.0
                        self.strategy.store(prev_state, Action.HOLD, terminal_reward, state, done=True)

                    # Clean up prev_state
                    if cid in self.prev_states:
                        del self.prev_states[cid]

                del self.markets[cid]

            if not self.markets:
                print("\nAll markets expired. Refreshing...")
                self.close_all_positions()
                self.refresh_markets()
                if not self.markets:
                    print("No new markets. Waiting...")
                    await asyncio.sleep(30)
                continue

            # Update states and make decisions
            for cid, m in self.markets.items():
                state = self.states.get(cid)
                if not state:
                    continue

                # Update state from orderbook - CRITICAL for 15-min
                ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                if ob and ob.mid_price:
                    state.prob = ob.mid_price
                    state.prob_history.append(ob.mid_price)
                    if len(state.prob_history) > 100:
                        state.prob_history = state.prob_history[-100:]
                    state.best_bid = ob.best_bid or 0.0
                    state.best_ask = ob.best_ask or 0.0
                    state.spread = ob.spread or 0.0

                    # Orderbook imbalance - L1 (top of book)
                    if ob.bids and ob.asks:
                        bid_vol_l1 = ob.bids[0][1] if ob.bids else 0
                        ask_vol_l1 = ob.asks[0][1] if ob.asks else 0
                        total_l1 = bid_vol_l1 + ask_vol_l1
                        state.order_book_imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0

                        # Orderbook imbalance - L5 (depth)
                        bid_vol_l5 = sum(size for _, size in ob.bids[:5])
                        ask_vol_l5 = sum(size for _, size in ob.asks[:5])
                        total_l5 = bid_vol_l5 + ask_vol_l5
                        state.order_book_imbalance_l5 = (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0.0
                        
                        # Spoofing detection - update detector and get probability
                        spoof_prob = self.spoof_detector.update(cid, ob.bids, ob.asks)
                        state.spoof_probability = spoof_prob
                        
                        # v7.2: VPIN 2.0 OFI-Based Toxic Flow Detection
                        # Uses Order Flow Imbalance (delta-depth) instead of static volume
                        vpin_detector_v2 = get_vpin_detector_v2(cid, window_size=50, alpha=0.1)
                        # Pass bid/ask volumes for OFI calculation
                        if bid_vol_l1 + ask_vol_l1 > 0:
                            vpin, signed_ofi = vpin_detector_v2.update(bid_vol_l1, ask_vol_l1)
                            state.vpin = vpin
                            state.signed_ofi = signed_ofi  # For "Toxic Alpha" piggybacking
                            state.vpin_cdf = vpin_detector_v2.get_vpin_cdf()  # CDF percentile
                        
                        # FIX: Compute edge_confidence (was always 0!)
                        # Edge = strong signal + low noise + low manipulation
                        imbalance_strength = abs(state.order_book_imbalance_l1)  # [0, 1]
                        spread_score = max(0, 1 - (state.spread / 0.10))  # <10% spread = good
                        vpin_safety = 1 - state.vpin  # Low VPIN = safe
                        spoof_safety = 1 - state.spoof_probability  # Low spoof = safe
                        
                        # Weighted combination: imbalance most important
                        state.edge_confidence = (
                            0.40 * imbalance_strength +   # Strong order flow signal
                            0.20 * spread_score +         # Tight spreads
                            0.20 * vpin_safety +          # No toxic flow
                            0.20 * spoof_safety           # No manipulation
                        )

                # Update binance price
                binance_price = self.price_streamer.get_price(m.asset)
                state.binance_price = binance_price
                open_price = self.open_prices.get(cid, binance_price)
                if open_price > 0:
                    state.binance_change = (binance_price - open_price) / open_price

                # Update futures data (focused on fast-updating features)
                futures = self.futures_streamer.get_state(m.asset)
                if futures:
                    # Order flow - THE EDGE
                    old_cvd = state.cvd
                    state.cvd = futures.cvd
                    state.cvd_acceleration = (futures.cvd - old_cvd) / 1e6 if old_cvd != 0 else 0.0
                    state.trade_flow_imbalance = futures.trade_flow_imbalance

                    # Ultra-short momentum
                    state.returns_1m = futures.returns_1m
                    state.returns_5m = futures.returns_5m
                    state.returns_10m = futures.returns_10m  # Properly computed from klines
                    
                    # === REGIME CLASSIFICATION & SWITCHING ===
                    # 1. Update volatility baseline (using realized_vol_1h)
                    current_vol = futures.realized_vol_1h
                    if not hasattr(self, "vol_history"): self.vol_history = {}  # Lazy init
                    if m.asset not in self.vol_history: self.vol_history[m.asset] = []
                    
                    self.vol_history[m.asset].append(current_vol)
                    if len(self.vol_history[m.asset]) > 100:  # Keep ~100 updates
                        self.vol_history[m.asset] = self.vol_history[m.asset][-100:]
                    
                    avg_vol = sum(self.vol_history[m.asset]) / len(self.vol_history[m.asset]) if self.vol_history[m.asset] else current_vol
                    
                    # 2. Classify Regime
                    regime, confidence = classify_regime(
                        m.asset, 
                        binance_price, 
                        returns_1m=state.returns_1m,
                        returns_5m=state.returns_5m,
                        volatility=current_vol,
                        avg_volatility=avg_vol
                    )
                    
                    # 3. Map to State Features
                    state.vol_regime = 1.0 if regime.value == "high_volatility" else (0.0 if regime.value == "low_volatility" else 0.5)
                    state.trend_regime = 1.0 if regime.value == "trending" else (-1.0 if regime.value == "mean_reverting" else 0.0)
                    
                    # 4. Apply Strategy Adjustments (Meta-Control)
                    adjustments = get_regime_adjustments(m.asset)
                    # Modify state params dynamically based on regime
                    # Note: These are "soft" adjustments that the strategy CAN use
                    state.regime_hold_time_mult = adjustments.get("hold_time", 1.0)
                    state.regime_size_mult = adjustments.get("position_size", 1.0)
                    state.regime_edge_mult = adjustments.get("edge_threshold", 1.0)
                    
                    # 5. Liquidation Heatmap Proximity Signal
                    state.liquidation_proximity = get_liq_signal(m.asset, binance_price)


                    # Microstructure - CRITICAL for 15-min
                    state.trade_intensity = futures.trade_intensity
                    state.large_trade_flag = futures.large_trade_flag

                    # Volatility
                    state.realized_vol_5m = futures.realized_vol_1h / 3.5 if futures.realized_vol_1h > 0 else 0.0
                    state.vol_expansion = futures.vol_ratio - 1.0

                    # === NEW HIGH-IMPACT SIGNALS ===
                    # Funding rate (extreme values precede reversals 60%+)
                    old_funding = getattr(state, '_prev_funding_rate', futures.funding_rate)
                    state.funding_rate = futures.funding_rate
                    state.funding_rate_velocity = futures.funding_rate - old_funding
                    state._prev_funding_rate = futures.funding_rate
                    
                    # Open Interest delta (tracks leveraged positioning)
                    state.open_interest_delta = futures.oi_change_1h
                    
                    # OI-Price divergence (key signal for trend continuation/reversal)
                    # OI up + price up = trend confirmation (bullish)
                    # OI up + price down = new shorts = bearish continuation
                    # OI down + price up = short covering = rally may exhaust
                    # OI down + price down = capitulation = potential bottom
                    price_direction = 1.0 if state.returns_5m > 0 else (-1.0 if state.returns_5m < 0 else 0.0)
                    oi_direction = 1.0 if state.open_interest_delta > 0.01 else (-1.0 if state.open_interest_delta < -0.01 else 0.0)
                    state.oi_price_divergence = price_direction * oi_direction  # +1 = confirming, -1 = diverging
                    
                    # Liquidation pressure (net pressure from forced selling)
                    state.liquidation_pressure = futures.liquidation_pressure
                    
                    # === EDGE CONFIDENCE (CONSOLIDATED) ===
                    # Combines orderbook signals (VPIN, spoof) with futures signals (funding, liquidation)
                    # NOTE: edge_confidence was computed earlier in orderbook block; this SUPPLEMENTS it
                    # by adding futures-derived signals. We take weighted average of both computations.
                    orderbook_edge = state.edge_confidence  # From lines 515-520
                    
                    # Futures-based edge signals
                    futures_edge_factors = [
                        abs(state.trade_flow_imbalance) * 0.25,    # Clear order flow = edge
                        abs(state.funding_rate * 100) * 0.20 if abs(state.funding_rate) > 0.0005 else 0,
                        abs(state.liquidation_pressure) * 0.20,    # Liq pressure = edge
                        (1.0 - state.vol_regime) * 0.15,           # Low vol = more predictable
                        abs(state.oi_price_divergence) * 0.20,     # OI-price divergence
                    ]
                    futures_edge = min(1.0, sum(futures_edge_factors))
                    
                    # Weighted combination: orderbook is real-time, futures is confirmatory
                    state.edge_confidence = 0.60 * orderbook_edge + 0.40 * futures_edge

                # Time remaining - CRITICAL
                state.time_remaining = (m.end_time - now).total_seconds() / 900
                
                # === NEW RESEARCH: Cross-Asset Features ===
                # ETH→BTC Lead-Lag (strongest cross-asset signal, -0.1178 correlation)
                if m.asset == "BTC":
                    eth_futures = self.futures_streamer.get_state("ETH")
                    if eth_futures:
                        state.eth_return_lag1 = eth_futures.returns_1m  # ETH return at t-1 for BTC
                
                # Time-of-Day: Volatility Peak Window (14:00-16:00 UTC)
                utc_hour = now.hour
                state.utc_hour = utc_hour
                state.is_volatility_peak = 1.0 if 14 <= utc_hour <= 16 else 0.0
                
                # === v7.4: FEE RATE FOR RL STATE ===
                # Give the agent visibility into current fee costs
                from helpers.polymarket_fees import fee_threshold_for_trade
                state.current_fee_rate = fee_threshold_for_trade(state.prob, self.trade_size)

                # Update position info in state
                pos = self.positions.get(cid)
                if pos and pos.size > 0:
                    state.has_position = True
                    state.position_side = pos.side
                    shares = pos.size / pos.entry_price
                    if pos.side == "UP":
                        state.position_pnl = (state.prob - pos.entry_price) * shares
                    else:
                        current_down_price = 1 - state.prob
                        state.position_pnl = (current_down_price - pos.entry_price) * shares
                else:
                    state.has_position = False
                    state.position_side = None
                    state.position_pnl = 0.0


                # === v7.2: VPIN HANDLED BY RL AGENT ===
                # The hard halt has been removed. The RL agent learns to manage
                # VPIN risk via reward shaping (penalty for holding during high VPIN).
                # The agent can now trade through natural imbalance while still
                # being sensitive to real toxic flow via its learned policy.
                
                # === v7.1 UPGRADE: PROBABILITY-BASED DYNAMIC EXIT ===
                # Intelligent exit based on market conviction and position PnL
                # Replaces fixed 3-min hard close with adaptive timing
                
                # v7.3: ARMS replaces hard $5 stop with dynamic ATR trailing stop
                NO_TRADE_ZONE = 0.33       # 5 min left - no new positions
                FALLBACK_MAX_LOSS = 10.00  # Fallback hard cap (2x old limit)
                
                # v7.3 FIX: Compute ACTUAL high/low from prob_history instead of synthetic ±0.01
                # This gives ARMS realistic ATR values based on actual market volatility
                if len(state.prob_history) >= 5:
                    recent_probs = state.prob_history[-5:]  # Last 5 observations (~2.5 seconds)
                    prob_high = max(recent_probs + [state.prob])
                    prob_low = min(recent_probs + [state.prob])
                else:
                    # Fallback: use volatility-based estimate if not enough history
                    vol_estimate = max(0.005, state.realized_vol_5m * 2)  # At least 0.5%
                    prob_high = state.prob + vol_estimate
                    prob_low = state.prob - vol_estimate
                
                update_arms_price(
                    state.asset,
                    high=prob_high,
                    low=prob_low,
                    close=state.prob
                )
                
                # 1. ARMS ATR TRAILING STOP - Replaces hard $5 stop
                if pos and pos.size > 0:
                    # v7.5 FIX: Update trailing stop tracking with correct prices
                    # - For UP positions: track highest UP token price (prob)
                    # - For DOWN positions: track highest DOWN token price (1 - prob)
                    if pos.side == "UP":
                        pos.highest_price = max(pos.highest_price, state.prob)
                        current_token_price = state.prob
                    else:
                        # DOWN token price increases when prob decreases
                        down_token_price = 1 - state.prob
                        pos.highest_price = max(pos.highest_price, down_token_price)
                        current_token_price = down_token_price
                    
                    # Check if ARMS trailing stop triggered
                    should_stop, stop_price, reason = get_trailing_stop_info(
                        asset=state.asset,
                        entry_price=pos.entry_price,
                        current_price=current_token_price,  # Use token price, not prob
                        highest_since_entry=pos.highest_price,
                        lowest_since_entry=pos.lowest_price,
                        side=pos.side
                    )
                    
                    if should_stop:
                        arms = get_arms()
                        atr = arms.get_atr(state.asset)
                        hurst = arms.get_hurst(state.asset)
                        mult = arms.get_atr_multiplier(state.asset)
                        print(f"    🔴 ARMS STOP: {state.asset} | {reason}")
                        print(f"       ATR={atr:.4f}, H={hurst:.2f}, Mult={mult:.1f}x, PnL=${state.position_pnl:+.2f}")
                        close_action = Action.SELL if pos.side == "UP" else Action.BUY
                        self.execute_action(cid, close_action, state)
                        continue
                    
                    # Fallback: Max loss cap (2x old limit, safety net only)
                    if state.position_pnl < -FALLBACK_MAX_LOSS:
                        print(f"    🔴 FALLBACK STOP: {state.asset} PnL ${state.position_pnl:.2f} < -${FALLBACK_MAX_LOSS}")
                        close_action = Action.SELL if pos.side == "UP" else Action.BUY
                        self.execute_action(cid, close_action, state)
                        continue
                
                # 2. PROBABILITY-BASED EXIT - v7.1 Upgrade
                if pos and pos.size > 0:
                    time_left_mins = state.time_remaining * 15  # Convert to minutes
                    
                    # 2a. WINNING TRADES (profitable) - Hold for max profit
                    if state.position_pnl > 0.50:  # $0.50+ profit
                        # Check if market has converged (high conviction)
                        market_converged = False
                        if pos.side == "UP" and state.prob >= 0.95:
                            market_converged = True  # Strong UP conviction
                        elif pos.side == "DOWN" and state.prob <= 0.05:
                            market_converged = True  # Strong DOWN conviction
                        
                        # Close if converged OR time running out
                        if market_converged or time_left_mins < 1.5:
                            reason = "PROB_CONVERGED" if market_converged else "TIME_LIMIT"
                            print(f"    ✅ PROFIT TAKE ({reason}): {state.asset} ${state.position_pnl:+.2f} @ {state.prob:.3f} | {time_left_mins:.1f}m left")
                            close_action = Action.SELL if pos.side == "UP" else Action.BUY
                            self.execute_action(cid, close_action, state)
                            continue
                    
                    # 2b. LOSING TRADES - Early exit if market disagrees
                    elif state.position_pnl < -1.00:  # More than $1 loss
                        # Check if market moved against us
                        market_disagrees = False
                        if pos.side == "UP" and state.prob < 0.45:
                            market_disagrees = True  # DOWN movement against UP position
                        elif pos.side == "DOWN" and state.prob > 0.55:
                            market_disagrees = True  # UP movement against DOWN position
                        
                        # Early exit at 5 min if losing and market disagrees
                        if market_disagrees and time_left_mins < 5.0:
                            print(f"    🔴 CUT LOSS: {state.asset} ${state.position_pnl:+.2f} @ {state.prob:.3f} | {time_left_mins:.1f}m left")
                            close_action = Action.SELL if pos.side == "UP" else Action.BUY
                            self.execute_action(cid, close_action, state)
                            continue
                    
                    # 2c. UNCERTAIN TRADES (small profit/loss) - Exit in optimal window
                    # Research shows 1.0-1.5 min is optimal liquidity window
                    if time_left_mins < 1.5:
                        print(f"    ⏰ OPTIMAL EXIT: {state.asset} ${state.position_pnl:+.2f} @ {state.prob:.3f} | {time_left_mins:.1f}m left")
                        close_action = Action.SELL if pos.side == "UP" else Action.BUY
                        self.execute_action(cid, close_action, state)
                        continue
                
                # Legacy: For non-RL strategies, force close at very_near_expiry
                if pos and pos.size > 0 and state.very_near_expiry:
                    if not isinstance(self.strategy, (RLStrategy, EarnHFTStrategy)):
                        print(f"    ⏰ EARLY CLOSE: {pos.asset}")
                        close_action = Action.SELL if pos.side == "UP" else Action.BUY
                        self.execute_action(cid, close_action, state)
                        continue

                # Get action from strategy
                action = self.strategy.act(state)

                # RL/EarnHFT: Store experience EVERY tick (dense learning signal)
                if isinstance(self.strategy, (RLStrategy, EarnHFTStrategy)) and self.strategy.training:
                    prev_state = self.prev_states.get(cid)
                    if prev_state:
                        step_reward = self._compute_step_reward(cid, state, action, pos)
                        # Episode not done unless market expired
                        self.strategy.store(prev_state, action, step_reward, state, done=False)

                    # Deep copy state for next iteration
                    self.prev_states[cid] = copy.deepcopy(state)

                # Execute
                if action != Action.HOLD:
                    # === EXTREME PROBABILITY FILTER (v7.5: relaxed) ===
                    # Only block when market is essentially resolved - let agent learn rest
                    if state.prob > 0.98 or state.prob < 0.02:
                        self.filtered_count += 1
                        if self.filtered_count % 20 == 0:
                            print(f"    [FILTERED x{self.filtered_count}] {state.asset}: market_resolved ({state.prob:.2f})")
                        continue
                    

                    
                    # EdgeConfidenceGate filtering - "when NOT to trade" (v7.4: fee-aware)
                    spoof_prob = getattr(state, 'spoof_probability', 0.0)
                    # Check if we're in training mode
                    is_training = isinstance(self.strategy, (RLStrategy, EarnHFTStrategy)) and self.strategy.training
                    
                    should_execute, filter_reason = self.edge_gate.should_trade(
                        edge_confidence=state.edge_confidence,
                        spread=state.spread,
                        time_remaining=state.time_remaining,
                        vol_regime=state.vol_regime,
                        spoof_probability=spoof_prob,
                        probability=state.prob,
                        trade_size=self.trade_size,
                    )
                    
                    if should_execute:
                        self.execute_action(cid, action, state)
                    else:
                        self.filtered_count += 1
                        # Log filtered trades occasionally for debugging
                        if self.filtered_count % 20 == 0:
                            print(f"    [FILTERED x{self.filtered_count}] {state.asset}: {filter_reason}")

            # Status update every 10 ticks (console), but dashboard every tick
            if tick % 10 == 0:
                self.print_status()
            else:
                # Update dashboard state every tick for responsiveness
                self._update_dashboard_only()

            # RL training: emit buffer progress every tick
            if isinstance(self.strategy, (RLStrategy, EarnHFTStrategy)) and self.strategy.training:
                # For EarnHFT, get experiences from currently active agent
                if isinstance(self.strategy, EarnHFTStrategy):
                    active_agent = self.strategy.current_agent
                    if active_agent:
                        buffer_size = len(active_agent.experiences)
                        agent_buffer_size = active_agent.buffer_size
                        experiences_ref = active_agent.experiences
                    else:
                        buffer_size = 0
                        agent_buffer_size = 128
                        experiences_ref = []
                else:
                    buffer_size = len(self.strategy.experiences)
                    agent_buffer_size = self.strategy.buffer_size
                    experiences_ref = self.strategy.experiences
                    
                # Compute average reward from recent experiences
                avg_reward = None
                if buffer_size > 0:
                    recent_rewards = [exp.reward for exp in experiences_ref[-50:]]  # Last 50
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                emit_rl_buffer(buffer_size, agent_buffer_size, avg_reward)

                # PPO update when buffer is full
                if buffer_size >= agent_buffer_size:
                    # Get buffer rewards before update clears them
                    buffer_rewards = [exp.reward for exp in experiences_ref]
                    
                    # Call the appropriate update method
                    metrics = self.strategy.update()
                    
                    if metrics:
                        # Handle EarnHFT multi-agent metrics
                        if isinstance(self.strategy, EarnHFTStrategy):
                            print(f"  [EarnHFT] Training update completed!")
                            if 'agents_updated' in metrics:
                                for agent_name, agent_metrics in metrics.get('agent_metrics', {}).items():
                                    print(f"    [{agent_name}] loss={agent_metrics.get('policy_loss', 0):.4f} "
                                          f"v_loss={agent_metrics.get('value_loss', 0):.4f} "
                                          f"ent={agent_metrics.get('entropy', 0):.3f}")
                        else:
                            print(f"  [RL] loss={metrics['policy_loss']:.4f} "
                                  f"v_loss={metrics['value_loss']:.4f} "
                                  f"ent={metrics['entropy']:.3f} "
                                  f"kl={metrics['approx_kl']:.4f} "
                                  f"ev={metrics['explained_variance']:.2f}")
                            
                        # Send to dashboard
                        metrics['buffer_size'] = buffer_size
                        update_rl_metrics(metrics)
                        
                        # Log to CSV
                        if self.logger:
                            self.logger.log_update(
                                metrics=metrics,
                                buffer_rewards=buffer_rewards,
                                cumulative_pnl=self.total_pnl,
                                cumulative_trades=self.trade_count,
                                cumulative_wins=self.win_count
                            )
                        
                        # AUTO-SAVE: Checkpoint every 50 PPO updates to prevent crash loss
                        if not hasattr(self, '_ppo_update_count'):
                            self._ppo_update_count = 0
                        self._ppo_update_count += 1
                        
                        if self._ppo_update_count % 50 == 0:
                            checkpoint_name = f"rl_checkpoint_{self._ppo_update_count}"
                            self.strategy.save(checkpoint_name)
                            print(f"  💾 AUTO-SAVE: {checkpoint_name} (update #{self._ppo_update_count})")

    def _update_dashboard_only(self):
        """Update dashboard state without printing to console."""
        now = datetime.now(timezone.utc)
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                vel = state._velocity()
                dashboard_markets[cid] = {
                    'asset': m.asset,
                    'prob': state.prob,
                    'time_left': mins_left,
                    'velocity': vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                    }

        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        win_rate = self.win_count / max(1, self.trade_count) * 100

        print(f"\n[{now.strftime('%H:%M:%S')}] {self.strategy.name.upper()}")
        print(f"  PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count} | Win: {win_rate:.0f}%")

        # Prepare dashboard data
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                pos_str = f"{pos.side} ${pos.size:.0f}" if pos and pos.size > 0 else "FLAT"
                vel = state._velocity()
                print(f"  {m.asset}: prob={state.prob:.3f} vel={vel:+.3f} | {pos_str} | {mins_left:.1f}m")

                # Dashboard data
                dashboard_markets[cid] = {
                    'asset': m.asset,
                    'prob': state.prob,
                    'time_left': mins_left,
                    'velocity': vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                    }

        # Update dashboard
        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_final_stats(self):
        """Print final results."""
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Total PnL: ${self.total_pnl:+.2f}")
        print(f"Trades: {self.trade_count}")
        print(f"Win Rate: {self.win_count / max(1, self.trade_count) * 100:.1f}%")
        if self.filtered_count > 0:
            print(f"Filtered (low-edge): {self.filtered_count}")
            print(f"Trade Selectivity: {self.trade_count / max(1, self.trade_count + self.filtered_count) * 100:.0f}%")

    async def run(self):
        """Run the trading engine."""
        self.running = True
        self.refresh_markets()

        if not self.markets:
            print("No markets to trade!")
            return

        tasks = [
            self.price_streamer.stream(),
            self.orderbook_streamer.stream(),
            self.futures_streamer.stream(),
            self.decision_loop(),
        ]

        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass  # Handle in finally
        finally:
            print("\n\nShutting down...")
            self.running = False
            self.price_streamer.stop()
            self.orderbook_streamer.stop()
            self.futures_streamer.stop()
            self.close_all_positions()
            self.print_final_stats()

            # Save RL model if training
            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                self.strategy.save("rl_model")
                print("  [RL] Model saved to rl_model.safetensors")


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading")
    parser.add_argument(
        "strategy",
        nargs="?",
        choices=AVAILABLE_STRATEGIES,
        help="Strategy to run"
    )
    parser.add_argument("--train", action="store_true", help="Enable training mode for RL")
    parser.add_argument("--size", type=float, default=50.0, help="Trade size in $")
    parser.add_argument("--load", type=str, help="Load RL model from file")
    parser.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading with real money (requires LIVE_TRADING=true in .env)")

    args = parser.parse_args()

    if not args.strategy:
        print("Available strategies:")
        for name in AVAILABLE_STRATEGIES:
            print(f"  - {name}")
        print("\nUsage: python run.py <strategy>")
        print("       python run.py rl --train")
        print("       python run.py rl --train --dashboard")
        print("       python run.py rl --live --size 5   # LIVE trading with $5 positions")
        return

    # Start dashboard in background if requested
    if args.dashboard:
        if DASHBOARD_AVAILABLE:
            dashboard_thread = threading.Thread(
                target=run_dashboard,
                kwargs={'port': args.port},
                daemon=True
            )
            dashboard_thread.start()
            import time
            time.sleep(1)  # Give dashboard time to start
        else:
            print("Warning: Dashboard not available. Install flask-socketio.")

    # Create strategy
    strategy = create_strategy(args.strategy)

    # RL-specific setup
    # RL/Lacuna setup
    if isinstance(strategy, (RLStrategy, LacunaStrategy)):
        if args.load:
            strategy.load(args.load)
            print(f"Loaded model from {args.load}")
        
        if hasattr(strategy, 'training'): # Only RL has training mode toggle usually
             if args.train:
                 strategy.train()
                 print("Training mode enabled")
             else:
                 strategy.eval()

    # Run
    live_mode = args.live and is_live_trading_enabled()
    if args.live and not live_mode:
        print("\n⚠️  --live flag set but LIVE_TRADING is not enabled in .env")
        print("    Set LIVE_TRADING=true in .env to enable live trading.\n")
    
    engine = TradingEngine(strategy, trade_size=args.size, live_mode=live_mode)
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
