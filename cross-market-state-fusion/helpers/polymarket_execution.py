#!/usr/bin/env python3
"""
Polymarket order execution layer using py-clob-client.

Handles:
- Order placement (buy/sell)
- Order cancellation
- Position tracking
- Safety checks (max exposure, slippage)
"""
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import py-clob-client
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, AssetType
from py_clob_client.order_builder.constants import BUY, SELL


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_size: Optional[float] = None
    error: Optional[str] = None
    status: str = "UNKNOWN"  # LIVE, MATCHED, CANCELLED, DELAYED


@dataclass
class SafetyConfig:
    """Safety configuration for live trading."""
    max_position_size: float = 500.0  # Max per-trade size in $
    max_exposure: float = 2000.0  # Max total exposure across all positions
    slippage_tolerance: float = 0.02  # 2% max slippage
    min_position_size: float = 1.0  # Minimum trade size in $
    enabled: bool = True  # Kill switch


class PolymarketExecutor:
    """
    Executes real orders on Polymarket using py-clob-client.
    
    Usage:
        executor = PolymarketExecutor()
        result = executor.place_order(
            token_id="...",
            side="BUY",
            size=10.0,
            price=0.45
        )
    """
    
    def __init__(self, safety_config: SafetyConfig = None):
        """Initialize the executor with credentials from .env"""
        # v8.1: Read safety limits from .env if not provided
        if safety_config is None:
            self.safety = SafetyConfig(
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "500")),
                max_exposure=float(os.getenv("MAX_EXPOSURE", "2000")),
            )
        else:
            self.safety = safety_config
        
        # Load credentials from environment
        self.private_key = os.getenv("PRIVATE_KEY")
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        self.funder = os.getenv("POLYMARKET_PROXY") or self.wallet_address
        self.chain_id = int(os.getenv("CHAIN_ID", "137"))
        self.host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
        self.signature_type = int(os.getenv("SIGNATURE_TYPE", "2"))
        
        # Check if live trading is enabled
        self.live_enabled = os.getenv("LIVE_TRADING", "false").lower() == "true"
        
        # Validate credentials
        if not self.private_key:
            raise ValueError("PRIVATE_KEY not set in .env file")
        if not self.wallet_address:
            raise ValueError("WALLET_ADDRESS not set in .env file")
        
        # Initialize client
        self.client = None
        self._init_client()
        
        # Track current exposure
        self.current_exposure = 0.0
        self.open_positions: Dict[str, dict] = {}
        
    def _init_client(self):
        """Initialize the CLOB client and derive API credentials."""
        try:
            self.client = ClobClient(
                host=self.host,
                chain_id=self.chain_id,
                key=self.private_key,
                signature_type=self.signature_type,
                funder=self.funder,
            )
            
            # Derive or create API credentials
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            print(f"[EXECUTOR] Initialized for wallet: {self.wallet_address[:10]}...")
            print(f"[EXECUTOR] Live trading: {'ENABLED' if self.live_enabled else 'DISABLED (paper mode)'}")
            
        except Exception as e:
            print(f"[EXECUTOR] Failed to initialize client: {e}")
            raise
    
    def _check_safety(self, size: float, price: float) -> Tuple[bool, str]:
        """
        Check if an order passes safety checks.
        
        Returns:
            (passed, reason)
        """
        if not self.safety.enabled:
            return True, "Safety disabled"
            
        # Check position size
        if size > self.safety.max_position_size:
            return False, f"Position size ${size:.2f} exceeds max ${self.safety.max_position_size:.2f}"
            
        if size < self.safety.min_position_size:
            return False, f"Position size ${size:.2f} below minimum ${self.safety.min_position_size:.2f}"
            
        # Check total exposure
        new_exposure = self.current_exposure + size
        if new_exposure > self.safety.max_exposure:
            return False, f"Would exceed max exposure: ${new_exposure:.2f} > ${self.safety.max_exposure:.2f}"
            
        # Check price sanity (not too close to 0 or 1)
        if price < 0.01 or price > 0.99:
            return False, f"Price {price:.3f} outside safe range [0.01, 0.99]"
            
        return True, "OK"
    
    def place_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        size: float,  # in dollars for BUY, or shares for SELL
        price: float,  # probability (0-1)
        order_type: str = "GTC",  # GTC, FOK, GTD
        shares_override: float = None,  # For SELL: exact shares to sell
        prefer_maker: bool = True,  # v7.4: Prefer maker orders to avoid fees
        urgency: float = 0.5,  # v7.4: 0=no urgency (maker), 1=critical (taker)
    ) -> OrderResult:
        """
        Place an order on Polymarket.
        
        v7.4 Fee-Aware Execution:
        - prefer_maker=True + urgency<0.7: Places limit order inside spread (maker, zero fees)
        - prefer_maker=True + urgency>=0.7: Places at market (still tries limit)
        - prefer_maker=False or urgency>=0.9: Market order (taker, pays fees)
        
        Args:
            token_id: The token ID (UP or DOWN token)
            side: "BUY" or "SELL"
            size: Position size in dollars (for BUY) or original dollar size (for SELL)
            price: Limit price (probability 0-1)
            order_type: Order type (GTC = Good Till Cancel)
            shares_override: If provided, sell this exact number of shares (for closing positions)
            prefer_maker: If True, try to place maker orders (zero fees)
            urgency: How urgent the trade is (0-1). Low urgency = more aggressive maker placement
            
        Returns:
            OrderResult with success status and details
        """
        # Safety checks (skip for SELL since we're closing)
        if side.upper() == "BUY":
            passed, reason = self._check_safety(size, price)
            if not passed:
                return OrderResult(success=False, error=f"Safety check failed: {reason}")
        
        
        # NOTE: Balance check disabled - get_token_balance API returning 0 even when shares owned
        # This was blocking valid SELL orders. The original Polymarket error is less harmful.
        # TODO: Fix get_token_balance API call to work correctly
        
        # v7.4: Calculate maker price (inside the spread for fee-free execution)
        maker_price = price
        is_maker_order = False
        if prefer_maker and urgency < 0.9:
            # Adjust price to be inside spread (maker-friendly)
            # Lower urgency = more passive (further from mid)
            spread_offset = 0.005 * (1.0 - urgency)  # 0.5% at low urgency, 0% at high
            if side.upper() == "BUY":
                maker_price = price - spread_offset  # Buy below mid
            else:
                maker_price = price + spread_offset  # Sell above mid
            is_maker_order = urgency < 0.7  # Only truly maker if low urgency
        
        # Check if live trading is enabled
        if not self.live_enabled:
            order_type_str = "MAKER" if is_maker_order else "TAKER"
            print(f"[EXECUTOR] PAPER MODE - Would place {side} ${size:.2f} @ {maker_price:.3f} ({order_type_str})")
            return OrderResult(
                success=True,
                order_id="paper_" + token_id[:8],
                filled_price=maker_price,
                filled_size=size,
                error=None
            )
        
        try:
            # Convert side string to constant
            order_side = BUY if side.upper() == "BUY" else SELL
            
            # Calculate shares
            if side.upper() == "SELL" and shares_override is not None:
                # For SELL: use the exact shares we own
                shares = shares_override
            elif side.upper() == "SELL":
                # Fallback: try to estimate shares from size at current price
                shares = size / max(price, 0.01)
            else:
                # For BUY: calculate shares from dollar amount
                shares = size / price
            
            # v8.3: ENFORCE MINIMUM 5 SHARES (Polymarket requirement)
            MIN_SHARES = 5.0
            if shares < MIN_SHARES:
                print(f"[EXECUTOR] Bumping shares {shares:.2f} -> {MIN_SHARES} (minimum)")
                shares = MIN_SHARES
            
            # Create order args - v7.4: use maker_price for fee optimization
            order_args = OrderArgs(
                token_id=token_id,
                price=maker_price,  # v7.4: Use adjusted price for maker orders
                size=shares,
                side=order_side,
            )
            
            # Place the order - v7.4: Log whether maker or taker
            order_type_str = "MAKER" if is_maker_order else "TAKER"
            print(f"[EXECUTOR] Placing {side} {order_type_str}: {shares:.2f} shares @ {maker_price:.3f}")
            response = self.client.create_and_post_order(order_args)
            
            if response and response.get("success"):
                order_id = response.get("orderID") or response.get("order_id")
                
                # Update exposure tracking
                if side.upper() == "BUY":
                    self.current_exposure += size
                    self.open_positions[order_id] = {
                        "token_id": token_id,
                        "side": side,
                        "size": size,
                        "shares": shares,
                        "price": price,
                    }
                else:
                    # SELL - reduce exposure
                    self.current_exposure = max(0, self.current_exposure - size)
                
                print(f"[EXECUTOR] Order placed successfully: {order_id}")
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    filled_price=price,
                    filled_size=size,
                )
            else:
                error_msg = response.get("error", "Unknown error") if response else "No response"
                print(f"[EXECUTOR] Order failed: {error_msg}")
                return OrderResult(success=False, error=error_msg)
                
        except Exception as e:
            print(f"[EXECUTOR] Order error: {e}")
            return OrderResult(success=False, error=str(e))
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.live_enabled:
            print(f"[EXECUTOR] PAPER MODE - Would cancel order {order_id}")
            return True
            
        try:
            response = self.client.cancel(order_id)
            if response and response.get("success"):
                # Update exposure tracking
                if order_id in self.open_positions:
                    pos = self.open_positions.pop(order_id)
                    self.current_exposure -= pos["size"]
                print(f"[EXECUTOR] Cancelled order: {order_id}")
                return True
            return False
        except Exception as e:
            print(f"[EXECUTOR] Cancel error: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        if not self.live_enabled:
            print("[EXECUTOR] PAPER MODE - Would cancel all orders")
            return 0
            
        try:
            response = self.client.cancel_all()
            cancelled = response.get("canceled", []) if response else []
            
            # Clear exposure tracking
            self.open_positions.clear()
            self.current_exposure = 0.0
            
            print(f"[EXECUTOR] Cancelled {len(cancelled)} orders")
            return len(cancelled)
        except Exception as e:
            print(f"[EXECUTOR] Cancel all error: {e}")
            return 0
    
    def get_open_orders(self) -> list:
        """Get list of open orders."""
        if not self.live_enabled:
            return []
            
        try:
            response = self.client.get_orders()
            return response if response else []
        except Exception as e:
            print(f"[EXECUTOR] Get orders error: {e}")
            return []
    
    def get_balance(self) -> float:
        """Get USDC balance."""
        try:
            response = self.client.get_balance_allowance()
            if response:
                # Balance is returned in wei (1e6 for USDC)
                balance = float(response.get("balance", 0)) / 1e6
                return balance
            return 0.0
        except Exception as e:
            print(f"[EXECUTOR] Balance error: {e}")
            return 0.0
    
    def get_token_balance(self, token_id: str) -> float:
        """
        Get the balance of a specific outcome token (shares owned).
        
        Args:
            token_id: The conditional token ID to check
            
        Returns:
            Number of shares owned (0 if none or error)
        """
        try:
            response = self.client.get_balance_allowance(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id
            )
            if response:
                # Balance is in shares (with 6 decimals)
                balance = float(response.get("balance", 0)) / 1e6
                return balance
            return 0.0
        except Exception as e:
            # Silently return 0 - this is expected when we don't own shares
            return 0.0
    
    def close_position(self, token_id: str, current_price: float) -> OrderResult:
        """
        Close an existing position by selling the token.
        
        Args:
            token_id: Token to sell
            current_price: Current market price
            
        Returns:
            OrderResult
        """
        # Find position for this token
        for order_id, pos in list(self.open_positions.items()):
            if pos["token_id"] == token_id:
                # Place a SELL order to close
                return self.place_order(
                    token_id=token_id,
                    side="SELL",
                    size=pos["size"],
                    price=current_price,
                )
        
        return OrderResult(success=False, error="No position found for token")
    
    def wait_for_fill(self, order_id: str, timeout: float = 5.0, poll_interval: float = 0.5) -> OrderResult:
        """
        Wait for an order to be filled/matched. Replaces the 15s sleep.
        
        Polls the order status every poll_interval seconds until:
        - Order is MATCHED/FILLED
        - Order is CANCELLED
        - Timeout is reached
        
        Args:
            order_id: The order ID to check
            timeout: Max seconds to wait (default 5s)
            poll_interval: Seconds between status checks (default 0.5s)
            
        Returns:
            OrderResult with final status
        """
        import time
        
        if not self.live_enabled or order_id.startswith("paper_"):
            return OrderResult(success=True, order_id=order_id, status="PAPER_FILLED")
        
        start_time = time.time()
        last_status = "PENDING"
        
        while (time.time() - start_time) < timeout:
            try:
                order = self.client.get_order(order_id)
                if order:
                    status = order.get("status", "UNKNOWN")
                    last_status = status
                    
                    if status in ["MATCHED", "FILLED"]:
                        print(f"[EXECUTOR] ✅ Order {order_id[:8]}... FILLED")
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            filled_price=float(order.get("price", 0)),
                            filled_size=float(order.get("size_matched", 0)),
                            status="MATCHED"
                        )
                    elif status in ["CANCELLED", "EXPIRED"]:
                        print(f"[EXECUTOR] ❌ Order {order_id[:8]}... {status}")
                        return OrderResult(success=False, order_id=order_id, status=status, error=status)
                    # LIVE or OPEN - keep waiting
                    
            except Exception as e:
                print(f"[EXECUTOR] Status check error: {e}")
            
            time.sleep(poll_interval)
        
        # Timeout - return last known status
        print(f"[EXECUTOR] ⏰ Order {order_id[:8]}... timeout (last: {last_status})")
        return OrderResult(success=True, order_id=order_id, status=last_status, error="TIMEOUT")
    
    def get_detailed_balance(self) -> dict:
        """
        Get detailed balance and exposure information.
        
        Returns:
            dict with:
                - usdc_balance: Available USDC
                - current_exposure: Total $ in open positions
                - position_count: Number of open positions
        """
        try:
            balance = self.get_balance()
            return {
                "usdc_balance": balance,
                "current_exposure": self.current_exposure,
                "position_count": len(self.open_positions),
                "max_exposure": self.safety.max_exposure,
            }
        except Exception as e:
            print(f"[EXECUTOR] Balance fetch error: {e}")
            return {
                "usdc_balance": 0.0,
                "current_exposure": self.current_exposure,
                "position_count": len(self.open_positions),
                "max_exposure": self.safety.max_exposure,
            }
    
    def emergency_stop(self):
        """Emergency stop - cancel all orders and disable trading."""
        print("[EXECUTOR] ⚠️ EMERGENCY STOP TRIGGERED")
        self.safety.enabled = False  # Prevent new orders
        self.cancel_all_orders()
        self.live_enabled = False
        print("[EXECUTOR] All orders cancelled, trading disabled")


# Singleton instance
_executor: Optional[PolymarketExecutor] = None


def get_executor() -> PolymarketExecutor:
    """Get or create the singleton executor instance."""
    global _executor
    if _executor is None:
        _executor = PolymarketExecutor()
    return _executor


def is_live_trading_enabled() -> bool:
    """Check if live trading is enabled in .env"""
    load_dotenv()
    return os.getenv("LIVE_TRADING", "false").lower() == "true"


if __name__ == "__main__":
    # Test initialization
    print("Testing Polymarket Executor...")
    print(f"Live trading enabled: {is_live_trading_enabled()}")
    
    try:
        executor = get_executor()
        print(f"Executor initialized successfully")
        print(f"Current exposure: ${executor.current_exposure:.2f}")
        print(f"Max exposure: ${executor.safety.max_exposure:.2f}")
    except Exception as e:
        print(f"Failed to initialize: {e}")
