#!/usr/bin/env python3
"""
Polymarket 15-Minute Market Fee Calculator (v7.4).

Implements the quadratic fee formula for 15-minute crypto markets:
- Taker-only fees (makers pay nothing)
- Fee peaks at 50% probability (~3-4% round-trip)
- Fee approaches zero at probability extremes

Based on official Polymarket documentation:
- feeRate=0.25, exponent=2 (quadratic)
- Fee = 0.25 * 4 * p * (1-p) * shares

Usage:
    from helpers.polymarket_fees import calculate_fee, round_trip_fee, fee_threshold_for_trade
    
    # Calculate single-side fee
    buy_fee = calculate_fee(price=0.50, shares=50, is_buy=True)
    
    # Calculate round-trip fee
    rt_fee = round_trip_fee(price=0.50, shares=50)
    
    # Get minimum edge needed to overcome fees
    min_edge = fee_threshold_for_trade(price=0.50, trade_size=25.0)
"""

from typing import Tuple


# Fee formula constants (calibrated to match official Polymarket tables)
# Official: 100 shares at 50% = 1.5625 tokens fee
# Formula: fee_tokens = base_rate * 4 * p * (1-p) * shares
# At p=0.50: fee = base_rate * 4 * 0.25 * shares = base_rate * shares
# For fee = 1.5625 with shares=100: base_rate = 1.5625/100 = 0.015625
FEE_RATE = 0.015625  # Base rate calibrated to official tables


def calculate_fee(price: float, shares: float, is_buy: bool) -> float:
    """
    Calculate Polymarket 15-min market taker fee.
    
    Formula (calibrated to official tables):
    fee_tokens = FEE_RATE * 4 * p * (1-p) * shares
    
    At 50%: fee = 0.015625 * 4 * 0.25 * shares = 0.015625 * shares
    For 100 shares at 50%: fee = 1.5625 tokens ≈ $0.78 buy, $1.56 sell
    
    The fee is deducted from proceeds:
    - Buying: fee in tokens (valued at token price)
    - Selling: fee in USDC (direct dollars)
    
    Args:
        price: Current market probability (0-1)
        shares: Number of shares being traded
        is_buy: True for buy orders, False for sell orders
        
    Returns:
        Fee amount in dollars
    """
    if price <= 0 or price >= 1:
        return 0.0  # No fee at extremes
    
    # Quadratic fee factor: peaks at 1.0 when p=0.50
    fee_factor = 4 * price * (1 - price)
    
    # Fee in token units
    fee_tokens = FEE_RATE * fee_factor * shares
    
    if is_buy:
        # Buy: fee is in tokens, value = fee_tokens * price
        return fee_tokens * price
    else:
        # Sell: fee is in USDC, value = fee_tokens directly
        return fee_tokens


def round_trip_fee(price: float, shares: float) -> float:
    """
    Calculate total fee for a round-trip trade (entry + exit) at the same price.
    
    Args:
        price: Market probability (0-1)
        shares: Number of shares
        
    Returns:
        Total round-trip fee in dollars
    """
    return calculate_fee(price, shares, is_buy=True) + calculate_fee(price, shares, is_buy=False)


def fee_threshold_for_trade(price: float, trade_size: float) -> float:
    """
    Calculate minimum edge (as percentage) needed to overcome round-trip fees.
    
    This is the break-even edge: any signal with edge below this is unprofitable
    after fees.
    
    Args:
        price: Market probability (0-1)
        trade_size: Trade size in dollars
        
    Returns:
        Minimum required edge as a decimal (e.g., 0.05 = 5%)
    """
    if price <= 0 or price >= 1 or trade_size <= 0:
        return 0.0
    
    shares = trade_size / price
    rt_fee = round_trip_fee(price, shares)
    
    return rt_fee / trade_size


def get_fee_info(price: float, trade_size: float) -> dict:
    """
    Get comprehensive fee information for a potential trade.
    
    Args:
        price: Market probability (0-1)
        trade_size: Trade size in dollars
        
    Returns:
        Dictionary with fee details
    """
    if price <= 0 or price >= 1:
        return {
            "shares": 0,
            "buy_fee": 0,
            "sell_fee": 0,
            "round_trip_fee": 0,
            "fee_as_pct": 0,
            "min_edge_required": 0,
        }
    
    shares = trade_size / price
    buy_fee = calculate_fee(price, shares, is_buy=True)
    sell_fee = calculate_fee(price, shares, is_buy=False)
    rt_fee = buy_fee + sell_fee
    
    return {
        "shares": shares,
        "buy_fee": buy_fee,
        "sell_fee": sell_fee,
        "round_trip_fee": rt_fee,
        "fee_as_pct": rt_fee / trade_size if trade_size > 0 else 0,
        "min_edge_required": rt_fee / trade_size + 0.02,  # 2% buffer
    }


def is_maker_order() -> bool:
    """
    Check if order is a maker order (limit order adding liquidity).
    
    Maker orders pay ZERO fees and earn rebates.
    
    Note: In current implementation, we always assume taker.
    Tier 3 implementation will add maker order support.
    """
    return False  # TODO: Tier 3 implementation


# Convenience functions for EdgeConfidenceGate integration
def should_skip_due_to_fees(
    edge_confidence: float,
    price: float,
    trade_size: float,
    buffer: float = 0.02
) -> Tuple[bool, str]:
    """
    Check if trade should be skipped due to insufficient edge vs fees.
    
    Args:
        edge_confidence: Computed edge score [0, 1]
        price: Market probability (0-1)
        trade_size: Trade size in dollars
        buffer: Additional buffer above fee threshold (default 2%)
        
    Returns:
        (should_skip, reason) tuple
    """
    min_edge = fee_threshold_for_trade(price, trade_size) + buffer
    
    if edge_confidence < min_edge:
        return True, f"below_fee_threshold ({edge_confidence:.2%} < {min_edge:.2%})"
    
    return False, "fee_check_passed"


if __name__ == "__main__":
    # Test the fee calculations
    print("Polymarket 15-Min Fee Calculator Test")
    print("=" * 50)
    
    # Test at various probability points
    test_prices = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    trade_size = 25.0  # $25 trade
    
    print(f"\nTrade Size: ${trade_size}")
    print(f"{'Price':<8} {'Shares':<8} {'Buy Fee':<10} {'Sell Fee':<10} {'RT Fee':<10} {'Fee %':<8}")
    print("-" * 60)
    
    for price in test_prices:
        info = get_fee_info(price, trade_size)
        print(f"${price:.2f}   {info['shares']:<8.1f} ${info['buy_fee']:<9.2f} ${info['sell_fee']:<9.2f} ${info['round_trip_fee']:<9.2f} {info['fee_as_pct']*100:<7.1f}%")
    
    print("\n" + "=" * 50)
    print("Fee is highest at 50% probability, lowest at extremes.")
