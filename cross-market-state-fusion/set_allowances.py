#!/usr/bin/env python3
"""
Set Polymarket token allowances for trading.

This script approves the required contracts to spend your USDC and
Conditional Tokens (CTF). This is required for SELLING shares.

Based on official Polymarket code:
https://gist.github.com/poly-rodr/44313920481de58d5a3f6d1f8226bd5e
"""
import os
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# Polygon RPC
RPC_URL = "https://polygon-rpc.com"

# Contract addresses
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Exchange contracts to approve
EXCHANGES = {
    "CTF_EXCHANGE": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "NEG_RISK_CTF_EXCHANGE": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "NEG_RISK_ADAPTER": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

# Max approval amount
MAX_INT = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

# ABIs
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    }
]

CTF_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"}
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "operator", "type": "address"}
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]


def check_allowances(web3, pub_key):
    """Check current allowance status."""
    usdc = web3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)
    ctf = web3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
    
    print("\n📊 Current Allowance Status:")
    print("-" * 50)
    
    all_approved = True
    
    for name, exchange in EXCHANGES.items():
        # Check USDC allowance
        usdc_allowance = usdc.functions.allowance(pub_key, exchange).call()
        usdc_ok = usdc_allowance > 0
        
        # Check CTF approval
        ctf_approved = ctf.functions.isApprovedForAll(pub_key, exchange).call()
        
        status = "✅" if (usdc_ok and ctf_approved) else "❌"
        print(f"{status} {name}:")
        print(f"   USDC: {'✅ Approved' if usdc_ok else '❌ NOT APPROVED'}")
        print(f"   CTF:  {'✅ Approved' if ctf_approved else '❌ NOT APPROVED'}")
        
        if not (usdc_ok and ctf_approved):
            all_approved = False
    
    return all_approved


def set_allowances(web3, priv_key, pub_key, chain_id=137):
    """Set all required allowances."""
    usdc = web3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)
    ctf = web3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
    
    print("\n🔧 Setting Allowances...")
    print("-" * 50)
    
    for name, exchange in EXCHANGES.items():
        print(f"\n📍 {name} ({exchange[:10]}...)")
        
        # Get current nonce
        nonce = web3.eth.get_transaction_count(pub_key)
        
        # Check if USDC already approved
        usdc_allowance = usdc.functions.allowance(pub_key, exchange).call()
        if usdc_allowance == 0:
            print("   🔄 Approving USDC...")
            try:
                txn = usdc.functions.approve(
                    exchange, 
                    int(MAX_INT, 0)
                ).build_transaction({
                    "chainId": chain_id,
                    "from": pub_key,
                    "nonce": nonce,
                    "gas": 100000,
                    "gasPrice": web3.eth.gas_price,
                })
                signed = web3.eth.account.sign_transaction(txn, private_key=priv_key)
                tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
                receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                print(f"   ✅ USDC approved! Tx: {tx_hash.hex()[:20]}...")
                nonce += 1
            except Exception as e:
                print(f"   ❌ USDC approval failed: {e}")
        else:
            print("   ✅ USDC already approved")
        
        # Check if CTF already approved
        ctf_approved = ctf.functions.isApprovedForAll(pub_key, exchange).call()
        if not ctf_approved:
            print("   🔄 Approving CTF (Conditional Tokens)...")
            try:
                nonce = web3.eth.get_transaction_count(pub_key)
                txn = ctf.functions.setApprovalForAll(
                    exchange,
                    True
                ).build_transaction({
                    "chainId": chain_id,
                    "from": pub_key,
                    "nonce": nonce,
                    "gas": 100000,
                    "gasPrice": web3.eth.gas_price,
                })
                signed = web3.eth.account.sign_transaction(txn, private_key=priv_key)
                tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
                receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                print(f"   ✅ CTF approved! Tx: {tx_hash.hex()[:20]}...")
            except Exception as e:
                print(f"   ❌ CTF approval failed: {e}")
        else:
            print("   ✅ CTF already approved")
    
    print("\n" + "=" * 50)
    print("✅ All allowances set! You can now SELL shares.")
    print("=" * 50)


def main():
    # Load credentials
    priv_key = os.getenv("PRIVATE_KEY")
    pub_key = os.getenv("WALLET_ADDRESS")
    
    if not priv_key or not pub_key:
        print("❌ Error: PRIVATE_KEY and WALLET_ADDRESS must be set in .env")
        return
    
    # Connect to Polygon
    print(f"🔗 Connecting to Polygon...")
    web3 = Web3(Web3.HTTPProvider(RPC_URL))
    
    if not web3.is_connected():
        print("❌ Failed to connect to Polygon RPC")
        return
    
    print(f"✅ Connected! Wallet: {pub_key}")
    
    # Check current status
    all_ok = check_allowances(web3, pub_key)
    
    if all_ok:
        print("\n✅ All allowances already set! No action needed.")
        return
    
    # Ask for confirmation
    print("\n⚠️  Some allowances need to be set.")
    print("This will send transactions to Polygon (gas cost ~$0.01-0.05)")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != "y":
        print("Cancelled.")
        return
    
    # Set allowances
    set_allowances(web3, priv_key, pub_key)
    
    # Verify
    print("\n🔍 Verifying...")
    check_allowances(web3, pub_key)


if __name__ == "__main__":
    main()
