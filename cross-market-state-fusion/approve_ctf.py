#!/usr/bin/env python3
"""
Set CTF allowances for Polymarket SELL orders.
Based on: https://github.com/Polymarket/agents/blob/main/agents/polymarket/polymarket.py
"""
import os
import json
import time
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# Config
POLYGON_RPC = "https://polygon-rpc.com"
CHAIN_ID = 137
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Connect
web3 = Web3(Web3.HTTPProvider(POLYGON_RPC))

# Get public key from private key
account = web3.eth.account.from_key(PRIVATE_KEY)
PUB_KEY = account.address

# Addresses
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Exchanges to approve
EXCHANGES = [
    ("CTF_EXCHANGE", "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
    ("NEG_RISK_CTF_EXCHANGE", "0xC5d563A36AE78145C45a50134d48A1215220f80a"),
    ("NEG_RISK_ADAPTER", "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"),
]

# ABIs
CTF_WRITE_ABI = json.loads('[{"inputs": [{"internalType": "address", "name": "operator", "type": "address"},{"internalType": "bool", "name": "approved", "type": "bool"}], "name": "setApprovalForAll", "outputs": [], "stateMutability": "nonpayable", "type": "function"}]')

CTF_READ_ABI = json.loads('[{"constant": true, "inputs": [{"name": "owner", "type": "address"},{"name": "operator", "type": "address"}], "name": "isApprovedForAll", "outputs": [{"name": "", "type": "bool"}], "type": "function"}]')

ctf_write = web3.eth.contract(address=CTF_ADDRESS, abi=CTF_WRITE_ABI)
ctf_read = web3.eth.contract(address=CTF_ADDRESS, abi=CTF_READ_ABI)

print("=" * 60)
print("POLYMARKET CTF ALLOWANCE SETUP")
print("=" * 60)
print(f"Connected: {web3.is_connected()}")
print(f"Wallet: {PUB_KEY}")
print(f"Balance: {web3.eth.get_balance(PUB_KEY) / 1e18:.4f} MATIC")
print("-" * 60)

# Check and set each exchange
all_approved = True
for name, exchange_addr in EXCHANGES:
    print(f"\n[{name}]")
    
    # Check if already approved
    already_approved = ctf_read.functions.isApprovedForAll(PUB_KEY, exchange_addr).call()
    
    if already_approved:
        print("  Status: Already APPROVED (skipping)")
        continue
    
    print("  Status: NOT APPROVED")
    print("  Action: Setting approval...")
    
    try:
        # Get fresh nonce
        nonce = web3.eth.get_transaction_count(PUB_KEY)
        gas_price = web3.eth.gas_price
        
        print(f"  Nonce: {nonce}, Gas Price: {gas_price / 1e9:.1f} Gwei")
        
        # Build transaction
        txn = ctf_write.functions.setApprovalForAll(
            exchange_addr, True
        ).build_transaction({
            "chainId": CHAIN_ID,
            "from": PUB_KEY,
            "nonce": nonce,
            "gas": 60000,  # CTF approval needs ~46k gas
            "gasPrice": gas_price,
        })
        
        # Sign and send
        signed = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  Tx Hash: {tx_hash.hex()}")
        
        # Wait for confirmation
        print("  Waiting for confirmation...")
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        if receipt["status"] == 1:
            print(f"  Result: SUCCESS (Block: {receipt['blockNumber']})")
        else:
            print(f"  Result: FAILED (reverted)")
            all_approved = False
        
        # Wait a bit before next transaction
        time.sleep(2)
        
    except Exception as e:
        print(f"  ERROR: {str(e)[:100]}")
        all_approved = False

# Final verification
print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

final_all_ok = True
for name, exchange_addr in EXCHANGES:
    approved = ctf_read.functions.isApprovedForAll(PUB_KEY, exchange_addr).call()
    status = "APPROVED" if approved else "NOT APPROVED"
    symbol = "[OK]" if approved else "[X]"
    print(f"{symbol} {name}: {status}")
    if not approved:
        final_all_ok = False

print("-" * 60)
if final_all_ok:
    print("SUCCESS! All CTF allowances set. SELL orders will work.")
else:
    print("WARNING: Some allowances missing. SELL may fail.")
print("=" * 60)
