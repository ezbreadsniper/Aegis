#!/usr/bin/env python3
"""Approve NEG_RISK_ADAPTER specifically."""
from web3 import Web3
import os
import json
from dotenv import load_dotenv

load_dotenv()

web3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
priv_key = os.getenv('PRIVATE_KEY')
account = web3.eth.account.from_key(priv_key)
pub_key = account.address

CTF = '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045'
ADAPTER = '0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296'

ABI = json.loads('[{"inputs": [{ "internalType": "address", "name": "operator", "type": "address" },{ "internalType": "bool", "name": "approved", "type": "bool" }],"name": "setApprovalForAll","outputs": [],"stateMutability": "nonpayable","type": "function"}]')

ctf = web3.eth.contract(address=CTF, abi=ABI)

print(f'Connected: {web3.is_connected()}')
print(f'Wallet: {pub_key}')
print(f'Approving NEG_RISK_ADAPTER...')

nonce = web3.eth.get_transaction_count(pub_key)
print(f'Nonce: {nonce}')

txn = ctf.functions.setApprovalForAll(ADAPTER, True).build_transaction({
    'chainId': 137,
    'from': pub_key,
    'nonce': nonce,
    'gas': 100000,
    'gasPrice': web3.eth.gas_price,
})

print('Signing...')
signed = web3.eth.account.sign_transaction(txn, private_key=priv_key)

print('Sending...')
tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
print(f'Tx: {tx_hash.hex()}')

receipt = web3.eth.wait_for_transaction_receipt(tx_hash, 120)
print(f'Block: {receipt["blockNumber"]} Status: {receipt["status"]}')
print('Done!')
