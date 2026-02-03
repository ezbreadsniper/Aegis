#!/usr/bin/env python3
"""Simple allowance check - ASCII only output."""
from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

web3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
pub_key = os.getenv('WALLET_ADDRESS')
CTF = '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045'

EXCHANGES = [
    ('CTF_EXCHANGE', '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E'),
    ('NEG_RISK_CTF', '0xC5d563A36AE78145C45a50134d48A1215220f80a'),
    ('NEG_RISK_ADAPTER', '0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296'),
]

ABI = [{
    'constant': True,
    'inputs': [
        {'name': 'owner', 'type': 'address'},
        {'name': 'operator', 'type': 'address'}
    ],
    'name': 'isApprovedForAll',
    'outputs': [{'name': '', 'type': 'bool'}],
    'type': 'function'
}]

ctf = web3.eth.contract(address=CTF, abi=ABI)

print('CTF ALLOWANCE STATUS')
print('-' * 40)
all_ok = True
for name, addr in EXCHANGES:
    approved = ctf.functions.isApprovedForAll(pub_key, addr).call()
    status = 'APPROVED' if approved else 'NOT APPROVED'
    print(f'{name}: {status}')
    if not approved:
        all_ok = False

print('-' * 40)
if all_ok:
    print('ALL APPROVED - SELL SHOULD WORK')
else:
    print('SOME MISSING')
